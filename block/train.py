import datetime
import json
import os
import random
import sys
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from dataLoader import dataset_dict
from opt import config_parser
from renderer import *
from utils import *
from collections import Counter
from huffman import *
from rle.np_impl import dense_to_rle, rle_length, rle_to_dense
#test!!
import imageio_ffmpeg
import pdb
import math


device = "cuda" if torch.cuda.is_available() else "cpu"

renderer = OctreeRender_trilinear_fast


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.randperm(self.total).to("cuda")   
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


@torch.no_grad()
def export_mesh(args):
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    phasorf = eval(args.model_name)(**kwargs)
    phasorf.load(ckpt)

    alpha,_ = phasorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',
                               bbox=phasorf.aabb.cpu(), level=0.005)

@torch.no_grad()
def save_seperate(args, phasorf):
    """
    Save model whose components of density and those of apperance grid are quantized seperately
    (Total 6 grids are quantized and huffman coded respectively)
    """
    n_block, _, den_chan, _, res_y, res_z = phasorf.den[0].shape
    app_chan = phasorf.app[0].shape[2]
    res_x = phasorf.den[1].shape[3]
    shape_info = torch.IntTensor([n_block, den_chan, app_chan, res_x, res_y, res_z])

    print(phasorf.mask_thres)

    mask_thres_tensor = torch.Tensor([phasorf.mask_thres]).to(device)
    if phasorf.mask_thres == 0.5:
        m_thres = 0
    else:
        m_thres = torch.log((1 - mask_thres_tensor) / mask_thres_tensor) * (-1)

    d = 0; a = 0; d_all = 0; a_all = 0
    for i in range(3):
        phasorf.den[i].requires_grad=False
        phasorf.app[i].requires_grad=False

        masked = phasorf.den_mask[i] < m_thres
        phasorf.den[i][masked] = 0 
        masked = phasorf.app_mask[i] < m_thres
        phasorf.app[i][masked] = 0 

        d += phasorf.den[i].count_nonzero().item()
        a += phasorf.app[i].count_nonzero().item()
        d_all += phasorf.den[i].numel()
        a_all += phasorf.app[i].numel()
    print("=====================================")
    print("At the end of training")
    print(f"den: {((d_all - d) / d_all) * 100:.2f}%, app: {((a_all - a) / a_all) * 100:.2f}% masked")
    print(f"grid size without mask bitmap = {(d + a) * 4 / 1024 / 1024:.2f}MB")
    print("=====================================")
    
    # Encoding
    # 1. Quantization
    '''
    Quantization formular from https://gaussian37.github.io/dl-concept-quantization/    (symmetric, signed)
    To use Daniel's quantization, please 
    comment line 101 ~ 134,         and   line 688 ~ 689    and
    uncomment line 138 ~ 142        and   line 692 ~ 693    (Ctrl + F Daniel)
    '''
    fp_min = torch.zeros((2,3), device=device)  # 0=den 1=app
    fp_max = torch.zeros((2,3), device=device)

    for i in range(3):
        fp_min[0,i] = phasorf.den[i].flatten().min(0)[0]
        fp_max[0,i] = phasorf.den[i].flatten().max(0)[0]
        fp_min[1,i] = phasorf.app[i].flatten().min(0)[0]
        fp_max[1,i] = phasorf.app[i].flatten().max(0)[0]
    
    asymmetric = False
    if asymmetric:
        quant_min = np.iinfo(np.int8).min
        quant_max = np.iinfo(np.int8).max
        print(quant_min, quant_max)
        s = (fp_max - fp_min) / (quant_max - quant_min)
        z = torch.round((fp_max * quant_min - fp_min * quant_max) / (fp_max - fp_min))
    else:
        quant_min = np.iinfo(np.int8).min + 1
        quant_max = np.iinfo(np.int8).max 
        s = (fp_max - fp_min) / (quant_max - quant_min)
        z = torch.zeros_like(s)

    den_quant = []
    app_quant = []
    print(quant_min, quant_max)

    for i in range(3):
        den = torch.round(phasorf.den[i] / s[0,i] + z[0,i]).clamp(min=quant_min, max=quant_max).to(torch.int8)
        app = torch.round(phasorf.app[i] / s[1,i] + z[1,i]).clamp(min=quant_min, max=quant_max).to(torch.int8)
        den_quant.append(den)
        app_quant.append(app)

    for i in range(3):
        phasorf.den[i].data = den_quant[i]
        phasorf.app[i].data = app_quant[i]
    
    ##### Daniel
    # s = torch.zeros((2,3)).to(phasorf.den[0].device)
    # z = torch.zeros_like(s)
    # for i in range(3):
    #     phasorf.den[i].data, s[0, i], z[0, i] = min_max_quantize(phasorf.den[i])
    #     phasorf.app[i].data, s[1, i], z[1, i] = min_max_quantize(phasorf.app[i])
    #####

    # 2. RLE
    den_perm_compressed = []; app_perm_compressed = []
    for i in range(3):
        '''
        Not use zigzag. Works fairly well;;
        To use zigzag scanning, please 
        comment line 154 ~ 159,         and   line 667 ~ 673    and
        uncomment line 162 ~ 169        and   line 676 ~ 684    (Ctrl + F zigzag_scan)
        '''
        den_perm = phasorf.den[i].squeeze().permute(0,2,3,1).flatten().cpu().detach().numpy() 
        app_perm = phasorf.app[i].squeeze().permute(0,2,3,1).flatten().cpu().detach().numpy()  
        rle_den_perm = dense_to_rle(den_perm)
        rle_app_perm = dense_to_rle(app_perm)
        den_perm_compressed.append(rle_den_perm)
        app_perm_compressed.append(rle_app_perm)

        ###### zigzag_scan
        # den_zig = phasorf.den[i].squeeze().permute(0,2,3,1).cpu().detach().numpy() 
        # app_zig = phasorf.app[i].squeeze().permute(0,2,3,1).cpu().detach().numpy() 
        # den_zig = zigzag(den_zig).flatten()
        # app_zig = zigzag(app_zig).flatten()
        # rle_den_perm = dense_to_rle(den_zig)
        # rle_app_perm = dense_to_rle(app_zig)
        # den_perm_compressed.append(rle_den_perm)
        # app_perm_compressed.append(rle_app_perm)
        #######
    
    # 3. Entropy coding
    enc_den = []; enc_app = []
    node_den = []; node_app = []
    huffman_tbl_den = []; huffman_tbl_app = []
    for i in range(3):
        freq_den = dict(Counter(den_perm_compressed[i]))
        freq_den = sorted(freq_den.items(), key=lambda x: x[1], reverse=True)
        _node_den = make_tree(freq_den)  
        _huffman_tbl_den = huffman_code_tree(_node_den)   
        huff_val_den = list(map(_huffman_tbl_den.get, den_perm_compressed[i]))  

        huff_den = ''.join(map(str, huff_val_den))
        enc_den.append(huff_den)
        huffman_tbl_den.append(_huffman_tbl_den)
        node_den.append(_node_den)

        freq_app = dict(Counter(app_perm_compressed[i]))
        freq_app = sorted(freq_app.items(), key=lambda x: x[1], reverse=True)
        _node_app = make_tree(freq_app)  
        _huffman_tbl_app = huffman_code_tree(_node_app)   
        huff_val_app = list(map(_huffman_tbl_app.get, app_perm_compressed[i]))  
        huff_app = ''.join(map(str, huff_val_app))
        enc_app.append(huff_app)
        huffman_tbl_app.append(_huffman_tbl_app)
        node_app.append(_node_app)

    # Store values
    # 1. store grid as byte tensor and mlps
    BIT = 8; grid_size = 0
    enc_tensors = []
    for enc in enc_den + enc_app:
        _len = len(enc)
        total_int = math.ceil(_len / BIT)

        st = 0
        print(_len, total_int)
        out = []
        idx_chunk = torch.split(torch.arange(_len), BIT)
        for i in range(total_int):
            target = enc[st: st+BIT]
            _int = int(target, 2)
            out.append(_int)
            st += BIT
        last_target_len = _len - BIT * (total_int - 1)
        out.append(last_target_len)
        bit2byte = torch.ByteTensor(out)
        enc_tensors.append(bit2byte)
        print(bit2byte.element_size() * bit2byte.numel() / 1024 / 1024, "byte")
        grid_size += bit2byte.element_size() * bit2byte.numel() / 1024 / 1024

    enc_tensors.append(shape_info)
    print(f"Grid(+shape): {grid_size}MB")
    
    save_path = f'{args.basedir}/{args.expname}/model/'
    os.makedirs(save_path, exist_ok=True)
    # save args
    kwargs = phasorf.get_kwargs()
    kwargs.update({"scale": s})
    kwargs.update({"zero": z})

    model_params = {
        "grid": enc_tensors,
        "args": kwargs,
        "net": {
            "basis_mat": phasorf.basis_mat,
            "mlp": phasorf.mlp,
            "renderModule": phasorf.renderModule
        },
        "alpha_params": phasorf.alpha_params,
        "beta": phasorf.beta

    }
    torch.save(model_params, save_path + 'model.pt')
    
    # 3. save nodes
    import pickle
    import gzip
    with gzip.open(save_path + 'gzip_node_den.pickle', 'wb') as f:
        pickle.dump(node_den, f)
        del node_den
    with gzip.open(save_path + 'gzip_node_app.pickle', 'wb') as f:
        pickle.dump(node_app, f)
        del node_app
  

@torch.no_grad()
def save_all(args, phasorf):    
    """
    Quantize all components of density and appearance at once.
    (Total 1 grid is quantized and huffman coded)
    """
    n_block, _, den_chan, _, res_y, res_z = phasorf.den[0].shape
    app_chan = phasorf.app[0].shape[2]
    res_x = phasorf.den[1].shape[3]
    shape_info = torch.IntTensor([n_block, den_chan, app_chan, res_x, res_y, res_z])

    print(phasorf.mask_thres)

    mask_thres_tensor = torch.Tensor([phasorf.mask_thres]).to(device)
    if phasorf.mask_thres == 0.5:
        m_thres = 0
    else:
        m_thres = torch.log((1 - mask_thres_tensor) / mask_thres_tensor) * (-1)

    d = 0; a = 0; d_all = 0; a_all = 0
    for i in range(3):
        phasorf.den[i].requires_grad=False
        phasorf.app[i].requires_grad=False

        masked = phasorf.den_mask[i] < m_thres
        phasorf.den[i][masked] = 0 
        masked = phasorf.app_mask[i] < m_thres
        phasorf.app[i][masked] = 0 

        d += phasorf.den[i].count_nonzero().item()
        a += phasorf.app[i].count_nonzero().item()
        d_all += phasorf.den[i].numel()
        a_all += phasorf.app[i].numel()
    print("=====================================")
    print("At the end of training")
    print(f"den: {((d_all - d) / d_all) * 100:.2f}%, app: {((a_all - a) / a_all) * 100:.2f}% masked")
    print(f"grid size without mask bitmap = {(d + a) * 4 / 1024 / 1024:.2f}MB")
    print("=====================================")
    
    # Encoding
    # 1. Quantization
    mins = torch.zeros((6), device=device)  # 0=den 1=app
    maxs = torch.zeros((6), device=device)

    for i in range(3):
        mins[i] = phasorf.den[i].flatten().min(0)[0]
        maxs[i] = phasorf.den[i].flatten().max(0)[0]
        mins[i+3] = phasorf.app[i].flatten().min(0)[0]
        maxs[i+3] = phasorf.app[i].flatten().max(0)[0]
    
    fp_min = torch.min(mins)
    fp_max = torch.max(maxs)

    print(fp_min, fp_max)
    
    asymmetric = False
    if asymmetric:
        quant_min = np.iinfo(np.int8).min
        quant_max = np.iinfo(np.int8).max
        print(quant_min, quant_max)
        s = (fp_max - fp_min) / (quant_max - quant_min)
        z = torch.round((fp_max * quant_min - fp_min * quant_max) / (fp_max - fp_min))
    else:
        quant_min = np.iinfo(np.int8).min + 1
        quant_max = np.iinfo(np.int8).max 
        s = (fp_max - fp_min) / (quant_max - quant_min)
        z = torch.zeros_like(s)

    den_quant = []
    app_quant = []

    for i in range(3):
        den = torch.round(phasorf.den[i] / s + z).clamp(min=quant_min, max=quant_max).to(torch.int8)
        app = torch.round(phasorf.app[i] / s + z).clamp(min=quant_min, max=quant_max).to(torch.int8)
        den_quant.append(den)
        app_quant.append(app)

    for i in range(3):
        phasorf.den[i].data = den_quant[i]
        phasorf.app[i].data = app_quant[i]

    # 2. RLE
    den_perm_compressed = []; app_perm_compressed = []
    for i in range(3):
        den_perm = phasorf.den[i].squeeze().permute(0,2,3,1).flatten().cpu().detach().numpy() 
        app_perm = phasorf.app[i].squeeze().permute(0,2,3,1).flatten().cpu().detach().numpy()  
        
        rle_den_perm = dense_to_rle(den_perm)
        rle_app_perm = dense_to_rle(app_perm)
        
        den_perm_compressed.append(rle_den_perm)
        app_perm_compressed.append(rle_app_perm)
    
    # 3. Entropy coding
    enc_den = []; enc_app = []
    node_den = []; node_app = []
    huffman_tbl_den = []; huffman_tbl_app = []

    all_rles = den_perm_compressed[0]
    for i in range(2):
        all_rles = np.concatenate((all_rles, den_perm_compressed[i+1]))
    for i in range(3):
        all_rles = np.concatenate((all_rles, app_perm_compressed[i]))
    
    freq = dict(Counter(all_rles))
    freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    _node = make_tree(freq)  
    _huffman_tbl = huffman_code_tree(_node)   
    huff_val = list(map(_huffman_tbl.get, all_rles))  
    huff = ''.join(map(str, huff_val)) 

    # Store values
    # 1. store grid as byte tensor and mlps
    BIT = 8; grid_size = 0
    enc_tensors = []
    _len = len(huff)
    total_int = math.ceil(_len / BIT)

    st = 0
    print(_len, total_int)
    out = []
    idx_chunk = torch.split(torch.arange(_len), BIT)
    for i in range(total_int):
        target = huff[st: st+BIT]
        _int = int(target, 2)
        out.append(_int)
        st += BIT
    last_target_len = _len - BIT * (total_int - 1)  # TODO DO not need to handle last 8 bits seperately
    out.append(last_target_len)
    bit2byte = torch.ByteTensor(out)

    enc_tensors.append(bit2byte)
    print(bit2byte.element_size() * bit2byte.numel() / 1024 / 1024, "byte")
    grid_size += bit2byte.element_size() * bit2byte.numel() / 1024 / 1024

    enc_tensors.append(shape_info)
    print(f"Grid(+shape): {grid_size}MB")
    
    save_path = f'{args.basedir}/{args.expname}/model/'
    os.makedirs(save_path, exist_ok=True)
    # save args
    kwargs = phasorf.get_kwargs()
    kwargs.update({"scale": s})
    kwargs.update({"zero": z})

    # 2. save MLPs  
    model_params = {
        "grid": enc_tensors,
        "args": kwargs,
        "net": {
            "basis_mat": phasorf.basis_mat,
            "mlp": phasorf.mlp,
            "renderModule": phasorf.renderModule
        },
        "alpha_params": phasorf.alpha_params,
        "beta": phasorf.beta

    }
    torch.save(model_params, save_path + 'model.pt')
    
    # 3. save nodes and huffman tables
    import pickle
    import gzip
    with gzip.open(save_path + 'gzip_node.pickle', 'wb') as f:
        pickle.dump(_node, f)
        del node_den

@torch.no_grad()
def save_half(args, phasorf):
    """
    Quantize density grid and appearance grid seperately.
    In other words, 3 components of density grid are quantized and huffman coded together and other 3 components of 
    appearance grid are quantized and huffman coded together.
    (Total 2 grids are quantized and huffman coded respectively)
    """

    n_block, _, den_chan, _, res_y, res_z = phasorf.den[0].shape
    app_chan = phasorf.app[0].shape[2]
    res_x = phasorf.den[1].shape[3]
    shape_info = torch.IntTensor([n_block, den_chan, app_chan, res_x, res_y, res_z])

    mask_thres_tensor = torch.Tensor([phasorf.mask_thres]).to(device)
    if phasorf.mask_thres == 0.5:
        m_thres = 0
    else:
        m_thres = torch.log((1 - mask_thres_tensor) / mask_thres_tensor) * (-1)

    d = 0; a = 0; d_all = 0; a_all = 0
    for i in range(3):
        phasorf.den[i].requires_grad=False
        phasorf.app[i].requires_grad=False

        masked = phasorf.den_mask[i] < m_thres
        phasorf.den[i][masked] = 0 
        masked = phasorf.app_mask[i] < m_thres
        phasorf.app[i][masked] = 0 

        d += phasorf.den[i].count_nonzero().item()
        a += phasorf.app[i].count_nonzero().item()
        d_all += phasorf.den[i].numel()
        a_all += phasorf.app[i].numel()
    print("=====================================")
    print("At the end of training")
    print(f"den: {((d_all - d) / d_all) * 100:.2f}%, app: {((a_all - a) / a_all) * 100:.2f}% masked")
    print(f"grid size without mask bitmap = {(d + a) * 4 / 1024 / 1024:.2f}MB")
    print("=====================================")
    
    # Encoding
    # 1. Quantization
    mins = torch.zeros((2,3), device=device)  # 0=den 1=app
    maxs = torch.zeros((2,3), device=device)
    fp_min = torch.zeros(2, device=device)  # 0 = den, 1 =app
    fp_max = torch.zeros(2, device=device)

    for i in range(3):
        mins[0, i] = phasorf.den[i].flatten().min(0)[0]
        maxs[0, i] = phasorf.den[i].flatten().max(0)[0]
        mins[1, i] = phasorf.app[i].flatten().min(0)[0]
        maxs[1, i] = phasorf.app[i].flatten().max(0)[0]
    
    fp_min[0] = torch.min(mins[0])
    fp_max[0] = torch.max(maxs[0])
    fp_min[1] = torch.min(mins[1])
    fp_max[1] = torch.max(maxs[1])

    asymmetric = False
    if asymmetric:
        quant_min = np.iinfo(np.int8).min
        quant_max = np.iinfo(np.int8).max
        print(quant_min, quant_max)
        s = (fp_max - fp_min) / (quant_max - quant_min)
        z = torch.round((fp_max * quant_min - fp_min * quant_max) / (fp_max - fp_min))
    else:
        quant_min = np.iinfo(np.int8).min + 1
        quant_max = np.iinfo(np.int8).max 
        s = (fp_max - fp_min) / (quant_max - quant_min)
        z = torch.zeros_like(s)

    den_quant = []
    app_quant = []
    print(quant_min, quant_max)

    for i in range(3):
        den = torch.round(phasorf.den[i] / s[0] + z[0]).clamp(min=quant_min, max=quant_max).to(torch.int8)
        app = torch.round(phasorf.app[i] / s[1] + z[1]).clamp(min=quant_min, max=quant_max).to(torch.int8)
        den_quant.append(den)
        app_quant.append(app)

    for i in range(3):
        phasorf.den[i].data = den_quant[i]
        phasorf.app[i].data = app_quant[i]

    # 2. RLE
    den_perm_compressed = []; app_perm_compressed = []
    for i in range(3):
        den_perm = phasorf.den[i].squeeze().permute(0,2,3,1).flatten().cpu().detach().numpy() 
        app_perm = phasorf.app[i].squeeze().permute(0,2,3,1).flatten().cpu().detach().numpy()  
        
        rle_den_perm = dense_to_rle(den_perm)
        rle_app_perm = dense_to_rle(app_perm)
        
        den_perm_compressed.append(rle_den_perm)
        app_perm_compressed.append(rle_app_perm)
    
    # 3. Entropy coding
    all_rles_den = den_perm_compressed[0]
    all_rles_app = app_perm_compressed[0]

    for i in range(2):
        all_rles_den = np.concatenate((all_rles_den, den_perm_compressed[i+1]))
        all_rles_app = np.concatenate((all_rles_app, app_perm_compressed[i+1]))

        freq_den = dict(Counter(all_rles_den))
        freq_den = sorted(freq_den.items(), key=lambda x: x[1], reverse=True)
        _node_den = make_tree(freq_den)  
        _huffman_tbl_den = huffman_code_tree(_node_den)   
        huff_val_den = list(map(_huffman_tbl_den.get, all_rles_den))  
        huff_den = ''.join(map(str, huff_val_den)) 

        freq_app = dict(Counter(all_rles_app))
        freq_app = sorted(freq_app.items(), key=lambda x: x[1], reverse=True)
        _node_app = make_tree(freq_app)  
        _huffman_tbl_app = huffman_code_tree(_node_app)   
        huff_val_app = list(map(_huffman_tbl_app.get, all_rles_app))  
        huff_app = ''.join(map(str, huff_val_app)) 

    # Store values
    # 1. store grid as byte tensor and mlps
    BIT = 8; grid_size = 0
    enc_tensors = []
    for enc in [huff_den, huff_app]:
        _len = len(enc)
        total_int = math.ceil(_len / BIT)

        st = 0
        print(_len, total_int)
        out = []
        idx_chunk = torch.split(torch.arange(_len), BIT)
        for i in range(total_int):
            target = enc[st: st+BIT]
            _int = int(target, 2)
            out.append(_int)
            st += BIT
        last_target_len = _len - BIT * (total_int - 1)
        out.append(last_target_len)
        bit2byte = torch.ByteTensor(out)
        enc_tensors.append(bit2byte)
        print(bit2byte.element_size() * bit2byte.numel() / 1024 / 1024, "byte")
        grid_size += bit2byte.element_size() * bit2byte.numel() / 1024 / 1024

    enc_tensors.append(shape_info)
    print(f"Grid(+shape): {grid_size}MB")
    
    save_path = f'{args.basedir}/{args.expname}/model/'
    os.makedirs(save_path, exist_ok=True)
    # save args
    kwargs = phasorf.get_kwargs()
    kwargs.update({"scale": s})
    kwargs.update({"zero": z})

    # 2. save MLPs  
    model_params = {
        "grid": enc_tensors,
        "args": kwargs,
        "net": {
            "basis_mat": phasorf.basis_mat,
            "mlp": phasorf.mlp,
            "renderModule": phasorf.renderModule
        },
        "alpha_params": phasorf.alpha_params,
        "beta": phasorf.beta

    }
    torch.save(model_params, save_path + 'model.pt')
    
    # 3. save nodes
    import pickle
    import gzip
    with gzip.open(save_path + 'gzip_node_den.pickle', 'wb') as f:
        pickle.dump(_node_den, f)
    with gzip.open(save_path + 'gzip_node_app.pickle', 'wb') as f:
        pickle.dump(_node_app, f)
    
  

@torch.no_grad()
def render_test(args, load_path=None): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logfolder = f'{args.basedir}/{args.expname}'    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{logfolder}/{args.expname}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if load_path == None:
        load_path = f'{args.basedir}/{args.expname}/model/'
    print(load_path)
    model_params = torch.load(load_path + 'model.pt')
    
    kwargs = model_params["args"]

    kwargs.update({'device': device})
    kwargs.update({'logger': logger})
    scale = kwargs["scale"]
    zero = kwargs["zero"]
    print(scale, zero)
    print("=====")
    del(kwargs["scale"])
    del(kwargs["zero"])
    phasorf = eval(args.model_name)(**kwargs)


    # 0. load values
    _grid = model_params["grid"]
    _basis_mat = model_params["net"]["basis_mat"]
    _mlp = model_params["net"]["mlp"]
    _renderModule = model_params["net"]["renderModule"]
    _alpha_params = model_params["alpha_params"]
    _beta = model_params["beta"]
    import gzip
    import pickle
    n_blk, den_chan, app_chan, res_x, res_y, res_z = _grid[-1]

    den_shape = [[n_blk, res_y, res_z, den_chan], [n_blk, res_x, res_z, den_chan], [n_blk, res_x, res_y, den_chan]]
    app_shape = [[n_blk, res_y, res_z, app_chan], [n_blk, res_x, res_z, app_chan], [n_blk, res_x, res_y, app_chan]]

    mode = 'seperate'   # seperate, all, half
    BIT = 8

    if mode == 'seperate':
        with gzip.open(load_path + 'gzip_node_den.pickle', 'rb') as f:
            node_den = pickle.load(f)
        with gzip.open(load_path + 'gzip_node_app.pickle', 'rb') as f:
            node_app = pickle.load(f)
        enc_den = _grid[:3]
        enc_app = _grid[3:6]

        for i in range(3):
            # 0. byte to bit
            enc_den_bit = byte2bit(enc_den[i])
            enc_app_bit = byte2bit(enc_app[i])

            # 1. DeHuffman
            dec_den = decode(node_den[i], enc_den_bit) 
            dec_app = decode(node_app[i], enc_app_bit) 

            # 2. DeRLE & permute
            dense_dec = rle_to_dense(dec_den)   
            dense_dec = dense_dec.reshape(den_shape[i]).transpose(0, 3, 1, 2)  
            decoded_den = torch.Tensor(dense_dec).to(device).unsqueeze(1).unsqueeze(i+3)

            dense_dec = rle_to_dense(dec_app)
            dense_dec = dense_dec.reshape(app_shape[i]).transpose(0, 3, 1, 2)
            decoded_app = torch.Tensor(dense_dec).to(device).unsqueeze(1).unsqueeze(i+3)

            ##### zigzag_scan
            # dense_dec = rle_to_dense(dec_den).reshape(n_blk, den_shape[i][1] * den_shape[i][2], den_chan)   
            # dense_dec = inverse_zigzag(dense_dec, *den_shape[i])
            # dense_dec = dense_dec.transpose(0, 3, 1, 2)  # transpose
            # decoded_den = torch.Tensor(dense_dec).to(device).unsqueeze(1).unsqueeze(i+3)

            # dense_dec = rle_to_dense(dec_app).reshape(n_blk, app_shape[i][1] * app_shape[i][2], app_chan)
            # dense_dec = inverse_zigzag(dense_dec, *app_shape[i])
            # dense_dec = dense_dec.transpose(0, 3, 1, 2)
            # decoded_app = torch.Tensor(dense_dec).to(device).unsqueeze(1).unsqueeze(i+3)
            ######

            # 3. Dequantization and restore grid weights
            phasorf.den[i].data =scale[0, i] * (decoded_den - zero[0, i])
            phasorf.app[i].data =scale[1, i] * (decoded_app - zero[1, i])

            ##### Daniel
            # phasorf.den[i].data = min_max_dequantize(decoded_den, scale[0, i], zero[0, i])
            # phasorf.app[i].data = min_max_dequantize(decoded_app, scale[1, i], zero[1, i])
            #####

    elif mode == 'half':
        with gzip.open(load_path + 'gzip_node_den.pickle', 'rb') as f:
            node_den = pickle.load(f)
        with gzip.open(load_path + 'gzip_node_app.pickle', 'rb') as f:
            node_app = pickle.load(f)
        enc_den = _grid[0]
        enc_app = _grid[1]

        # 0. byte to bit
        enc_den_bit = byte2bit(enc_den)
        enc_app_bit = byte2bit(enc_app)

        # 1. DeHuffman
        dec_den = decode(node_den, enc_den_bit) 
        dec_app = decode(node_app, enc_app_bit) 

        # 2. DeRLE & permute
        dense_dec = rle_to_dense(dec_den)  
        st = 0
        for i in range(3):
            N = np.prod(den_shape[i]).item()
            den_dense_dec = dense_dec[st: st+ N]
            den_dense_dec = den_dense_dec.reshape(den_shape[i]).transpose(0, 3, 1, 2)  # transpose
            decoded_den = torch.Tensor(den_dense_dec).to(device).unsqueeze(1).unsqueeze(i+3)
            # 3. Dequantization and restore grid weights
            phasorf.den[i].data =scale[0] * (decoded_den - zero[1])
            st += N 
            # 3. Dequantization and restore grid weights
        
        dense_dec = rle_to_dense(dec_app)  
        st = 0
        for i in range(3):
            N = np.prod(app_shape[i]).item()
            den_dense_dec = dense_dec[st: st+ N]
            den_dense_dec = den_dense_dec.reshape(app_shape[i]).transpose(0, 3, 1, 2)  # transpose
            decoded_den = torch.Tensor(den_dense_dec).to(device).unsqueeze(1).unsqueeze(i+3)
            # 3. Dequantization and restore grid weights
            phasorf.app[i].data =scale[0] * (decoded_den - zero[1])
            st += N 


    elif mode == 'all':
        with gzip.open(load_path + 'gzip_node.pickle', 'rb') as f:
            node = pickle.load(f)
        enc = _grid[0]
        # 0. byte to bit
        enc_bit = byte2bit(enc) 

        # 1. DeHuffman
        dec = decode(node, enc_bit) 

        # 2. DeRLE & permute
        dense_dec = rle_to_dense(dec)   

        st = 0
        for i in range(3):
            N = np.prod(den_shape[i]).item()
            den_dense_dec = dense_dec[st: st+ N]
            den_dense_dec = den_dense_dec.reshape(den_shape[i]).transpose(0, 3, 1, 2)  # transpose
            decoded_den = torch.Tensor(den_dense_dec).to(device).unsqueeze(1).unsqueeze(i+3)
            # 3. Dequantization and restore grid weights
            phasorf.den[i].data =scale * (decoded_den - zero)
            st += N

        for i in range(3):
            N = np.prod(app_shape[i]).item()
            den_dense_dec = dense_dec[st: st+ N]
            den_dense_dec = den_dense_dec.reshape(app_shape[i]).transpose(0, 3, 1, 2)  # transpose
            decoded_den = torch.Tensor(den_dense_dec).to(device).unsqueeze(1).unsqueeze(i+3)
            # 3. Dequantization and restore grid weights
            phasorf.app[i].data =scale * (decoded_den - zero)
            st += N

    else: 
        raise NotImplementedError

    # 4. restore other weights
    phasorf.basis_mat = _basis_mat
    phasorf.mlp = _mlp
    phasorf.renderModule = _renderModule
    phasorf.alpha_params = _alpha_params
    phasorf.beta = _beta

    # test
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test',
                           downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray
    os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
    PSNRs_test = evaluation(test_dataset, phasorf, args, renderer,
                            f'{logfolder}/{args.expname}/imgs_test_all/',
                            N_vis=-1, N_samples=-1, white_bg=white_bg,
                            ndc_ray=ndc_ray, device=device)
    print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} '
        f'<========================')
    logger.info(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} '
        f'<========================')


def reconstruction(args, return_bbox=False, return_memory=False,
                   bbox_only=False):

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train',
                            downsample=args.downsample_train, is_stack=False)
    test_dataset = dataset(args.datadir, split='test',
                           downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}' \
                    f'{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)
    # save config files
    json.dump(args.__dict__, open(f'{logfolder}/config.json',mode='w'),indent=2)

    import logging
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{logfolder}/{args.expname}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    print(args.expname)


    # init parameters
    if not bbox_only and args.dataset_name == 'blender':
        # use tight bbox pre-extracted and stored in misc.py,
        # which takes 2k iters
        data = args.datadir.split('/')[-1]
        from misc import blender_aabb
        aabb = torch.tensor(blender_aabb[data]).reshape(2,3).to(device)
    else:
        # run bbox from scratch
        aabb = train_dataset.scene_bbox.to(device)

    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    
    import math
    var_split = True if type(args.block_split) is list else False



    adaptive_block = True 
    mask_learning = False
    mask_schedule = True
    entropy_weight = args.entropy_weight
    entropy = torch.zeros(1) # for tqdm

    if mask_learning:
        mask_thres = 0.5
    else:
        mask_thres = 0

    if mask_schedule:
        mask_iter = args.mask_iter
        mask_thres_list = args.mask_thres_list
        assert len(mask_iter) == len(mask_thres_list)
        logger.info(f"masking schedule: {mask_iter},   {mask_thres_list}")

    if adaptive_block:
        ratio = True 
        if ratio:
            bbox_size = aabb[1] - aabb[0]   
            bbox_ind = torch.sort(bbox_size)[1]
            target_block = args.target_block

            _blk_split = [args.block_split[0]]*3
            print(_blk_split)
            smallest_bbox_size = bbox_size[bbox_ind[0]]
            print(bbox_size, smallest_bbox_size)

            tmp = 1
            for i in range(len(bbox_ind)):
                tmp *= (bbox_size[bbox_ind[i]] / smallest_bbox_size)
            
            base_blk_split = math.ceil((target_block / tmp).pow(1/3))
            _blk_split = [base_blk_split]*3
            print(base_blk_split)
            
            for i in range(len(bbox_ind)):  # TODO refactorization required. (Fuse w/ above codes)
                _blk_split[bbox_ind[i]] = int(_blk_split[bbox_ind[i]] * (bbox_size[bbox_ind[i]] / smallest_bbox_size))

            args.block_split = _blk_split
            print(args.block_split)
            logger.info(f"block split: {args.block_split}")
            
        else:
            bbox_size = aabb[1] - aabb[0]
            blk_per_axis = args.block_split[0] + args.block_split[1] + args.block_split[2] 
            max_block = args.block_split[0] * args.block_split[1] * args.block_split[2] 
            total_size = bbox_size.sum().item()
            x_split = int(blk_per_axis * bbox_size[0].item()  / total_size)
            y_split = int(blk_per_axis * bbox_size[1].item()  / total_size)
            z_split = int(max_block / x_split / y_split)
            args.block_split = [x_split, y_split, z_split]
    

    if var_split:
        reso_cur = [math.ceil(reso_cur[i] / args.block_split[i]) * args.block_split[i] for i in range(len(reso_cur))]
    else:
        reso_cur = [math.ceil(reso / args.block_split) * args.block_split for reso in reso_cur]

    logger.info(f"resolution: {reso_cur}")
    nSamples = min(args.nSamples,
                   cal_n_samples(reso_cur,args.step_ratio))

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        kwargs.update({'logger':logger})
        phasorf = eval(args.model_name)(**kwargs)
        phasorf.load(ckpt)

        phasorf.mask_thres = ckpt['mask_thres']
        mask_thres_tensor = torch.Tensor([phasorf.mask_thres]).to(device)
        if phasorf.mask_thres == 0.5:
            m_thres = 0
        else:
            m_thres = torch.log((1 - mask_thres_tensor) / mask_thres_tensor) * (-1)

    else:
        phasorf = eval(args.model_name)(aabb, reso_cur, device,
                    # modeling
                    den_num_comp=args.den_num_comp, 
                    app_num_comp=args.app_num_comp, 
                    app_dim=args.app_dim, 
                    softplus_beta=args.softplus_beta,
                    app_aug=args.app_aug,
                    app_ksize = args.app_ksize,
                    den_ksize = args.den_ksize,
                    alpha_init=args.alpha_init,
                    den_scale=args.den_scale,
                    app_scale=args.app_scale,
                    update_dd=args.update_dd, 
                    # rendering 
                    near_far=near_far,
                    shadingMode=args.shadingMode, 
                    alphaMask_thres=args.alpha_mask_thre, 
                    density_shift=args.density_shift, 
                    distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, 
                    view_pe=args.view_pe, 
                    fea_pe=args.fea_pe, 
                    featureC=args.featureC, 
                    step_ratio=args.step_ratio, 
                    fea2denseAct=args.fea2denseAct,
                    block_split=args.block_split,
                    logger=logger, mask_lr=args.mask_lr, mask=True)

        phasorf.mask_thres = mask_thres
        logger.info(args)

    phasorf.logger = logger
    eps = 1e-4
    phasorf.mask_learning = mask_learning
    
    mask_thres_tensor = torch.Tensor([phasorf.mask_thres]).to(device) # threshold for sigmoided value, not pure x

    if phasorf.mask_thres == 0.5:
        m_thres = 0
    else:
        m_thres = torch.log((1 - mask_thres_tensor) / mask_thres_tensor) * (-1)
    logger.info(f"mask threshold = {phasorf.mask_thres}, m_thres = {m_thres}")
    logger.info(f"split {phasorf.resolution} with {phasorf.block_split} blocks. Block res = {phasorf.block_resolution} and freq is {phasorf.n_freq}. net stride: {phasorf.network_strides[1]}")

    L1_weight = 8e-5
    L2_weight = 1e-6

    grad_vars = phasorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    logger.info(f"lr decay  {args.lr_decay_target_ratio}  {args.lr_decay_iters}")

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))
        

    # linear in logrithmic space
    if upsamp_list:
        N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), 
            np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]

    torch.cuda.empty_cache()
    PSNRs, PSNRs_test = [], [0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray:
        allrays, allrgbs = phasorf.filtering_rays(allrays, allrgbs,
                                                  bbox_only=True)
    allrays = allrays.to(device)
    allrgbs = allrgbs.to(device)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    TV_weight_density = args.TV_weight_density
    TV_weight_app = args.TV_weight_app
    print(f"initial TV_weight density: {TV_weight_density} "
          f"appearance: {TV_weight_app}")
    logger.info(f"initial TV_weight density: {TV_weight_density}  appearance: {TV_weight_app}")

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate)

    phasorf.print_size()

    


    for iteration in pbar:
        torch.cuda.empty_cache()
        phasorf.iter = iteration
        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(
            rays_train, phasorf, chunk=args.batch_size, N_samples=nSamples,
            white_bg=white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

        loss = torch.mean((rgb_map - rgb_train) ** 2)
        # loss
        torch.set_printoptions(precision=10)
        total_loss = loss
        
        if TV_weight_density > 0 and (iteration % args.TV_step == 0):
            
            # # 1. Parseval loss
            TV_weight_density *= lr_factor
            reg = phasorf.Parseval_Loss() * TV_weight_density

            # 2. L1 loss
            # L1_weight *= lr_factor
            # reg = phasorf.L1_loss() * L1_weight

            # # 3. L2 loss
            # L2_weight *= lr_factor
            # reg = phasorf.L2_loss() * L2_weight


            total_loss = total_loss + reg
            summary_writer.add_scalar('train/reg_tv_density',
                                      reg.detach().item(),
                                      global_step=iteration)

        if TV_weight_app > 0:
            TV_weight_app *= lr_factor
            raise NotImplementedError('not implemented')

        if mask_learning:
            mask_loss = ((phasorf.num_unmasked_den + phasorf.num_unmasked_app - phasorf.target_param) ** 2)**(0.5) * 1e-5
            total_loss += mask_loss 

        if entropy_weight > 0:
            entropy = phasorf.Entropy_Loss() * entropy_weight
            # entropy = phasorf.Compressibility_Loss() * entropy_weight
            total_loss = total_loss + entropy
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if mask_learning and iteration % 30 == 0:
            print(f"mask learning: target # param: {phasorf.target_param}, cur unmasked param: {phasorf.num_unmasked_app + phasorf.num_unmasked_den}  cur mask thres: {phasorf.mask_thres}, mask loss:{mask_loss}, {phasorf.mask_thres.grad}, {phasorf.mask_thres.grad_fn}")
            phasorf.logger.info(f"mask learning: target # param: {phasorf.target_param}, cur unmasked param: {phasorf.num_unmasked_app + phasorf.num_unmasked_den}  cur mask thres: {phasorf.mask_thres}, mask loss:{mask_loss}")


        loss = loss.detach().item()

        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1],
                                  global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                f' mse = {loss:.6f} tv_loss = {reg.detach().item():.10f} entropy={entropy.detach().item():.4f}' )

            if iteration % (args.progress_refresh_rate * 10) == 0:
                logger.info(f"Iter {iteration}: {float(np.mean(PSNRs))}   tv_loss = {reg.detach().item():.10f}")
            PSNRs = []

        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            PSNRs_test = evaluation(test_dataset,phasorf, args, renderer, 
                                    f'{logfolder}/imgs_vis/',
                                    prtx=f'{iteration:06d}_',
                                    N_samples=nSamples, N_vis=args.N_vis,
                                    white_bg = white_bg, ndc_ray=ndc_ray, 
                                    compute_extra_metrics=args.compute_extra_metric)
            print(np.mean(PSNRs_test))
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test),
                                      global_step=iteration)
            logger.info(f"Iter {iteration}, test psnr: {np.mean(PSNRs_test)}")


        # # TODO: to accelerate 
        if update_AlphaMask_list is not None \
                and iteration in update_AlphaMask_list:

            # update volume resolution
            # if reso_cur[0] * reso_cur[1] * reso_cur[2] < 256**3:
            #     reso_mask = reso_cur

            # new_aabb = phasorf.updateAlphaMask(phasorf.domain_min, phasorf.domain_max, tuple(reso_mask))
            # print("al mask!!")
            # print(reso_mask, new_aabb)
            # logger.info(f"reso_mask: {reso_mask}")

            # if bbox_only:
            #     return new_aabb

            # if return_bbox:
            #     return (new_aabb[1]-new_aabb[0]).prod().cpu().numpy()

            if iteration == update_AlphaMask_list[0]:
                # use tight aabb already
                # phasorf.shrink(new_aabb)
                if args.TV_weight_density_reset >= 0:
                    TV_weight_density = args.TV_weight_density_reset
                    print(f'TV weight density reset to '
                          f'{args.TV_weight_density_reset}')
                    L1_weight = 4e-5

            # if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
            #     # filter rays outside the bbox
            #     # allrays,allrgbs = phasorf.filtering_rays(allrays,allrgbs, bbox_only=True)   
            #     allrays,allrgbs = phasorf.filtering_rays(allrays,allrgbs)   
            #     trainingSampler = SimpleSampler(allrgbs.shape[0],
            #                                     args.batch_size)
            #     allrays = allrays.cuda()
            #     allrgbs = allrgbs.cuda()

        if upsamp_list is not None and iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            irregular_reso = N_to_reso(n_voxels, phasorf.aabb)
            if var_split:
                reso_cur = [math.ceil(irregular_reso[i] / args.block_split[i]) * args.block_split[i] for i in range(len(irregular_reso))]
            else:
                reso_cur = [math.ceil(reso / args.block_split) * args.block_split for reso in irregular_reso]

            print(f"calculated: {irregular_reso} ---> {reso_cur}")
            nSamples = min(args.nSamples,
                           cal_n_samples(reso_cur, args.step_ratio))
            phasorf.upsample_volume_grid(reso_cur)
            logger.info(f"upsample. reso to {reso_cur}, sample to {nSamples}")

            if args.lr_upsample_reset:  
                print("reset lr to initial")
                logger.info("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio**(iteration/args.n_iters)
                print(f'lr set {lr_scale}')
                logger.info(f'lr set {lr_scale}')
            grad_vars = phasorf.get_optparam_groups(args.lr_init*lr_scale,
                                                    args.lr_basis*lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))


        # print size
        if iteration % 1000 == 0:
            # phasorf.save(f'{logfolder}/{args.expname}.th')
            if mask_learning:
                mask_thres_tensor[0] = phasorf.mask_thres
                if phasorf.mask_thres == 0.5:
                    m_thres = 0
                else:
                    m_thres = torch.log((1 - mask_thres_tensor) / mask_thres_tensor) * (-1)

            numel = sum([p.numel() for p in phasorf.parameters()])
            if hasattr(phasorf, 'den_mask'):
                numel -= sum([m.numel() for m in phasorf.den_mask])
                numel -= sum([m.numel() for m in phasorf.app_mask])

            print(f'Total size: {numel*4/1_048_576:.4f}MB')
            logger.info(f'Total size: {numel*4/1_048_576:.4f}MB')
            if hasattr(phasorf, 'den_mask'):
                reduced = sum([d.numel() * (m < m_thres).float().mean()
                            for d, m in zip(phasorf.den, phasorf.den_mask)]) \
                        + sum([d.numel() * (m < m_thres).float().mean()
                            for d, m in zip(phasorf.app, phasorf.app_mask)])
                print(f'reduced size: {(numel - reduced)*4/1_048_576:.4f}MB')
                logger.info(f'reduced size: {(numel - reduced)*4/1_048_576:.4f}MB')
        
        if not mask_learning and mask_schedule:
            if iteration in mask_iter:
                phasorf.mask_thres =  mask_thres_list.pop(0)
                mask_thres_tensor[0] = phasorf.mask_thres
                if phasorf.mask_thres == 0.5:
                    m_thres = 0
                else:
                    m_thres = torch.log((1 - mask_thres_tensor) / mask_thres_tensor) * (-1)
                logger.info(f"mask threshold = {phasorf.mask_thres}, m_thres = {m_thres}")
                logger.info(f"split {phasorf.resolution} with {phasorf.block_split} blocks. Block res = {phasorf.block_resolution} and freq is {phasorf.n_freq}. net stride: {phasorf.network_strides[1]}")


    save_seperate(args, phasorf)
    # save_all(args, phasorf)
    # save_half(args, phasorf)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    seed = 2020233254
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    args = config_parser()
    print(args)

    if args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)
        render_test(args)

