import os
import math
from opt import config_parser
from renderer import *
from utils import *
from scan import *
from huffman import *
from run_length_encoding.rle.np_impl import dense_to_rle, rle_to_dense
from collections import OrderedDict
from dataLoader import dataset_dict


def cubify(arr, newshape):
    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)


def uncubify(arr, oldshape):
    N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)    
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)


def bit2byte(enc):
    BIT = 8
    length = len(enc)
    total_int = math.ceil(length/BIT)
    
    start, out = 0, []
    for i in range(total_int):
        target = enc[start:start+BIT]
        out.append(int(target, 2))
        start += BIT

    last_target_length = length - BIT * (total_int - 1)
    out.append(last_target_length)
    enc_byte_tensor = torch.ByteTensor(out)
    return enc_byte_tensor


def byte2bit(bytes):
    bit = []
    bytecode = bytes[:-2]
    for byte in bytecode:
        b = format(byte, '08b')
        bit.append(b)

    last_ele = format(bytes[-2], 'b')   # 이걸 왜 08로 안했지?
    last_tar_len = bytes[-1]
    num_to_add_zeros = last_tar_len - len(last_ele)
    output =''.join(bit) + '0'*num_to_add_zeros + last_ele
    return output


def quantize_float(inputs, bits):
    if bits == 32:
        return inputs
    n = float(2**(bits-1) - 1)
    out = np.floor(np.abs(inputs) * n) / n
    rounded = out * np.sign(inputs)
    return rounded

def quantize_int(inputs, bits):
    if bits == 32:
        return inputs
    minvl = torch.amin(inputs)
    maxvl = torch.amax(inputs)
    scale = (maxvl - minvl).clip(min=1e-8) / (2**bits-2)
    rounded = torch.round((inputs - minvl)/scale) + 1
    return rounded, scale, minvl

def dequantize_int(inputs, scale, minvl):
    return (inputs - 1) * scale + minvl


@torch.no_grad()
def compress_dct(args, device):
    # check if ckpt exists
    if not os.path.exists(args.ckpt):
        print("the ckpt path does not exists!")
        return

    # load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)

    # update kwargs
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})

    # NOTE: temp code
    del kwargs['trans_func']

    # make model
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    # ship to cpu
    tensorf.to('cpu')

    # (1) mask reconstruction
    den_plane_mask, den_line_mask = [], []
    app_plane_mask, app_line_mask = [], []
    for i in range(3):
        den_plane_mask += [np.where(tensorf.density_plane[i] != 0, 1, 0)]
        den_line_mask += [np.where(tensorf.density_line[i] != 0, 1, 0)]
        app_plane_mask += [np.where(tensorf.app_plane[i] != 0, 1, 0)]
        app_line_mask += [np.where(tensorf.app_line[i] != 0, 1, 0)]
    
    # mask shape
    mask_shape = {
        "density_plane": [x.shape for x in den_plane_mask],
        "density_line": [x.shape for x in den_line_mask],
        "app_plane": [x.shape for x in app_plane_mask],
        "app_line": [x.shape for x in app_line_mask]
    }

    # (2) get non-masked values in the feature grids
    den_plane, den_line = [], []
    app_plane, app_line = [], []
    for i in range(3):
        den_plane += [tensorf.density_plane[i][(den_plane_mask[i][None, ...] == 1)].flatten()]
        den_line += [tensorf.density_line[i][(den_line_mask[i][None, ...] == 1)].flatten()]
        app_plane += [tensorf.app_plane[i][(app_plane_mask[i][None, ...] == 1)].flatten()]
        app_line += [tensorf.app_line[i][(app_line_mask[i][None, ...] == 1)].flatten()]

    # scale & minimum value
    scale = {k: [0]*3 for k in mask_shape.keys()}
    minvl = {k: [0]*3 for k in mask_shape.keys()}

    # (3) quantize non-masked values
    for i in range(3):
        den_plane[i], scale["density_plane"][i], minvl["density_plane"][i] = quantize_int(den_plane[i], tensorf.grid_bit)
        den_line[i], scale["density_line"][i], minvl["density_line"][i] = quantize_int(den_line[i], tensorf.grid_bit)
        app_plane[i], scale["app_plane"][i], minvl["app_plane"][i] = quantize_int(app_plane[i], tensorf.grid_bit)
        app_line[i], scale["app_line"][i], minvl["app_line"][i] = quantize_int(app_line[i], tensorf.grid_bit)

    # (4) convert dtype (float -> uint8)
    for i in range(3):
        den_plane[i] = den_plane[i].to(torch.uint8)
        den_line[i] = den_line[i].to(torch.uint8)
        app_plane[i] = app_plane[i].to(torch.uint8)
        app_line[i] = app_line[i].to(torch.uint8)

    # (5) zigzag scan (channel-first)
    for i in range(3):
        den_plane_mask[i] = zigzag_block(den_plane_mask[i].transpose(0, 2, 3, 1))
        app_plane_mask[i] = zigzag_block(app_plane_mask[i].transpose(0, 2, 3, 1))

    # (6) pack bits to byte
    for i in range(3):
        den_plane_mask[i] = np.packbits(den_plane_mask[i])
        den_line_mask[i] = np.packbits(den_line_mask[i])
        app_plane_mask[i] = np.packbits(app_plane_mask[i])
        app_line_mask[i] = np.packbits(app_line_mask[i])

    # (7) RLE masks
    for i in range(3):
        den_plane_mask[i] = dense_to_rle(den_plane_mask[i].flatten(), np.int8).astype(np.int8)
        den_line_mask[i] = dense_to_rle(den_line_mask[i].flatten(), np.int8).astype(np.int8)
        app_plane_mask[i] = dense_to_rle(app_plane_mask[i].flatten(), np.int8).astype(np.int8)
        app_line_mask[i] = dense_to_rle(app_line_mask[i].flatten(), np.int8).astype(np.int8)
    
    # (6) concatenate masks
    mask = np.concatenate([*den_plane_mask, *den_line_mask, *app_plane_mask, *app_line_mask])
    rle_length = {
        "density_plane": [r.shape[0] for r in den_plane_mask],
        "density_line": [r.shape[0] for r in den_line_mask],
        "app_plane": [r.shape[0] for r in app_plane_mask],
        "app_line": [r.shape[0] for r in app_line_mask]
    }

    # (7) Huffman masks
    mask, mask_tree = huffman(mask)

    # (8) bit -> byte, numpy -> tensor
    mask = bit2byte(mask)
    # mask = torch.ByteTensor(np.packbits(np.array(list(mask), np.uint8)))

    # (9) save params
    params = {
        "feature": {
            "density_plane": den_plane,
            "density_line": den_line,
            "app_plane": app_plane,
            "app_line": app_line
        },
        "scale": scale,
        "minvl": minvl,
        "mask": mask,
        "mask_tree": mask_tree,
        "mask_shape": mask_shape,
        "rle_length": rle_length,
        "render_module": tensorf.renderModule,
        "basis_mat": tensorf.basis_mat
    }

    # set directory
    root_dir = args.ckpt.split('/')[:-1]
    param_path = os.path.join(*root_dir, 'params.th')
    torch.save(params, param_path)

    param_size = os.path.getsize(param_path)/1024/1024
    print(f"============> Grid + Mask + MLP (mb): {param_size} <============")

    # (10) save kwargs
    kwargs_path = os.path.join(*root_dir, 'kwargs.th')
    torch.save({"kwargs": tensorf.get_kwargs()}, kwargs_path)

    kwargs_size = os.path.getsize(kwargs_path)/1024/1024
    print(f"============> kwargs (mb): {kwargs_size} <============")

    if tensorf.alphaMask is not None:
        alpha_volume = tensorf.alphaMask.alpha_volume.bool().cpu().numpy()
        alpha_mask = {
            'alphaMask.shape': alpha_volume.shape,
            'alphaMask.mask': np.packbits(alpha_volume.reshape(-1)),
            'alphaMask.aabb': tensorf.alphaMask.aabb.cpu()
        }

        alpha_mask_path = os.path.join(*root_dir, 'alpha_mask.th')
        torch.save(alpha_mask, alpha_mask_path)

        mask_size = os.path.getsize(alpha_mask_path)/1024/1024
        print(f"============> Alpha mask (mb): {mask_size} <============")

    print("encoding done.")


@torch.no_grad()
def decompress_dct(args):
    # check if ckpt exists
    if not os.path.exists(args.ckpt):
        print("the ckpt path does not exists!")
        return

    # set directory
    root_dir = args.ckpt.split('/')[:-1]
    param_path = os.path.join(*root_dir, 'params.th')

    # load checkpoint
    ckpt = torch.load(param_path, map_location='cpu')

    # dictionary keys
    state_keys = ["density_plane", "density_line", "app_plane", "app_line"]

    # (1) byte -> bit
    mask = byte2bit(ckpt["mask"])
    # mask = np.unpackbits(ckpt["mask"].numpy())

    # (2) inverse huffman
    mask = dehuffman(ckpt["mask_tree"], mask)

    # (3) split an array into multiple arrays and inverse RLE
    masks = OrderedDict({k: [] for k in state_keys})

    begin = 0
    for key in masks.keys():
        for length in ckpt["rle_length"][key]:
            masks[key] += [np.unpackbits(rle_to_dense(mask[begin:begin+length]).astype(np.uint8))]
            masks[key][-1][masks[key][-1] == 0] = -1
            begin += length

    # (4) inverse zigzag and reshape
    for key in state_keys:
        for i in range(3):
            B, C, H, W = ckpt["mask_shape"][key][i]
            if key in ["density_plane", "app_plane"]:
                mask = inverse_zigzag_block(masks[key][i].reshape(B, H*W, C), B, H, W, C).transpose(0, 3, 1, 2)
            else:
                mask = masks[key][i].reshape((B, C, H, W))
            masks[key][i] = nn.Parameter(torch.from_numpy(mask).to(torch.float32))
        masks[key] = nn.ParameterList(masks[key])

    # (5) dequantize feature grid
    features = {k: [] for k in state_keys}
    for key in features.keys():
        for i in range(3):
            feat = ckpt["feature"][key][i]
            scale = ckpt["scale"][key][i]
            minvl = ckpt["minvl"][key][i]
            features[key] += [nn.Parameter(torch.zeros(ckpt["mask_shape"][key][i]))]
            features[key][-1][masks[key][i] == 1] = dequantize_int(feat, scale, minvl)
        features[key] = nn.ParameterList(features[key])
    
    # load kwargs
    kwargs_path = os.path.join(*root_dir, 'kwargs.th')
    kwargs = torch.load(kwargs_path, map_location='cpu')["kwargs"]

    # check kwargs
    kwargs.update({'device': device})

    # IMPORTANT: aabb to cuda
    kwargs["aabb"] = kwargs["aabb"].to(device)

    # load params
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.density_plane = features["density_plane"].to(device)
    tensorf.density_line = features["density_line"].to(device)
    tensorf.app_plane = features["app_plane"].to(device)
    tensorf.app_line = features["app_line"].to(device)
    tensorf.density_plane_mask = masks["density_plane"].to(device)
    tensorf.density_line_mask = masks["density_line"].to(device)
    tensorf.app_plane_mask = masks["app_plane"].to(device)
    tensorf.app_line_mask = masks["app_line"].to(device)
    tensorf.renderModule = ckpt["render_module"].to(device)
    tensorf.basis_mat = ckpt["basis_mat"].to(device)

    # load alpha mask
    alpha_mask_path = os.path.join(*root_dir, 'alpha_mask.th')
    if os.path.exists(alpha_mask_path):
        print("loading alpha mask...")
        alpha_mask = torch.load(alpha_mask_path, map_location=device)
        length = np.prod(alpha_mask['alphaMask.shape'])
        alpha_volume = torch.from_numpy(np.unpackbits(alpha_mask['alphaMask.mask'])[:length].reshape(alpha_mask['alphaMask.shape']))
        tensorf.alphaMask = AlphaGridMask(device, alpha_mask['alphaMask.aabb'].to(device), alpha_volume.float().to(device))

    print("model loaded.")

    if args.decompress_and_validate:
        # renderder
        renderer = OctreeRender_trilinear_fast

        # init dataset
        dataset = dataset_dict[args.dataset_name]
        test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)

        white_bg = test_dataset.white_bg
        ndc_ray = args.ndc_ray

        logfolder = os.path.dirname(args.ckpt)

        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=args.N_vis, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'============> {args.expname} test all psnr: {np.mean(PSNRs_test)} <============')



if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = config_parser()

    if args.compress:
        compress_dct(args, device)

    if args.decompress:
        decompress_dct(args)