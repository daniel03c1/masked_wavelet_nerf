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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def split_grid(grid, level):
    if level < 1:
        return np.stack(grid)

    H, W = grid.shape[-2:]
    if H % 2 != 0 or W % 2 != 0:
        raise ValueError("grid dimension is not divisable.")
    
    grid = np.squeeze(cubify(grid, (1, H//2, W//2))) # (C*4, H, W)
    idxs = np.arange(len(grid)) # number of channels

    if level >= 1:
        topleft = split_grid(grid[idxs%4 == 0, ...], level-1)
        others = grid[idxs%4 != 0, ...]
        return topleft, others


def concat_grid(grids):
    if len(grids) < 2:
        raise ValueError("# of girds must be greater than 1.")
    # the highest level of grid
    topleft = grids[-1]
    # high level (small) to low level (large)
    for others in reversed(grids[:-1]):
        # interleave blocks along channel axis
        # [c1_1, c2_1, c2_2, c2_3, c1_2, c2_4, ...]
        (c1, h1, w1), c2 = topleft.shape, others.shape[0]
        temp = np.empty((c1+c2, h1, w1), dtype=topleft.dtype)
        idxs = np.arange(c1+c2)
        temp[idxs%4 == 0] = topleft
        temp[idxs%4 != 0] = others
        # uncubify ((c1+c2), 1, h, w) -> ((c1+c2)//4, h*2, w*2)
        topleft = uncubify(temp[:, None, ...], ((c1+c2)//4, h1*2, w1*2))
    return topleft


def get_levelwise_shape(grids, dwt_level):
    total_shapes = []
    for i in range(3):
        grid = grids[i]
        shape_per_lv = []
        # from low (large) to high (small)
        for j in range(dwt_level):
            # split level
            topleft, others = grid
            # save shape
            shape_per_lv += [others.shape]
            # upgrad grid
            grid = topleft
        # save the last level shape in channel-wise
        shape_per_lv += [topleft.shape]
        total_shapes += [shape_per_lv]
    return total_shapes


def packbits_by_level(grids, dwt_level):
    new_grids = []
    for i in range(3):
        grid = grids[i]
        grid_per_lv = [] # dim: (level+1,)
        # from low (large) to high (small)
        for j in range(dwt_level):
            # split level
            topleft, others = grid
            # save high level feat in channel-wise
            grid_per_lv += [np.packbits(others.transpose(1, 2, 0))]
            # update grid
            grid = topleft
        # save the last level feat in channel-wise
        grid_per_lv += [np.packbits(topleft.transpose(1, 2, 0))]
        new_grids += [grid_per_lv]
    return new_grids


@torch.no_grad()
def compress_dwt(args): # save grid + mask
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
    if "trans_func" in kwargs:
        del kwargs['trans_func']

    # make model
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    # ship to cpu
    tensorf.to('cpu')

    if args.reconstruct_mask:
        # (1) mask reconstruction
        den_plane_mask, den_line_mask = [], []
        app_plane_mask, app_line_mask = [], []
        for i in range(3):
            den_plane_mask += [np.where(tensorf.density_plane[i] != 0, 1, 0)]
            den_line_mask += [np.where(tensorf.density_line[i] != 0, 1, 0)]
            app_plane_mask += [np.where(tensorf.app_plane[i] != 0, 1, 0)]
            app_line_mask += [np.where(tensorf.app_line[i] != 0, 1, 0)]
    else:
        # (1) binarize mask
        den_plane_mask, den_line_mask = [], []
        app_plane_mask, app_line_mask = [], []
        for i in range(3):
            den_plane_mask += [np.where(tensorf.density_plane_mask[i]>=0, 1, 0)]
            den_line_mask += [np.where(tensorf.density_line_mask[i]>=0, 1, 0)]
            app_plane_mask += [np.where(tensorf.app_plane_mask[i]>=0, 1, 0)]
            app_line_mask += [np.where(tensorf.app_line_mask[i]>=0, 1, 0)]


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
    
    # (5) pack bits to byte
    for i in range(3):
        den_plane_mask[i] = np.packbits(den_plane_mask[i])
        den_line_mask[i] = np.packbits(den_line_mask[i])
        app_plane_mask[i] = np.packbits(app_plane_mask[i])
        app_line_mask[i] = np.packbits(app_line_mask[i])

    # (5) RLE masks
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
def decompress_dwt(args):
    # check if ckpt exists
    if not os.path.exists(args.ckpt):
        print("the ckpt path does not exists!")
        return

    # set directory
    root_dir = args.ckpt.split('/')[:-1]
    kwargs_path = os.path.join(*root_dir, 'kwargs.th')
    param_path = os.path.join(*root_dir, 'params.th')

    # load kwargs
    kwargs = torch.load(kwargs_path, map_location='cpu')["kwargs"]

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
    for key in state_keys:
        for length in ckpt["rle_length"][key]:
            masks[key] += [np.unpackbits(rle_to_dense(mask[begin:begin+length]).astype(np.uint8))]
            masks[key][-1][masks[key][-1] == 0] = -1
            begin += length

    # (4) reshape mask
    for key in state_keys:
        for i in range(3):
            shape = ckpt["mask_shape"][key][i]
            masks[key][i] = nn.Parameter(
                torch.from_numpy(masks[key][i]).to(torch.float32).reshape(*shape)
            )
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


@torch.no_grad()
def compress_dwt_levelwise(args):
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
    if "trans_func" in kwargs:
        del kwargs['trans_func']

    # make model
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    # ship to cpu
    tensorf.to('cpu')

    # dictionary keys
    state_keys = ["density_plane", "density_line", "app_plane", "app_line"]

    # ---------------------- feature grid compression ---------------------- #

    if args.reconstruct_mask:
        # (1) mask reconstruction
        den_plane_mask, den_line_mask = [], []
        app_plane_mask, app_line_mask = [], []
        for i in range(3):
            den_plane_mask += [np.where(tensorf.density_plane[i] != 0, 1, 0)]
            den_line_mask += [np.where(tensorf.density_line[i] != 0, 1, 0)]
            app_plane_mask += [np.where(tensorf.app_plane[i] != 0, 1, 0)]
            app_line_mask += [np.where(tensorf.app_line[i] != 0, 1, 0)]
    else:
        # (1) binarize mask
        den_plane_mask, den_line_mask = [], []
        app_plane_mask, app_line_mask = [], []
        for i in range(3):
            den_plane_mask += [np.where(tensorf.density_plane_mask[i]>=0, 1, 0)]
            den_line_mask += [np.where(tensorf.density_line_mask[i]>=0, 1, 0)]
            app_plane_mask += [np.where(tensorf.app_plane_mask[i]>=0, 1, 0)]
            app_line_mask += [np.where(tensorf.app_line_mask[i]>=0, 1, 0)]

    # (2) get non-masked values in the feature grids
    den_plane, den_line = [], []
    app_plane, app_line = [], []
    for i in range(3):
        den_plane += [tensorf.density_plane[i][(den_plane_mask[i][None, ...] == 1)].flatten()]
        den_line += [tensorf.density_line[i][(den_line_mask[i][None, ...] == 1)].flatten()]
        app_plane += [tensorf.app_plane[i][(app_plane_mask[i][None, ...] == 1)].flatten()]
        app_line += [tensorf.app_line[i][(app_line_mask[i][None, ...] == 1)].flatten()]

    # scale & minimum value
    scale = {k: [0]*3 for k in state_keys}
    minvl = {k: [0]*3 for k in state_keys}

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

    # ---------------------- mask compression ---------------------- #

    dwt_level = kwargs["dwt_level"]

    # (5) split by level: (((lv3 topleft, lv3 others), lv2 others), lv1 others)
    for i in range(3):
        den_plane_mask[i] = split_grid(den_plane_mask[i].squeeze(0), level=dwt_level)
        app_plane_mask[i] = split_grid(app_plane_mask[i].squeeze(0), level=dwt_level)

    # mask shape for reconstruction
    mask_shape = {
        "density_plane": get_levelwise_shape(den_plane_mask, dwt_level),
        "density_line": [x.shape for x in den_line_mask],
        "app_plane": get_levelwise_shape(app_plane_mask, dwt_level),
        "app_line": [x.shape for x in app_line_mask]
    }

    # (6) pack bits by level
    den_plane_mask = packbits_by_level(den_plane_mask, dwt_level)
    app_plane_mask = packbits_by_level(app_plane_mask, dwt_level)
    den_line_mask = [np.packbits(den_line_mask[i]) for i in range(3)]
    app_line_mask = [np.packbits(app_line_mask[i]) for i in range(3)]

    # (7) RLE (masks), save rle length
    rle_length = {k: [] for k in state_keys}
    for i in range(3):
        # RLE line
        den_line_mask[i] = dense_to_rle(den_line_mask[i], np.int8).astype(np.int8)
        app_line_mask[i] = dense_to_rle(app_line_mask[i], np.int8).astype(np.int8)
        # save line length
        rle_length["density_line"] += [den_line_mask[i].shape[0]]
        rle_length["app_line"] += [app_line_mask[i].shape[0]]
        # RLE plane container
        den_plane_rle_length = []
        app_plane_rle_length = []
        for j in range(dwt_level+1):
            # RLE plane by level
            den_plane_mask[i][j] = dense_to_rle(den_plane_mask[i][j], np.int8).astype(np.int8)
            app_plane_mask[i][j] = dense_to_rle(app_plane_mask[i][j], np.int8).astype(np.int8)
            # save plane length
            den_plane_rle_length += [den_plane_mask[i][j].shape[0]]
            app_plane_rle_length += [app_plane_mask[i][j].shape[0]]
        rle_length["density_plane"] += [den_plane_rle_length]
        rle_length["app_plane"] += [app_plane_rle_length]
        # concat mask by axis (x, y, z)
        den_plane_mask[i] = np.concatenate(den_plane_mask[i])
        app_plane_mask[i] = np.concatenate(app_plane_mask[i])

    # (8) concatenate masks
    mask = np.concatenate([*den_plane_mask, *den_line_mask, *app_plane_mask, *app_line_mask])

    # (9) Huffman (masks)
    mask, mask_tree = huffman(mask)

    # (10) pack bits (string) to byte, numpy to tensor
    mask = bit2byte(mask)

    # (11) save params
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

    # (12) save kwargs
    kwargs_path = os.path.join(*root_dir, 'kwargs.th')
    torch.save({"kwargs": tensorf.get_kwargs()}, kwargs_path)

    kwargs_size = os.path.getsize(kwargs_path)/1024/1024
    print(f"============> kwargs (mb): {kwargs_size} <============")

    # (13) save alphaMask
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
def decompress_dwt_levelwise(args):
    # check if ckpt exists
    if not os.path.exists(args.ckpt):
        print("the ckpt path does not exists!")
        return

    # set directory
    root_dir = args.ckpt.split('/')[:-1]
    kwargs_path = os.path.join(*root_dir, 'kwargs.th')
    param_path = os.path.join(*root_dir, 'params.th')

    # load kwargs
    kwargs = torch.load(kwargs_path, map_location='cpu')["kwargs"]

    # load checkpoint
    ckpt = torch.load(param_path, map_location='cpu')

    # ---------------------- mask reconstruction ---------------------- #

    # (1) unpack byte to bits
    mask = byte2bit(ckpt["mask"])

    # (2) inverse Huffman
    mask = dehuffman(ckpt["mask_tree"], mask)

    # dictionary keys
    state_keys = ["density_plane", "density_line", "app_plane", "app_line"]

    dwt_level = kwargs["dwt_level"]

    # (3) split mask vector, inverse RLE, and unpack bits
    begin = 0
    masks = OrderedDict({k: [] for k in state_keys})
    for key in state_keys:
        for i in range(3):
            rle_length = ckpt["rle_length"][key][i]
            mask_shape = ckpt["mask_shape"][key][i]
            if key in ["density_plane", "app_plane"]:
                mask_per_lv = []
                # from low level to high level
                for j in range(dwt_level+1):
                    # unpack bits
                    mask_per_lv += [np.unpackbits(rle_to_dense(mask[begin:begin+rle_length[j]]).astype(np.uint8))]
                    # unpack(inv_reshape(inv_transpose(A))) = B
                    # reshape to transposed shape, then transpose
                    c, h, w = mask_shape[j]
                    mask_per_lv[-1] = mask_per_lv[-1].reshape((h, w, c)).transpose(2, 0, 1)
                    mask_per_lv[-1][mask_per_lv[-1] == 0] = -1 # to make masked area zero
                    begin += rle_length[j]
                masks[key] += [mask_per_lv]
            else:
                masks[key] += [np.unpackbits(rle_to_dense(mask[begin:begin+rle_length]).astype(np.uint8))]
                masks[key][-1] = masks[key][-1].reshape(mask_shape)
                masks[key][-1][masks[key][-1] == 0] = -1 # to make masked area zero
                begin += rle_length
    
    # (4) concatenate levelwise masks
    for i in range(3):
        masks["density_plane"][i] = concat_grid(masks["density_plane"][i])[None, ...]
        masks["app_plane"][i] = concat_grid(masks["app_plane"][i])[None, ...]

    # (5) convert dtype: int8 -> float32
    for key in state_keys:
        for i in range(3):
            masks[key][i] = nn.Parameter(
                torch.from_numpy(masks[key][i].astype(np.float32))
            )
        masks[key] = nn.ParameterList(masks[key])

    # ---------------------- grid reconstruction ---------------------- #
    
    # (6) dequantize feature grid
    features = {k: [] for k in masks.keys()}
    for key in features.keys():
        for i in range(3):
            feat = ckpt["feature"][key][i]
            scale = ckpt["scale"][key][i]
            minvl = ckpt["minvl"][key][i]
            features[key] += [nn.Parameter(torch.zeros(masks[key][i].shape))]
            features[key][-1][masks[key][i] == 1] = dequantize_int(feat, scale, minvl)
        features[key] = nn.ParameterList(features[key])

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

    args = config_parser()

    if args.compress:
        if args.compress_levelwise:
            compress_dwt_levelwise(args)
        else:
            compress_dwt(args)

    if args.decompress:
        if args.decompress_levelwise:
            decompress_dwt_levelwise(args)
        else:
            decompress_dwt(args)