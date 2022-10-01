import torch
from torch import autograd

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
owow = load(name="asdd", sources=["./cuda/grid_2d.cpp", "./cuda/grid_2d_kernel.cu"])

import pdb
class GridSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, block_x, block_y, block_z, block_coords, book, iter=0):  
        n_block, N, C, IH_x, IW_x = block_x.shape
        _, _, _, IH_y, IW_y = block_y.shape
        _, _, _, IH_z, IW_z = block_z.shape

        IH = torch.Tensor([IH_x, IH_y, IH_z]).to(block_coords)  
        IW = torch.Tensor([IW_x, IW_y, IW_z]).to(block_coords)

        W = 1
        ctx.IW = IW
        ctx.C = C

        block_x = block_x.contiguous() if not block_x.is_contiguous() else block_x
        block_y = block_y.contiguous() if not block_y.is_contiguous() else block_y
        block_z = block_z.contiguous() if not block_z.is_contiguous() else block_z

        input_x = block_x.view(n_block, N, C, -1)    
        input_y = block_y.view(n_block, N, C, -1)    
        input_z = block_z.view(n_block, N, C, -1)    

        
        if block_coords.isnan().count_nonzero() > 0:
            print("st", iter, C)
            pdb.set_trace()

        normalized_coord = owow.normalize(block_coords, IW -1, IH -1)
        
        if normalized_coord.isnan().count_nonzero() > 0:
            print("norm", iter, C)
            pdb.set_trace()
        
        # 2. compute corner indices
        with torch.no_grad():
            corner = owow.get_corner(normalized_coord)   

            
            if corner.isnan().count_nonzero() > 0:
                print("corner", iter, C)
                pdb.set_trace()
        
        # 3. compute weights & get points
        weight = owow.get_weight(normalized_coord, corner)   
        if weight.isnan().count_nonzero() > 0:
            print("weight", iter, C)
            pdb.set_trace()

        point = owow.get_point(weight)
        if point.isnan().count_nonzero() > 0:
            print("point", iter, C)
            pdb.set_trace()
        
        with torch.no_grad():   
            for i in range(3):
                torch.clamp(corner[i,:2,:], 0, IW[i]-1, out=corner[i,:2,:])
                torch.clamp(corner[i,2:,:], 0, IH[i]-1, out=corner[i,2:,:])


        IW_float = IW.float()
        vals = owow.gather(input_x, input_y, input_z, corner, IW_float, book, C)   
        if vals.isnan().count_nonzero() > 0:
            print("vals", iter, C)
            pdb.set_trace()

        intpl = owow.interpolate(vals, point, C)
        if intpl.isnan().count_nonzero() > 0:
            print("intpl", iter, C)
            pdb.set_trace()
        
        ctx.C = C
        ctx.n_block = n_block

        ctx.save_for_backward(corner, point, book, IW, IH)
        return intpl    # N C 1 E
    
    @staticmethod
    def backward(ctx, grad_out):    
        C = ctx.C
        n_block = ctx.n_block
        
        corner, point, book, IW, IH = ctx.saved_tensors


        grad_x, grad_y, grad_z = owow.cell_backward(grad_out, corner, point, IW, IH, C, n_block, book)   

        return grad_x.view(-1, 1, C, int(IH[0].item()), int(IW[0].item())), grad_y.view(-1, 1, C, int(IH[1].item()), int(IW[1].item())), grad_z.view(-1, 1, C, int(IH[2].item()), int(IW[2].item())), None, None, None

