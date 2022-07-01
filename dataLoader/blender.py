import json
import os
import torch
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from .ray_utils import *


class BlenderDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False,
                 n_vis=-1):
        self.root_dir = datadir
        self.split = split
        self.downsample = downsample
        self.is_stack = is_stack
        self.n_vis = n_vis

        self.img_wh = (int(800/downsample), int(800/downsample))
        self.transform = T.ToTensor()

        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array([[ 1, 0, 0, 0],
                                        [ 0,-1, 0, 0],
                                        [ 0, 0,-1, 0],
                                        [ 0, 0, 0, 1]])
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = True
        self.near_far = [2.0, 6.0]
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    def read_meta(self):
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"),
                  'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        # original focal length
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])
        # modify focal length to match size self.img_wh
        self.focal *= self.img_wh[0] / 800 

        # ray directions for all pixels, same for all images (same H, W, focal)
        # (h, w, 3)
        self.directions = get_ray_directions(h, w, [self.focal,self.focal])
        self.directions = self.directions \
                        / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal, 0, w/2],
                                        [0, self.focal, h/2],
                                        [0, 0, 1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []
        self.downsample = 1.0

        img_eval_interval = 1 if self.n_vis < 0 else len(self.meta['frames']) // self.n_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))

        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):
            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            
            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        self.poses = torch.stack(self.poses)
        if not self.is_stack:
            # (len(self.meta['frames'])*h*w, 3)
            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
        else:
            # (len(self.meta['frames']), h*w, 3)
            self.all_rays = torch.stack(self.all_rays, 0)
            self.all_rgbs = torch.stack(self.all_rgbs, 0) \
                                 .reshape(-1, *self.img_wh[::-1], 3)
 
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) \
                      @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        if self.split == 'train':  # use data in the buffers
            return self.all_rays[idx], self.all_rgbs[idx]
            # sample = {'rays': self.all_rays[idx], 'rgbs': self.all_rgbs[idx]}
        else:  # create data for each image separately
            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx] # for quantity evaluation
            return rays, img, mask
        #     sample = {'rays': rays, 'rgbs': img, 'mask': mask}
        # return sample

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32) # (800, 800)
        return depth

