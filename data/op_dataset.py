import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, get_half_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import PIL
from pdb import set_trace as st
import random
import torch

class OPDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.paths = make_dataset(self.dir)
        self.paths = sorted(self.paths)
        self.size = len(self.paths)
        self.fineSize = opt.fineSize
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        path = self.paths[index % self.size]
        B_img = Image.open(path).convert('RGB')
                
        w, h = B_img.size
        rw = random.randint(0, w - self.fineSize)
        rh = random.randint(0, h - self.fineSize)
        # print(rw, rh)
        B_img = B_img.crop((rw, rh, rw + self.fineSize, rh + self.fineSize))

        w, h = B_img.size

        rw = random.randint(0, w // 2)
        A_img = B_img.crop((rw, 0, rw + w // 2, self.fineSize))

        # A_img = B_img.crop((w // 4, 0, 3 * (w // 4), self.fineSize))

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        mask = torch.ones(1, B_img.size(1), B_img.size(2))
        mask[0, :, rw:rw + w // 2] = 0

        return {'A': A_img, 'B': B_img,
                'A_paths': path, 'B_paths': path,
                'A_start_point':[(rw, 0)], 'M':mask}

    def __len__(self):
        return self.size

    def name(self):
        return 'OPDataset'


class OPDataset2(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.paths = make_dataset(self.dir)
        self.paths = sorted(self.paths)
        self.size = len(self.paths)
        self.fineSize = opt.fineSize
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        path = self.paths[index % self.size]
        B_img = Image.open(path).convert('RGB')

        w, h = B_img.size
        tr = int(self.fineSize * 1.5)
        # tr = 156
        rw = random.randint(0, w - tr)
        rh = random.randint(0, h - self.fineSize)
        # print(rw, rh)
        B_img = B_img.crop((rw, rh, rw + tr, rh + self.fineSize))

        w, h = B_img.size

        rw = random.randint(0, w - self.fineSize)
        A_img = B_img.crop((rw, 0, rw + self.fineSize, self.fineSize))

        # A_img = B_img.crop((w // 4, 0, 3 * (w // 4), self.fineSize))

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        mask = torch.ones(1, B_img.size(1), B_img.size(2))
        mask[0, :, rw:rw + self.fineSize] = 0

        # mask = torch.zeros(1, B_img.size(1), B_img.size(2))
        # mask[0, :, w // 4: 3 * w // 4] = 1

        return {'A': A_img, 'B': B_img,
                'A_paths': path, 'B_paths': path,
                'A_start_point': [(rw, 0)], 'M': mask}

    def __len__(self):
        return self.size

    def name(self):
        return 'OPDataset2'

class OPDataset3(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.paths = make_dataset(self.dir)
        self.paths = sorted(self.paths)
        self.size = len(self.paths)
        self.fineSize = opt.fineSize
        self.transform = get_transform(opt)

    # def __getitem__(self, index):
    #     path = self.paths[index % self.size]
    #     B_img = Image.open(path).convert('RGB')
    #
    #     w, h = B_img.size
    #     rw = random.randint(0, w - self.fineSize * 2)
    #     rh = random.randint(0, h - 256)
    #     # print(rw, rh)
    #     B_img = B_img.crop((rw, rh, rw + self.fineSize * 2, rh + 256))
    #
    #     w, h = B_img.size
    #
    #     rw = random.randint(0, w - self.fineSize)
    #     A_img = B_img.crop((rw, 0, rw + self.fineSize, 256))
    #
    #     A_img = self.transform(A_img)
    #     B_img = self.transform(B_img)
    #
    #     mask = torch.ones(1, B_img.size(1), B_img.size(2))
    #     mask[0, :, rw:rw + self.fineSize] = 0
    #
    #     return {'A': A_img, 'B': B_img,
    #             'A_paths': path, 'B_paths': path,
    #             'A_start_point': [(rw, 0)], 'M': mask}

    def __getitem__(self, index):
        path = self.paths[index % self.size]
        B_img = Image.open(path).convert('RGB')

        w, h = B_img.size
        rw = random.randint(0, w - self.fineSize * 2)
        B_img = B_img.crop((rw, 0, rw + self.fineSize * 2, h))

        w, h = B_img.size

        rw = random.randint(0, w - self.fineSize)
        A_img = B_img.crop((rw, 0, rw + self.fineSize, h))

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        mask = torch.ones(1, B_img.size(1), B_img.size(2))
        mask[0, :, rw:rw + self.fineSize] = 0

        return {'A': A_img, 'B': B_img,
                'A_paths': path, 'B_paths': path,
                'A_start_point': [(rw, 0)], 'M': mask}

    def __len__(self):
        return self.size

    def name(self):
        return 'OPDataset3'