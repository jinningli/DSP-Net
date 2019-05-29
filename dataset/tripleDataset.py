import os.path
import random
import torchvision.transforms as transforms
import torch
from dataset.baseDataset import BaseDataset
from dataloader.image_folder import get_imglist
from PIL import Image

class TripleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_A = os.path.join(opt.dataroot, opt.phase +'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase +'B')
        self.dir_C = os.path.join(opt.dataroot, opt.phase +'C')

        self.A_paths = sorted(get_imglist(self.dir_A))
        self.B_paths = sorted(get_imglist(self.dir_B))
        self.C_paths = sorted(get_imglist(self.dir_C))

    def __getitem__(self, index):

        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        C_path = self.C_paths[index]

        # load and resize
        A = Image.open(A_path).convert('RGB').resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        B = Image.open(B_path).convert('RGB').resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        C = Image.open(C_path).convert('RGB').resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)


        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        C = transforms.ToTensor()(C)

        # random crop
        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        C = C[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
        C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C)

        # set channel (=3)
        input_nc = self.opt.input_nc
        output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            C = C.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
            tmp = C[0, ...] * 0.299 + C[1, ...] * 0.587 + C[2, ...] * 0.114
            C = tmp.unsqueeze(0)

        return {'A': A, 'B': B, 'C': C,
                'A_paths': A_path, 'B_paths': A_path, 'C_paths': C_path}

    def __len__(self):
        # assert (len(self.A_paths) == len(self.B_paths) and len(self.A_paths) == len(self.C_paths))
        return len(self.A_paths)

    def name(self):
        return 'Triple Dataset'
