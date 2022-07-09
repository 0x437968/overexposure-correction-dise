from torch.utils.data import Dataset
import os
import os.path
import utils
import random
import torch
import re 
import torchvision.transforms as transforms
from LECRM import CameraModel

class Project_Dataset(Dataset):
    def __init__(self, opt):
        self.patch_size=opt.patch_size
        self.opt=opt
        self.train=opt.isTrain 
        self.path = self.makelist()

    def __getitem__(self, index):
        data=dict()        
        if self.train:
            if self.opt.model == 'dise':
                im_path= self.path[index]
                ext=os.path.splitext(im_path)[-1]
                n1=os.path.basename(im_path).replace(ext,'_1'+ext)
                n2=os.path.basename(im_path).replace(ext,'_2'+ext)
                n3=os.path.basename(im_path).replace(ext,'_3'+ext)
                im = utils.load_image(im_path)
                if self.patch_size:
                    im=self.crop_images(im)[0]
                model=CameraModel()
                im1 = (model.adjust(im,random.uniform(0.3,0.65)))*2-1.0
                im2 = (model.adjust(im,random.uniform(0.65,0.9)))*2-1.0
                im3 = im*2-1.0
                data['imseq']=[im1,im2,im3]
                data['nameseq']=[n1,n2,n3]
            if self.opt.model == 'reco':
                gt_path, over_path = self.path[index]
                gt = utils.load_image(gt_path)*2-1.0
                over = utils.load_image(over_path)*2-1.0
                n=os.path.basename(gt_path)
                if self.patch_size:
                    gt,over=self.crop_images(gt,over)
                data['gt']=gt
                data['over']=over
                data['name']=n
        else:
            if self.opt.model == 'dise':
                test_path,e_path = self.path[index]
                name = os.path.basename(test_path)
                test = utils.load_image(test_path)*2-1.0
                e= utils.load_image(e_path)*2-1.0
                data['test']=test
                data['expo']=e
                data['name']=name
            if 'reco' in self.opt.model :
                test_path = self.path[index]
                name = os.path.basename(test_path)
                test = utils.load_image(test_path)*2-1.0
                data['test']=test
                data['name']=name
        return data 

    def __len__(self):
        return len(self.path)

    def crop_images(self, *images):
        h = images[0].shape[1]
        w=images[0].shape[2]
        h_most = random.randint(0, h - self.patch_size)
        w_most = random.randint(0, w - self.patch_size)
        cropped = []
        for image in images:
            if type(image) is not list:
                cropped.append(image[:, h_most: h_most + self.patch_size, w_most: w_most + self.patch_size])
            else:
                seq=[]
                for im in image:
                    seq.append(im[:, h_most: h_most + self.patch_size, w_most: w_most + self.patch_size])
                cropped.append(seq)
        return cropped

    def makelist(self):
        pathlist = []
        if self.opt.model=='dise':
            if self.train:
                dir_in = os.path.join(self.opt.data_root, self.opt.dir_in)
                for x in os.listdir(dir_in):
                    pathlist.append(os.path.join(dir_in, x))

            else:
                test_dir = os.path.join(self.opt.data_root, self.opt.dir_in)
                e_path = os.path.join(self.opt.data_root, self.opt.path_e)
                for x in os.listdir(test_dir):
                    pathlist.append([os.path.join(test_dir, x), e_path])
        if self.opt.model=='reco':
            if self.train:
                dir_gt = os.path.join(self.opt.data_root, self.opt.dir_gt)
                dir_in = os.path.join(self.opt.data_root, self.opt.dir_in)
                for x in os.listdir(dir_gt):
                    pathlist.append([os.path.join(dir_gt, x), os.path.join(dir_in, x)])
            else:
                test_dir = os.path.join(self.opt.data_root, self.opt.dir_in)
                for x in os.listdir(test_dir):
                    pathlist.append(os.path.join(test_dir, x))
        return pathlist