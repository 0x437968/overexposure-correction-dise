from torch.utils.data import Dataset
import os
import os.path
import utils
import random
import torch
import re 
import torchvision.transforms as transforms

class project_dataset(Dataset):

    def __init__(self, opt):
        self.patch_size=opt.patch_size
        self.opt=opt
        self.train=opt.isTrain 
        self.path = self.makelist()
        # transform_list=[]
        # # transform_list.append(transforms.Resize(size=self.load_size))
        # transform_list.append(transforms.ToTensor())
        # transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
        # self.trans = transforms.Compose(transform_list) 

    def __getitem__(self, index):
        data=dict()        
        if self.train:
            if self.opt.model == 'dise':
                im1_path,im2_path, im3_path= self.path[index]
                n1=os.path.basename(im1_path)
                n2=os.path.basename(im2_path)
                n3=os.path.basename(im3_path)
                im1 = utils.load_image(im1_path)
                im2 = utils.load_image(im2_path)
                im3 = utils.load_image(im3_path)
                if self.patch_size:
                    im1,im2,im3=self.crop_images(im1,im2,im3)
                data['imseq']=[im1,im2,im3]
                data['nameseq']=[n1,n2,n3]
                
            if self.opt.model == 'reco':
                gt_path, over_path = self.path[index]
                gt = utils.load_image(gt_path)
                over = utils.load_image(over_path)
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
                test = utils.load_image(test_path)
                e= utils.load_image(e_path)
                data['test']=test
                data['expo']=e
                data['name']=name
            if 'reco' in self.opt.model :
                test_path = self.path[index]
                name = os.path.basename(test_path)
                test = utils.load_image(test_path)
                data['test']=test
                data['name']=name
        return data 

    def __len__(self):
        return len(self.path)

    def crop_images(self, *images):
        sizex = images[0].shape[1]
        sizey=images[0].shape[2]
        left_most = random.randint(0, sizex - self.patch_size)
        top_most = random.randint(0, sizey - self.patch_size)
        cropped = []
        for image in images:
            if type(image) is not list:
                cropped.append(image[:, left_most: left_most + self.patch_size, top_most: top_most + self.patch_size])
            else:
                seq=[]
                for im in image:
                    seq.append(im[:, left_most: left_most + self.patch_size, top_most: top_most + self.patch_size])
                cropped.append(seq)
        return cropped

    def makelist(self):
        pathlist = []
        if self.opt.model=='dise':
            if self.train:
                dir_gt = os.path.join(self.opt.data_root, self.opt.dir_gt)
                dir_in = os.path.join(self.opt.data_root, self.opt.dir_in)
                for x in os.listdir(dir_gt):
                    n1 = x.replace('.', '_1.')
                    n2 = x.replace('.', '_2.')
                    n3 = x.replace('.', '_3.')
                    pathlist.append([os.path.join(dir_in, n1),
                                     os.path.join(dir_in, n2),
                                     os.path.join(dir_in, n3)])
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