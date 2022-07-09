import torch
import torch.nn as nn
from model.basemodel import BaseModel
from model.network.components import Unet
from model.network.discriminator import ResDiscriminator
from model.loss import AdversarialLoss,scene_loss,vgg_style_loss
import utils
import numpy as np
from PIL import Image

class RECO(BaseModel):
    def __init__(self,opt):
        BaseModel.__init__(self,opt)
        self.net_names=['G']
        self.net_G=Unet(inchannel=3,outchannel=3,
                        ndf=64,enc_blocks=1,dec_blocks=1,depth=3,
                        concat=True,bilinear=self.opt.bilinear,norm_layer='LN')
        if self.isTrain:
            self.loss_names=['G_ad_gen','D','G','G_style','G_scene']
            self.net_names+=['D']
            self.net_D = ResDiscriminator(ndf=32, img_f=128, layers=4,use_spect=True)
            self.GANloss=AdversarialLoss('lsgan').to(self.device)
            self.Styleloss=vgg_style_loss().to(self.device)
            self.sceneloss=scene_loss(self.opt.scenepath).to(self.device)
            
            self.optim_G=torch.optim.Adam(self.net_G.parameters(),lr=opt.lr,betas=(0.0,0.999))
            self.optim_D=torch.optim.Adam(self.net_D.parameters(),lr=opt.lr*opt.lrg2d,betas=(0.0,0.999))

    @staticmethod
    def modify_options(parser):
        opt, _ =parser.parse_known_args()
        parser.add_argument('--bilinear',  action='store_true', help='bool for using bilinear or convtranspose ')
        if opt.phase=='train':
            parser.add_argument('--lr', type=float, default=0.0001, help='learning rate ')
            parser.add_argument('--lrg2d', type=float, default=0.1, help='learning rate ratio G to D')
            parser.add_argument('--gamma_scene', type=float, default=10000.0, help='weight for scene loss')
            parser.add_argument('--gamma_style', type=float, default=2000.0, help='weight for style loss')
            parser.add_argument('--gamma_gen', type=float, default=2.0, help='weight for generation loss')
            parser.add_argument('--scenepath',  default='./checkpoints/net_E_s.path', help='scene encoder path')

            parser.add_argument('--dir_in',  default='over', help='over-exposure images diretory')
            parser.add_argument('--dir_gt',  default='gt', help='ground truth images diretory')
        else:
            parser.add_argument('--dir_in',  default='test', help='test images diretory')
        return parser

    def set_input(self, data):
        if self.isTrain:
            self.gt,self.over,self.n=data['gt'],data['over'],data['name']
            self.gt,self.over=self.gt.to(self.device),self.over.to(self.device)
        else:
            self.over,self.n=data['test'],data['name']
            self.over=self.over.to(self.device)

    def forward(self):
        self.ocim=self.net_G(self.over)

    def backward_D(self):
        utils._unfreeze(self.net_D)
        self.loss_D=self.backward_D_basic(netD=self.net_D,real=self.gt,fake=self.ocim)
        self.D_sumloss.append(self.loss_D.item()) 

    def backward_G(self):
        utils._freeze(self.net_D)
        fake=self.net_D(self.ocim)
        self.loss_G_ad_gen=self.GANloss(fake,True,False)*self.opt.gamma_gen
        self.loss_G_scene=self.sceneloss(self.gt,self.ocim)*self.opt.gamma_scene
        self.loss_G_style=self.Styleloss(self.gt,self.ocim)*self.opt.gamma_style
        self.loss_G=[]
        for loss_name in self.loss_names:
            if 'G_' in loss_name:
                self.loss_G.append(getattr(self,'loss_'+loss_name))
        self.loss_G=sum(self.loss_G)
        #backward
        self.loss_G.backward()
        self.G_sumloss.append(self.loss_G.item())

    def train(self):
        self.forward()
        self.optim_D.zero_grad()
        self.backward_D()
        self.optim_D.step()

        self.optim_G.zero_grad()
        self.backward_G()
        self.optim_G.step()

    def print_loss(self,name,epoch,idx,length):
        print('[%s_%s   ep /%d(%d) it /%d(%d)  g_loss: %.4f  d_loss: %.4f]' % (self.opt.model,name, epoch,self.opt.epochs, idx + 1,length,
                                                                          sum(self.G_sumloss)/len(self.G_sumloss),
                                                                          sum(self.D_sumloss) / len(self.D_sumloss)
                                                                          ))


    def clear_sumloss(self):
        self.D_sumloss=[]
        self.G_sumloss=[]

    def write(self,writer,step):
        loss_dic={}
        for n in self.loss_names:
            loss=getattr(self,'loss_'+n)
            loss_dic['loss_'+n]=loss.item()
        writer.add_scalars('%s_%s_loss'%(self.opt.model,self.opt.name),loss_dic,global_step=step)

    def save_results(self):
        im_save_dir=self.opt.im_save_dir
        if self.isTrain:
            data1 = utils.transform_to_numpy(self.ocim)
            data2=utils.transform_to_numpy(self.over)
            data3=utils.transform_to_numpy(self.gt)
            for k in range(min(data1.shape[0],2)):
                image1 = Image.fromarray(data1[k])
                image1.save( '%s/%s' % (im_save_dir, self.n[k]))
                image2 = Image.fromarray(data2[k])
                image2.save('%s/%s' % (im_save_dir, self.n[k].replace('.','over.')))
                image3 = Image.fromarray(data3[k])
                image3.save('%s/%s' % (im_save_dir, self.n[k].replace('.', 'gt.')))
        else:
            out=utils.transform_to_numpy(self.ocim)
            for k in range(out.shape[0]):
                img=Image.fromarray(out[k])
                img.save( '%s/%s' % (im_save_dir, self.n[k]))

    def backward_D_basic(self,netD,real,fake):
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real,True,True)

        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)

        D_loss=(D_real_loss+D_fake_loss)
        D_loss.backward()

        return D_loss
