import torch
import torch.nn as nn
from model.basemodel import BaseModel
from model.network.components import Encoder_S,Encoder_E,Decoder_SE
from model.loss import vgg_style_loss
import utils
import numpy as np
from PIL import Image

class DISE(BaseModel):
    def __init__(self,opt):
        BaseModel.__init__(self,opt)
        self.net_names=['E_s','E_e','G_se']
        self.net_E_s=Encoder_S(n_downsample=3,ndf=64,norm_layer='LN')
        self.net_E_e=Encoder_E(inc=3,n_downsample=4,outc=self.opt.dimen_e,ndf=64,usekl=opt.kl)
        self.net_G_se=Decoder_SE(s_inc=self.net_E_s.outc,e_inc=self.opt.dimen_e,
                                        n_upsample=self.net_E_s.n_downsample,norm_layer='LN')
        if self.isTrain:
            self.l1loss=nn.L1Loss()
            self.set_Adam_optims(self.net_names)
            self.Styleloss=vgg_style_loss().to(self.device)
            self.loss_names=['kl','style','recon','ident','total']

    @staticmethod
    def modify_options(parser):
        opt, _ =parser.parse_known_args()
        parser.add_argument('--usekl', action='store_false', help='bool for kl loss in E_e') 
        parser.add_argument('--dimen_e', type=int,default=8, help='dimension for exposure vector')
        if opt.phase=='train':
            parser.add_argument('--lr', type=float, default=0.0001, help='learning rate ')
            parser.add_argument('--lambda_recon', type=float, default=50.0, help='weight for recon loss')
            parser.add_argument('--lambda_kl', type=float, default=0.01, help='weight for kl loss')
            parser.add_argument('--lambda_style', type=float, default=1000.0, help='weight for style loss')
            parser.add_argument('--lambda_ident', type=float, default=1000.0, help='weight for scene loss')
            parser.add_argument('--dir_in',  default='train', help='sequence images diretory')
            parser.add_argument('--dir_gt',  default='gt', help='ground truth images diretory')
        else:
            parser.add_argument('--dir_in',  default='testgt', help='scene images diretory')
            parser.add_argument('--path_e',  default='', help='exposure image relative path')
        return parser

    def set_input(self, data):
        if self.isTrain:
            self.im1,self.im2,self.im3=data['imseq']
            self.n1,self.n2,self.n3=data['nameseq']

            self.im1=self.im1.to(self.device)
            self.im2=self.im2.to(self.device)
            self.im3=self.im3.to(self.device)
        else:
            self.test,self.e,self.n=data['test'],data['expo'],data['name']
            self.test=self.test.to(self.device)
            self.e=self.e.to(self.device)

    def forward(self):
        if self.isTrain:
            self.s1 = self.net_E_s(self.im1)
            self.s2 = self.net_E_s(self.im2)
            self.s3=self.net_E_s(self.im3)

            if self.opt.usekl:
                self.mean1, self.var1 = self.net_E_e(self.im1)
                self.std1 = torch.exp(self.var1*0.5)
                sample_z1 =torch.randn(self.mean1.size(0),self.mean1.size(1)).to(self.device)
                e1 =sample_z1*self.std1+self.mean1

                self.mean2, self.var2 = self.net_E_e(self.im2)
                self.std2 = torch.exp(self.var2 * 0.5)
                sample_z2 = torch.randn(self.mean2.size(0), self.mean2.size(1)).to(self.device)
                e2 = sample_z2 * self.std2 + self.mean2

                self.mean3, self.var3 = self.net_E_e(self.im3)
                self.std3 = torch.exp(self.var3 * 0.5)
                sample_z3 = torch.randn(self.mean3.size(0), self.mean3.size(1)).to(self.device)
                e3 = sample_z3 * self.std3 + self.mean3
            else:
                e1=self.net_E_e(self.im1)
                e2=self.net_E_e(self.im2)
                e3=self.net_E_e(self.im3)

            self.out11=self.net_Dec(self.s2,e1)
            self.out22=self.net_Dec(self.s2,e2)
            self.out33=self.net_Dec(self.s2,e3)

        else:
            self.s=self.net_E_s(self.test)
            if self.opt.usekl:
                sample_z = torch.randn(self.s.size(0), 8).to(self.device)
                mean,var=self.net_E_e(self.e)
                std=torch.exp(var*0.5)
                e = sample_z* std + mean
            else:
                e=self.net_E_e(self.e)
            self.out=self.net_Dec(self.s,e)

    def backward_(self):
        #scene constraint
        s_avr=(self.s1+self.s2+self.s3)/3
        self.loss_ident=(self.l1loss(self.s1,s_avr)+self.l1loss(self.s2,s_avr)+self.l1loss(self.s3,s_avr))*self.opt.lambda_ident
        #style loss
        self.loss_style=(self.Styleloss(self.out11,self.im1)+self.Styleloss(self.out33,self.im3)+self.Styleloss(self.out22,self.im2))*self.opt.lambda_style
        #reconstruction loss
        self.loss_recon=(self.l1loss(self.out11,self.im1)+self.l1loss(self.out33,self.im3)+self.l1loss(self.out22,self.im2))*self.opt.lambda_recon
        #kl loss
        if self.opt.usekl:
            loss_kl1 = torch.sum(self.mean1**2+torch.exp(self.var1)-self.var1-1.0)*0.5
            loss_kl2=torch.sum(self.mean2**2+torch.exp(self.var2)-self.var2-1.0)*0.5
            loss_kl3=torch.sum(self.mean3**2+torch.exp(self.var3)-self.var3-1.0)*0.5
            self.loss_kl = (loss_kl1+loss_kl3+loss_kl2) * self.opt.lambda_kl
        else:
            self.loss_kl=torch.tensor(0).to(self.device)
        #total loss
        self.loss_total=[]
        for loss_name in self.loss_names:
            if 'total' not in loss_name:
                self.loss_total.append(getattr(self,'loss_'+loss_name))
        self.loss_total=sum(self.loss_total)

        self.loss_total.backward()
        self.sumloss.append(self.loss_total.item())

    def train(self):
        self.forward()

        self.optim_E_s.zero_grad()
        self.optim_E_e.zero_grad()
        self.optim_Dec.zero_grad()
        self.backward_()
        self.optim_Dec.step()
        self.optim_E_s.step()
        self.optim_E_e.step()

    def print_loss(self,name,epoch,idx,length):
        print('[%s_%s   ep /%d(%d) it /%d(%d)  loss: %.4f]' % (self.opt.model,name, epoch,self.opt.epochs, idx + 1,length,
                                                                          sum(self.sumloss)/len(self.sumloss),
                                                                          ))


    def clear_sumloss(self):
        self.sumloss=[]

    def write(self,writer,step):
        loss_dic={}
        for n in self.loss_names:
            loss=getattr(self,'loss_'+n)
            loss_dic['loss_'+n]=loss.item()
        writer.add_scalars('%s_%s_loss'%(self.opt.model,self.opt.name),loss_dic,global_step=step)

    def save_results(self):
        im_save_dir=self.opt.im_save_dir
        if self.isTrain:
            gt1 = utils.transform_to_numpy(self.im1)
            gt2 = utils.transform_to_numpy(self.im2)
            gt3 = utils.transform_to_numpy(self.im3)
            data1 = utils.transform_to_numpy(self.out11)
            data2 = utils.transform_to_numpy(self.out22)
            data3 = utils.transform_to_numpy(self.out33)
            for k in range(min(gt1.shape[0],2)):
                img1 = Image.fromarray(gt1[k])
                img1.save('%s/%s' % (im_save_dir, self.n1[k].replace('.', 'gt.') ))
                img2 = Image.fromarray(gt2[k])
                img2.save('%s/%s' % (im_save_dir, self.n2[k].replace('.', 'gt.') ))
                img3 = Image.fromarray(gt3[k])
                img3.save('%s/%s' % (im_save_dir, self.n3[k].replace('.', 'gt.') ))

                img4 = Image.fromarray(data1[k])
                img4.save('%s/%s' % (im_save_dir, self.n1[k].replace('.', 'out.') ))
                img5 = Image.fromarray(data2[k])
                img5.save('%s/%s' % (im_save_dir, self.n2[k].replace('.', 'out.') ))
                img6 = Image.fromarray(data3[k])
                img6.save('%s/%s' % (im_save_dir, self.n3[k].replace('.', 'out.') ))
        else:
            out = utils.transform_to_numpy(self.out)
            for k in range(out.shape[0]):
                img=Image.fromarray(out[k])
                img.save('%s/%s'%(im_save_dir,self.n[k]))
