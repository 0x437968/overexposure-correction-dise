import torch
import torch.nn as nn
from model.basemodel import BaseModel
from model.network.components import S_Encoder,SE_Decoder,E_Encoder
from model.loss import vgg_style_loss
import utils
import numpy as np
from PIL import Image

class DISE(BaseModel):
    def __init__(self,opt):
        BaseModel.__init__(self,opt)
        self.net_names=['E_s','E_e','D_se']
        self.net_E_s=S_Encoder()
        self.net_E_e=E_Encoder(outc=self.opt.dimen_e,ndf=64,usekl=True)
        self.net_D_se=SE_Decoder(e_inc=self.opt.dimen_e)

        if self.isTrain:
            self.l1loss=nn.L1Loss()
            self.Styleloss=vgg_style_loss().to(self.device)
            self.loss_names=['kl','style','recon','ident','total']

            self.optim_E_s=torch.optim.Adam(self.net_E_s.parameters(),lr=opt.lr,betas=(0.0,0.999))
            self.optim_E_e=torch.optim.Adam(self.net_E_e.parameters(),lr=opt.lr,betas=(0.0,0.999))
            self.optim_D_se=torch.optim.Adam(self.net_D_se.parameters(),lr=opt.lr,betas=(0.0,0.999))

    @staticmethod
    def modify_options(parser):
        opt, _ =parser.parse_known_args()
        parser.add_argument('--dimen_e', type=int,default=8, help='dimension for exposure vector')
        if opt.phase=='train':
            parser.add_argument('--lr', type=float, default=0.0001, help='learning rate ')
            parser.add_argument('--lambda_recon', type=float, default=50.0, help='weight for recon loss')
            parser.add_argument('--lambda_kl', type=float, default=0.01, help='weight for kl loss')
            parser.add_argument('--lambda_style', type=float, default=1500.0, help='weight for style loss')
            parser.add_argument('--lambda_ident', type=float, default=1000.0, help='weight for scene loss')
            parser.add_argument('--dir_in',  default='', help='sequence images diretory')
        else:
            parser.add_argument('--dir_in',  default='', help='scene images diretory')
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
            self.s3 = self.net_E_s(self.im3)
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

            self.out11=self.net_D_se(self.s2,e1)
            self.out22=self.net_D_se(self.s2,e2)
            self.out33=self.net_D_se(self.s2,e3)

        else:
            self.s=self.net_E_s(self.test)
            sample_z = torch.randn(self.s.size(0), self.opt.dimen_e).to(self.device)
            mean,var=self.net_E_e(self.e)
            std=torch.exp(var*0.5)
            e = sample_z* std + mean
            self.out=self.net_D_se(self.s,e)

    def backward_(self):
        #   scene constraint
        self.loss_ident=(self.l1loss(self.s1,self.s3)+self.l1loss(self.s1,self.s2)+self.l1loss(self.s2,self.s3))*self.opt.lambda_ident

        #   style loss
        self.loss_style=(self.Styleloss(self.out11,self.im1)+self.Styleloss(self.out33,self.im3)+self.Styleloss(self.out22,self.im2))*self.opt.lambda_style

        #   reconstruction loss
        self.loss_recon=(self.l1loss(self.out11,self.im1)+self.l1loss(self.out33,self.im3)+self.l1loss(self.out22,self.im2))*self.opt.lambda_recon

        #   kl loss
        loss_kl1 = torch.sum(self.mean1**2+torch.exp(self.var1)-self.var1-1.0)*0.5
        loss_kl2 = torch.sum(self.mean2**2+torch.exp(self.var2)-self.var2-1.0)*0.5
        loss_kl3 = torch.sum(self.mean3**2+torch.exp(self.var3)-self.var3-1.0)*0.5
        self.loss_kl = (loss_kl1+loss_kl3+loss_kl2) * self.opt.lambda_kl

        #   total loss
        self.loss_total=self.loss_ident+self.loss_style+self.loss_recon+self.loss_kl

        self.loss_total.backward()
        self.sumloss.append(self.loss_total.item())

    def train(self):
        self.forward()

        self.optim_E_s.zero_grad()
        self.optim_E_e.zero_grad()
        self.optim_D_se.zero_grad()
        self.backward_()
        self.optim_D_se.step()
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
            out1 = utils.transform_to_numpy(self.out11)
            out2 = utils.transform_to_numpy(self.out22)
            out3 = utils.transform_to_numpy(self.out33)
            for k in range(min(gt1.shape[0],2)):
                img1 = Image.fromarray(gt1[k])
                img1.save('%s/%s' % (im_save_dir, self.n1[k].replace('.', 'in.') ))
                img2 = Image.fromarray(gt2[k])
                img2.save('%s/%s' % (im_save_dir, self.n2[k].replace('.', 'in.') ))
                img3 = Image.fromarray(gt3[k])
                img3.save('%s/%s' % (im_save_dir, self.n3[k].replace('.', 'in.') ))

                img4 = Image.fromarray(out1[k])
                img4.save('%s/%s' % (im_save_dir, self.n1[k].replace('.', 'out.') ))
                img5 = Image.fromarray(out2[k])
                img5.save('%s/%s' % (im_save_dir, self.n2[k].replace('.', 'out.') ))
                img6 = Image.fromarray(out3[k])
                img6.save('%s/%s' % (im_save_dir, self.n3[k].replace('.', 'out.') ))
        else:
            out = utils.transform_to_numpy(self.out)
            for k in range(out.shape[0]):
                img=Image.fromarray(out[k])
                img.save('%s/%s'%(im_save_dir,self.n[k]))
