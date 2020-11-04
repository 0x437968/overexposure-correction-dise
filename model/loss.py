import torch.nn as nn
import torch
import torch.nn.functional as F
from model.network.components import VGG19,encoder_s
import utils



class grad_loss(nn.Module):
    def __init__(self,device):
        super(grad_loss,self).__init__()
        self.device=device
        self.criterion=nn.L1Loss()

    def __call__(self,x,gt):
        gt_gx = self.calculate_grad(gt, 'x',self.device)
        gt_gy = self.calculate_grad(gt, 'y',self.device)
        re_gx = self.calculate_grad(x, 'x',self.device)
        re_gy = self.calculate_grad(x, 'y',self.device)
        return self.criterion(re_gx+re_gy,gt_gx+gt_gy)

    def calculate_grad(self,input_tensor, direction,device):
        #h, w = input_tensor.size()[2], input_tensor.size()[3]
        input_tensor=input_tensor/2+0.5
        kernel_x = torch.reshape(torch.Tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]), (1, 1, 3, 3)).to(device)
        kernel_y = torch.reshape(torch.Tensor([[-1.0, -2, -1], [0, 0, 0], [1, 2, 1]]), (1, 1, 3, 3)).to(device)
        kernel_x=kernel_x.repeat(1,input_tensor.size()[1],1,1)
        kernel_y=kernel_y.repeat(1,input_tensor.size()[1],1,1)
        assert direction in ['x', 'y']
        if direction == "x":
            kernel = kernel_x
        else:
            kernel = kernel_y

        out = F.conv2d(input_tensor, kernel, padding=(1, 1))
        out=torch.sum(torch.abs(out),dim=1)#.view(input_tensor.size()[0],1,input_tensor.size()[2],input_tensor.size()[3])
        return out
class vgg_style_loss(nn.Module):
    def __init__(self):
        super(vgg_style_loss,self).__init__()
        self.vgg=VGG19()
        self.criterion=nn.L1Loss()
    def __call__(self,x,y):
        x_vgg,y_vgg=self.vgg(x),self.vgg(y)
        style_loss = 0.0
        style_loss += self.criterion(utils.gram(x_vgg['relu2_2']), utils.gram(y_vgg['relu2_2']))
        style_loss += self.criterion(utils.gram(x_vgg['relu3_4']), utils.gram(y_vgg['relu3_4']))
        style_loss += self.criterion(utils.gram(x_vgg['relu4_4']), utils.gram(y_vgg['relu4_4']))
        style_loss += self.criterion(utils.gram(x_vgg['relu5_2']), utils.gram(y_vgg['relu5_2']))
        return style_loss

class scene_loss(nn.Module):
    def __init__(self,path):
        super(scene_loss,self).__init__()
        self.E_s=encoder_s(norm_layer='LN')
        try:
            self.E_s.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
            print('load %s' %path)
        except FileNotFoundError:
            raise ValueError('no E_s in %s'%path)
        utils._freeze(self.E_s)
        self.criterion=nn.L1Loss()

    def __call__(self,x,y):
        s1=self.E_s(x)
        s2=self.E_s(y)
        return self.criterion(s1,s2)

class AdversarialLoss(nn.Module):
    """
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='lsgan', target_real_label=1.0, target_fake_label=0.0):
        """
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, for_dis=None):
        if self.type == 'hinge':
            if for_dis:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss