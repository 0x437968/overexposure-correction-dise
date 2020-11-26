import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import torchvision.models as models
from model.network.base_function import SPADEResnetBlock,ADAIN
from torch.nn import init
import utils
import random

##################################################################

def get_norm_layer(type):
    if type=='LN':
        return functools.partial(LayerNorm,affine=True)
    if type=='IN':
        return functools.partial(nn.InstanceNorm2d,affine=True)
    if type=='BN':
        return functools.partial(nn.BatchNorm2d,momentum=0.1,affine=True)
    return None

def get_non_linearity(layer_type='relu'):
  if layer_type == 'relu':
    nl_layer = functools.partial(nn.ReLU)
  elif layer_type == 'lrelu':
    nl_layer = functools.partial(nn.LeakyReLU,0.1)
  elif layer_type == 'selu':
    nl_layer = functools.partial(nn.SELU)
  else:
    raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
  return nl_layer

def block_nl_nll_conv(inc,outc,kernel_size=3,stride=1,padding=1,norm_layer='LN',non_linerity_layer='relu'):
    norm=get_norm_layer(norm_layer)
    nll=get_non_linearity(layer_type=non_linerity_layer)
    model=[]
    if norm is not None:
        model.append(norm(inc))
    model+=[nll(),nn.ReflectionPad2d(padding),nn.Conv2d(inc,outc,kernel_size=kernel_size,stride=stride,padding=0)]
    return nn.Sequential(*model)

def padding_conv(inc,outc,kernel_size=3,stride=1,padding=1):
    model=[]
    model+=[nn.ReflectionPad2d(padding),nn.Conv2d(inc,outc,kernel_size=kernel_size,stride=stride,padding=0)]
    return nn.Sequential(*model)

def block_conv_nl_nll(inc,outc,kernel_size=3,stride=1,padding=1,norm_layer='LN',non_linerity_layer='relu'):
    norm=get_norm_layer(norm_layer)
    nll=get_non_linearity(layer_type=non_linerity_layer)
    model=[]
    model.append(nn.ReflectionPad2d(padding))
    model+=[nn.Conv2d(inc,outc,kernel_size=kernel_size,stride=stride,padding=0)]
    if norm is not None:
        model.append(norm(outc))
    model+=[nll()]
    return nn.Sequential(*model)

class resblock_conv(nn.Module):
    def __init__(self,inc,outc=None,hiddenc=None,norm_layer='LN',non_linerity_layer='relu'):
        super(resblock_conv,self).__init__()
        hiddenc=inc if hiddenc is None else hiddenc
        outc=inc if outc is None else outc
        norm=get_norm_layer(norm_layer)
        nll=get_non_linearity(layer_type=non_linerity_layer)
        model=[]
        if norm is not None:
            model.append(norm(inc))
        model+=[nll(),nn.ReflectionPad2d(1),nn.Conv2d(inc,hiddenc,3,1,padding=0)]
        if norm is not None:
            model.append(norm(hiddenc))
        model+=[nll(),nn.ReflectionPad2d(1),nn.Conv2d(hiddenc,outc,3,1,padding=0)]
        self.bp=False
        if outc!=inc:
            self.bp=True
            self.bypass=nn.Conv2d(inc,outc,1,1,0)
        self.model=nn.Sequential(*model)

    def forward(self,x):
        residual=x
        if self.bp:
            out=self.model(x)+self.bypass(residual)
        else:
            out=self.model(x)+residual
        return out

class resblock_transconv(nn.Module):
    def __init__(self,inc,outc,hiddenc=None,norm_layer='LN',non_linerity_layer='relu'):
        super(resblock_transconv,self).__init__()
        norm=get_norm_layer(norm_layer)
        nll=get_non_linearity(layer_type=non_linerity_layer)
        hiddenc=inc if hiddenc is None else hiddenc
        self.model=[]
        if norm is not None:
            self.model.append(norm(inc))
        self.model+=[nll(),
            nn.Conv2d(inc,hiddenc,3,1,1)]
        if norm is not None:
            self.model.append(norm(hiddenc))
        self.model+=[nll(),
            nn.ConvTranspose2d(hiddenc,outc,3,2,1,1)]
        self.model=nn.Sequential(*self.model)
        self.bypass=nn.Sequential(nn.ConvTranspose2d(inc,outc,3,2,1,1))

    def forward(self,x):
        residual=x
        out=self.model(x)+self.bypass(residual)
        return out

class resblock_upbilin(nn.Module):
    def __init__(self,inc,outc=None,norm_layer='LN',non_linerity_layer='relu'):
        super(resblock_upbilin,self).__init__()
        norm=get_norm_layer(norm_layer)
        nll=get_non_linearity(layer_type=non_linerity_layer)
        outc=inc if outc is None else outc
        self.model=[]
        if norm is not None:
            self.model.append(norm(inc))
        self.model+=[nll(),
            nn.Conv2d(inc,outc,3,1,1)]
        if norm is not None:
            self.model.append(norm(outc))
        self.model+=[nll(),nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
        self.model=nn.Sequential(*self.model)
        self.bypass=nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

    def forward(self,x):
        residual=x
        out=self.model(x)+self.bypass(residual)
        return out

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def initialize_weights(net,init_type='xavier', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    for m in net.modules():
        init_func(m)
        
###########################################################################
class Layer_down(nn.Module):
    def __init__(self, inchannel, outchannel, blocks=1,norm_layer='LN'):
        super(Layer_down, self).__init__()
        self.flatten = nn.Sequential(
            get_norm_layer(norm_layer)(inchannel),
            get_non_linearity()(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            padding_conv(inchannel,outchannel,3,1,1)
        )
        self.layer_blocks = []
        for _ in range(blocks):
            self.layer_blocks.append(resblock_conv(outchannel,norm_layer=norm_layer))
        self.layer_blocks = nn.Sequential(*self.layer_blocks)

    def forward(self, x):
        out = self.flatten(x)
        out = self.layer_blocks(out)
        return out

class Layer_up(nn.Module):
    def __init__(self, inchannel,outchannel, blocks=1,bilinear=False,norm_layer='LN', enc_inc=None):
        super(Layer_up, self).__init__()
        enc_inc=inchannel if enc_inc is None else enc_inc
        if bilinear:
            self.heighten=nn.Sequential(get_norm_layer(norm_layer)(inchannel),
                                            get_non_linearity()(),
                                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                padding_conv(inchannel,outchannel,3,1,1),
                                    ) 
        else:
            self.heighten = nn.Sequential(resblock_transconv(inchannel,inchannel,norm_layer=norm_layer),
                                    block_nl_nll_conv(inchannel,outchannel,norm_layer=norm_layer))
        self.bypass_sc = block_nl_nll_conv(enc_inc,outchannel,3,1,1,norm_layer=norm_layer)
        self.layer_blocks = []
        for _ in range(blocks):
            self.layer_blocks.append(resblock_conv(outchannel,norm_layer=norm_layer))
        self.layer_blocks = nn.Sequential(*self.layer_blocks)

    def forward(self, x, cat):
        y = self.heighten(x)
        y=y+self.bypass_sc(cat)
        y = self.layer_blocks(y)
        return y

#####################################################################################


class Unet_encoder(nn.Module):
    def __init__(self, inchannel, ndf=32,blocks=1, depth=5,norm_layer='LN'):
        super(Unet_encoder, self).__init__()
        self.Inconv=nn.Sequential(padding_conv(inchannel,ndf,7,1,3))
        self.layer_blocks = []
        for _ in range(blocks):
            self.layer_blocks.append(resblock_conv(ndf,norm_layer=norm_layer))
        self.layer_blocks = nn.Sequential(*self.layer_blocks)
        self.down_layers = []
        channel_in=ndf
        for _ in range(depth - 1):
            self.down_layers.append(Layer_down(channel_in, channel_in*2, blocks,norm_layer=norm_layer))
            channel_in=channel_in*2
        self.down_layers = nn.ModuleList(self.down_layers)
        initialize_weights(self)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        tmp = []
        out = self.Inconv(x)
        out=self.layer_blocks(out)
        tmp.append(out)
        for layer in self.down_layers[:-1]:
            out = layer(out)
            tmp.append(out)
        out = self.down_layers[-1](out)
        return out, tmp

class Unet_decoder(nn.Module):
    def __init__(self, outchannel,ndf=32, blocks=1, depth=5,bilinear=False,norm_layer='LN'):
        super(Unet_decoder, self).__init__()
        self.up_layers = []
        channel_in=ndf*(2**(depth-1))
        for _ in range(depth - 1):
            self.up_layers.append(Layer_up(channel_in, channel_in//2, blocks=blocks,
                                    bilinear=bilinear,norm_layer=norm_layer,enc_inc=channel_in//2))
            channel_in=channel_in//2
        self.up_layers = nn.ModuleList(self.up_layers)
        self.outconv = nn.Sequential(block_nl_nll_conv(channel_in,outchannel,3,1,1,norm_layer=norm_layer),
                                           block_nl_nll_conv(outchannel,outchannel,1,1,0,norm_layer=norm_layer),
                                                )
        initialize_weights(self)

    def forward(self, x, tmp):
        out=x
        for idx, layer in enumerate(self.up_layers):
            out = layer(out, tmp[-(idx + 1)])
        out = self.outconv(out)
        return out

class Unet(nn.Module):
    def __init__(self, inchannel=3, outchannel=3, ndf=64,enc_blocks=1, dec_blocks=3, depth=5,bilinear=False,norm_layer='LN'):
        super(Unet, self).__init__()
        self.e = Unet_encoder(inchannel=inchannel,ndf=ndf,blocks= enc_blocks,depth= depth,norm_layer=norm_layer)
        self.d = Unet_decoder(outchannel=outchannel,ndf=ndf,blocks= dec_blocks,depth= depth,bilinear=bilinear,norm_layer=norm_layer)

    def forward(self, x):
        out, temp = self.e(x)
        out = self.d(out, temp)
        out=torch.tanh(out)
        return out


######################################################################################
class Encoder_S(nn.Module):
    def __init__(self,inc=3,n_downsample=2,ndf=32,norm_layer='LN'):
        super().__init__()
        self.Inconv=padding_conv(inc,ndf,7,1,3)
        channel_in=ndf
        self.model=list()
        for _ in range(n_downsample):
            self.model+=[block_nl_nll_conv(channel_in,channel_in*2,4,2,1,norm_layer=norm_layer)]
            channel_in=channel_in*2
        for _ in range(2):
            self.model+=[resblock_conv(channel_in,norm_layer=norm_layer)]
        self.model=nn.Sequential(*self.model)
        self.Outconv=nn.Sequential(block_nl_nll_conv(channel_in,channel_in,1,1,0,norm_layer=norm_layer))
        self.outc=channel_in
        self.n_downsample=n_downsample
        initialize_weights(self)

    def forward(self, x):
        y=self.Inconv(x)
        # ho=[]
        # for layer in self.model:
        #     y=layer(y)
        #     ho.append(y)
        y=self.model(y)
        y=self.Outconv(y)
        return y

class Encoder_E(nn.Module):
    def __init__(self,inc=3,n_downsample=4,outc=8,ndf=64,usekl=True):
        super().__init__()
        self.usekl=usekl
        self.conv1=block_conv_nl_nll(inc,ndf,7,1,3,norm_layer=None)
        self.downconv=[]
        channel_in=ndf
        for _ in range(2):
            self.downconv.append(block_conv_nl_nll(channel_in,channel_in*2,4,2,1,norm_layer=None))
            channel_in*=2
        for _ in range(n_downsample-2):
            self.downconv+=[block_conv_nl_nll(channel_in,channel_in,4,2,1,norm_layer=None)]
        self.downconv.append(nn.AdaptiveAvgPool2d(1))
        self.downconv=nn.Sequential(*self.downconv)
        if usekl:
            self.mean_fc =nn.Linear(channel_in, outc)
            self.var_fc= nn.Linear(channel_in, outc)
        else:
            self.fc= nn.Linear(channel_in, outc)
        initialize_weights(self)

    def forward(self, x):
        y = self.conv1(x)
        y=self.downconv(y)
        y = y.view(x.size(0), -1)
        if self.usekl:
            mean=self.mean_fc(y)
            var=self.var_fc(y)
            return mean,var
        else:
            y=self.fc(y)
            return y

class Decoder_SE(nn.Module):
    def __init__(self,s_inc=128,e_inc=8,outc=3,n_upsample=2,norm_layer='LN'):
        super().__init__()
        inc=s_inc
        self.adin=ADAIN(inc,e_inc)
        self.model=[]
        self.model+=[padding_conv(inc,inc,1,1,0),resblock_conv(inc,norm_layer=norm_layer)]
        channel_in=inc
        for _ in range(n_upsample):
            self.model+=[resblock_transconv(channel_in,channel_in//2,norm_layer=norm_layer)]
            channel_in=channel_in//2
        self.model=nn.Sequential(*self.model)
        self.outconv = nn.Sequential(
            block_nl_nll_conv(channel_in,outc,3,1,1,norm_layer=norm_layer),
        )
        initialize_weights(self)

    def forward(self, x1,x2):
        out = self.adin(x1, x2)
        out=self.model(out)
        out=self.outconv(out)
        out=torch.tanh(out)
        return out

#######################################################################################

#VGG 19
class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        vgg=models.vgg19()
        vgg.load_state_dict(torch.load('./checkpoints/vgg19/vgg19-dcbb9e9d.pth',map_location=torch.device('cpu')))
        features = vgg.features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out
