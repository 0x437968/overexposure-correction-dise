import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import torchvision.models as models
from model.network.base_function import SPADEResnetBlock,ADAIN
from torch.nn import init
#
def Get_Norml(type):
    if type=='LN':
        return functools.partial(LayerNorm,affine=True)
    if type=='IN':
        return functools.partial(nn.InstanceNorm2d,affine=True)
    if type=='BN':
        return functools.partial(nn.BatchNorm2d,momentum=0.1,affine=True)
    return None

def Get_Actv(type='rl'):
    if type=='rl':
        return functools.partial(nn.ReLU)
    if type=='lrl':
        return functools.partial(nn.LeakyReLU)
    raise NotImplementedError('undefined activation layer %s ' %type)

def nolinear_conv(inc,outc,kernel_size=3,stride=1,padding=1,norm_layer='BN'):
    norm=Get_Norml(norm_layer)
    seq=[]
    if norm is not None:
        seq.append(norm(inc))
    seq+=[nn.ReLU(),nn.ReflectionPad2d(padding),nn.Conv2d(inc,outc,kernel_size=kernel_size,stride=stride,padding=0)]
    return nn.Sequential(*seq)

def conv_nolinear(inc,outc,kernel_size=3,stride=1,padding=1,norm_layer='BN'):
    norm=Get_Norml(norm_layer)
    seq=[]
    seq.append(nn.ReflectionPad2d(padding))
    seq+=[nn.Conv2d(inc,outc,kernel_size=kernel_size,stride=stride,padding=0)]
    if norm is not None:
        seq.append(norm(outc))
    seq+=[nn.ReLU()]
    return nn.Sequential(*seq)

class resblock(nn.Module):
    def __init__(self,dim,norm_layer='BN'):
        super(resblock,self).__init__()
        norm=Get_Norml(norm_layer)
        seq=[]
        if norm is not None:
            seq.append(norm(dim))
        seq+=[nn.ReLU(),nn.ReflectionPad2d(1),nn.Conv2d(dim,dim,3,1,padding=0)]
        if norm is not None:
            seq.append(norm(dim))
        seq+=[nn.ReLU(),nn.ReflectionPad2d(1),nn.Conv2d(dim,dim,3,1,padding=0)]
        self.seq=nn.Sequential(*seq)

    def forward(self,x):
        res=x
        out=self.seq(x)+res
        return out

class res_conv_block(nn.Module):
    def __init__(self,inc,outc,hiddenc=None,norm_layer='BN'):
        super(res_conv_block,self).__init__()
        hiddenc=inc if hiddenc is None else hiddenc
        norm=Get_Norml(norm_layer)
        seq=[]
        if norm is not None:
            seq.append(norm(inc))
        seq+=[nn.ReLU(),nn.ReflectionPad2d(1),nn.Conv2d(inc,hiddenc,3,1,padding=0)]
        if norm is not None:
            seq.append(norm(hiddenc))
        seq+=[nn.ReLU(),nn.ReflectionPad2d(1),nn.Conv2d(hiddenc,outc,3,1,padding=0)]
        self.bypass=nn.Conv2d(inc,outc,1,1,0)
        self.seq=nn.Sequential(*seq)

    def forward(self,x):
        out=self.seq(x)+self.bypass(x)
        return out

class res_transconv(nn.Module):
    def __init__(self,inc,outc,hiddenc=None,norm_layer='BN'):
        super(res_transconv,self).__init__()
        norm=Get_Norml(norm_layer)
        hiddenc=inc if hiddenc is None else hiddenc
        self.model=[]
        if norm is not None:
            self.model.append(norm(inc))
        self.model+=[nn.ReLU(),
            nn.Conv2d(inc,hiddenc,3,1,1)]
        if norm is not None:
            self.model.append(norm(hiddenc))
        self.model+=[nn.ReLU(),
            nn.ConvTranspose2d(hiddenc,outc,3,2,1,1)]
        self.model=nn.Sequential(*self.model)
        self.bypass=nn.Sequential(nn.ConvTranspose2d(inc,outc,3,2,1,1))

    def forward(self,x):
        out=self.model(x)+self.bypass(x)
        return out

class encoder_s(nn.Module):
    def __init__(self,inc=3,depth=3,ndf=64,blocks=3,norm_layer='BN'):
        super(encoder_s, self).__init__()
        self.seq=[]
        self.seq+=[conv_nolinear(inc,ndf,kernel_size=3,stride=1,padding=1,norm_layer=norm_layer)]
        for _ in range(depth-1):
            self.seq.append(conv_nolinear(ndf,ndf*2,kernel_size=4,stride=2,padding=1,norm_layer=norm_layer))
            ndf *= 2
        self.seq+=[nn.Conv2d(ndf,ndf*2,3,1,1)]
        for _ in range(blocks):
            self.seq.append(resblock(ndf*2,norm_layer=norm_layer))
        self.seq+=[nolinear_conv(ndf*2,ndf*2,3,1,1,norm_layer=norm_layer)]
        self.seq=nn.Sequential(*self.seq)
        initialize_weights(self)

    def forward(self, x):
        y=self.seq(x)
        return y

class encoder_e(nn.Module):
    def __init__(self,inc=3,depth=4,outc=128,ndf=64,usekl=True,norm_layer='none'):
        super(encoder_e, self).__init__()
        self.usekl=usekl
        self.conv1=conv_nolinear(inc,ndf,4,2,1,norm_layer=norm_layer)
        self.downconv=[]
        for i in range(depth-1):
            if outc>=64:
                x=ndf*(2**i)
                y=ndf*(2**(i+1))
            else:
                x=ndf*(i+1)
                y=ndf*(i+2)
            self.downconv.append(conv_nolinear(x,y,4,2,1,norm_layer=norm_layer))
        self.downconv.append(nn.AdaptiveAvgPool2d(1))
        self.downconv=nn.Sequential(*self.downconv)
        if usekl:
            self.mean_fc = nn.Sequential(
                nn.Linear(y, outc),
            )
            self.var_fc=nn.Sequential(
                nn.Linear(y, outc),
            )
        else:
            self.fc=nn.Sequential(
                nn.Linear(y, outc),
            )
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

class decoder_se(nn.Module):
    def __init__(self,inc=512,e_inc=16,outc=3,depth=3,norm_layer='BN'):
        super(decoder_se, self).__init__()
        self.adin=ADAIN(inc,e_inc)
        self.seq=[]
        self.seq+=[nn.Conv2d(inc,inc,3,1,1)]
        for _ in range(3):
            self.seq.append(resblock(inc,norm_layer=norm_layer))
        self.seq+=[nolinear_conv(inc,inc//2,3,1,1,norm_layer=norm_layer)]
        inc=inc//2
        for _ in range(depth-1):
            self.seq+=[res_transconv(inc,inc//2,norm_layer=norm_layer)]
            inc=inc//2
        self.seq=nn.Sequential(*self.seq)
        self.outconv = nn.Sequential(
            nolinear_conv(inc,outc,3,1,1,norm_layer=norm_layer),
            nn.Tanh(),
        )
        initialize_weights(self)

    def forward(self, x1,x2):
        out = self.adin(x1, x2)
        out=F.leaky_relu(out)
        out=self.seq(out)
        out=self.outconv(out)
        return out

#
class down(nn.Module):
    def __init__(self, inchannel, outchannel, blocks=1,norm_layer='BN'):
        super(down, self).__init__()
        self.seq1 = nn.Sequential(
            Get_Norml(norm_layer)(inchannel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(inchannel,outchannel,3,1,1),
        )
        if blocks >= 0:
            self.seq2 = []
            for _ in range(blocks):
                self.seq2.append(resblock(outchannel,norm_layer=norm_layer))
            self.seq2 = nn.Sequential(*self.seq2)
        else:
            self.seq2=nolinear_conv(outchannel,outchannel,kernel_size=3,stride=1,padding=1,norm_layer=norm_layer)

    def forward(self, x):

        out = self.seq1(x)
        out = self.seq2(out)
        return out


class up(nn.Module):
    def __init__(self, inchannel, outchannel, blocks=1, concat=False,bilinear=False,norm_layer='BN'):
        super(up, self).__init__()
        self.concat=concat
        if self.concat:
            if bilinear:
                self.upconv=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.upconv = res_transconv(inchannel//2,inchannel//2,norm_layer=norm_layer)
        else:
            if bilinear:
                self.upconv=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.upconv = res_transconv(inchannel,inchannel,norm_layer=norm_layer)
            self.enc_conv = nolinear_conv(inchannel,inchannel,3,1,1,norm_layer=norm_layer)
        self.seq1=nolinear_conv(inchannel, outchannel, kernel_size=3, stride=1, padding=1,norm_layer=norm_layer)
        if blocks >= 0:
            self.seq2 = []
            for _ in range(blocks):
                self.seq2.append(resblock(outchannel,norm_layer=norm_layer))
            self.seq2 = nn.Sequential(*self.seq2)
        else:
            self.seq2=nolinear_conv(outchannel,outchannel,kernel_size=3,stride=1,padding=1,norm_layer=norm_layer)

    def forward(self, x, cat):
        out = self.upconv(x)
        if self.concat:
            out = torch.cat([out, cat], dim=1)
        else:
            out=out+self.enc_conv(cat)
        out = self.seq1(out)
        out = self.seq2(out)
        return out

class spade_up(nn.Module):
    def __init__(self, inchannel, outchannel, blocks=1, concat=False,bilinear=False,norm_layer='BN'):
        super().__init__()
        self.concat=concat
        if self.concat:
            if bilinear:
                self.upconv=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.upconv = res_transconv(inchannel//2,inchannel//2,norm_layer=norm_layer)
        else:
            if bilinear:
                self.upconv=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.upconv = res_transconv(inchannel,inchannel,norm_layer=norm_layer)
            self.enc_conv = nolinear_conv(inchannel,inchannel,3,1,1,norm_layer=norm_layer)
        self.seq1=nolinear_conv(inchannel, outchannel, kernel_size=3, stride=1, padding=1,norm_layer=norm_layer)
        if blocks >= 0:
            self.seq2 = []
            for _ in range(blocks):
                self.seq2.append(resblock(outchannel,norm_layer=norm_layer))
            self.seq2 = nn.Sequential(*self.seq2)
        else:
            self.seq2=nolinear_conv(outchannel,outchannel,kernel_size=3,stride=1,padding=1,norm_layer=norm_layer)
        self.spa_res=SPADEResnetBlock(outchannel,outchannel,None,label_nc=1)


    def forward(self, x, cat,seg):
        out = self.upconv(x)
        if self.concat:
            out = torch.cat([out, cat], dim=1)
        else:
            out=out+self.enc_conv(cat)
        out = self.seq1(out)
        out = self.seq2(out)
        out=self.spa_res(out,seg)
        return out

#
class encod(nn.Module):
    def __init__(self, inchannel, ndf=32,blocks=1, depth=5,norm_layer='BN'):
        super(encod, self).__init__()
        self.conv1=nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(inchannel,ndf,7,1,0))
        if blocks >= 0:
            self.conv2 = []
            for _ in range(blocks):
                self.conv2.append(resblock(ndf,norm_layer=norm_layer))
            self.conv2 = nn.Sequential(*self.conv2)
        else:
            self.conv2 = nolinear_conv(ndf,ndf,3,1,1,norm_layer=norm_layer)
        self.downconv = []
        for i in range(depth - 2):
            self.downconv.append(down(ndf * 2 ** (i), ndf * 2 ** (i + 1), blocks,norm_layer=norm_layer))
        self.downconv = nn.ModuleList(self.downconv)
        self.finaldown = down(ndf * 2 ** (depth - 2), ndf * 2 ** (depth - 2), blocks,norm_layer=norm_layer)
        initialize_weights(self)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        temp = []
        out = self.conv1(x)
        out=self.conv2(out)
        temp.append(out)
        for idx, layer in enumerate(self.downconv):
            out = layer(out)
            temp.append(out)
        out = self.finaldown(out)
        return out, temp

class decod(nn.Module):
    def __init__(self, outchannel,ndf=32, blocks=1, depth=5,concat=False,bilinear=False,norm_layer='BN'):
        super(decod, self).__init__()
        self.upconv = []
        self.concat=concat
        if concat:
            for i in range(depth - 2):
                self.upconv.append(up(ndf * 2 ** (depth - i - 1), ndf * 2 ** (depth - i - 3), blocks=blocks,concat=concat,bilinear=bilinear,norm_layer=norm_layer))
            self.upconv2 = up(ndf * 2, ndf, blocks=blocks, concat=concat, bilinear=bilinear,norm_layer=norm_layer)
            self.conv=nolinear_conv(ndf,outchannel,kernel_size=3,stride=1,padding=1,norm_layer=norm_layer)
            self.resblocks = []
            for _ in range(blocks):
                self.resblocks.append(resblock(outchannel,norm_layer=norm_layer))
            self.resblocks = nn.Sequential(*self.resblocks)
        else:
            for i in range(depth - 2):
                self.upconv.append(up(ndf * 2 ** (depth - i - 2), ndf * 2 ** (depth - i - 3), blocks=blocks,concat=concat,bilinear=bilinear,norm_layer=norm_layer))
            self.upconv2 = up(ndf, ndf, blocks=blocks, concat=concat, bilinear=bilinear,norm_layer=norm_layer)
        self.upconv = nn.ModuleList(self.upconv)

        self.final = nn.Sequential(nolinear_conv(ndf,outchannel,3,1,1,norm_layer=norm_layer),nn.Tanh())
        initialize_weights(self)

    def forward(self, x, temp):
        out=x
        for idx, layer in enumerate(self.upconv):
            out = layer(out, temp[-(idx + 1)])
        out = self.upconv2(out, temp[0])
        if self.concat:
            out=self.conv(out)
            out=self.resblocks(out)
        out = self.final(out)
        return out

class Unet(nn.Module):
    def __init__(self, inchannel=3, outchannel=3, ndf=64,enc_blocks=1, dec_blocks=3, depth=5,concat=False,bilinear=False,norm_layer='BN'):
        super(Unet, self).__init__()
        self.enc = encod(inchannel=inchannel,ndf=ndf,blocks= enc_blocks,depth= depth,norm_layer=norm_layer)
        self.dec = decod(outchannel=outchannel,ndf=ndf,blocks= dec_blocks,depth= depth,concat=concat,bilinear=bilinear,norm_layer=norm_layer)

    def forward(self, input):
        out, temp = self.enc(input)
        out = self.dec(out, temp)
        return out



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

#
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

#
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