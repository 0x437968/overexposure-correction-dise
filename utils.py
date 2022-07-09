import importlib
import torch
import os
import os.path
import glob
from PIL import Image
import numpy as np

def load_image(path,resize=False,size=256):
    im=Image.open(path)
    if resize :
        im=im.resize((size,size),Image.BICUBIC)
    im = np.array(im)
    im = im.astype(np.float32)
    im = im / 255
    if len(im.shape) != 3:
        im = np.expand_dims(im, 0)
    else:
        im = np.transpose(im, (2, 0, 1))
    return im

def init_folder(*folders):
    for f in folders:
        if not os.path.exists(f):
            os.makedirs(f)


def _freeze(*args):
    """freeze the network for forward process"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def _unfreeze(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True


def transform_to_numpy(tensor_):
    im = tensor_.detach_().cpu().numpy()
    im = im / 2 + 0.5
    if im.shape[1] == 3:
        im = np.transpose(im, (0, 2, 3, 1))
    else:
        im=np.squeeze(im)
    im=(im*255).astype(np.uint8)
    return im



def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w * h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (
        module, target_cls_name))
        exit(0)

    return cls

