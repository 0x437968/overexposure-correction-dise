import os
import torch


class BaseModel():
    def __init__(self,opt):
        self.opt=opt
        self.gpu_idx=opt.gpu_idx
        self.isTrain=opt.isTrain
        self.net_save_dir=os.path.join(opt.checkpoints_dir,opt.model+'_'+opt.name)
        self.device = self.set_device()
        self.net_names=[]

    def name(self):
        return 'BaseModel'

    @staticmethod
    def modify_options(parser):
        """Add new options and rewrite default values for existing options"""
        return parser

    def save_results(self):
        pass


    def print_loss(self,name,epoch,idx):
        pass

    def set_device(self):
        if torch.cuda.is_available() and self.gpu_idx>=0:
            return torch.device('cuda:%d' % self.gpu_idx)
        else:
            print('use cpu')
            return torch.device('cpu')

    def clear_sumloss(self):
        pass

    # save models
    def save_networks(self):
        """Save all the networks to the disk"""
        for name in self.net_names:
            if isinstance(name, str):
                print('save %s '%name)
                save_filename = 'net_%s.path' % name
                save_path = os.path.join(self.net_save_dir, save_filename)
                net = getattr(self, 'net_' + name)
                torch.save(net.state_dict(), save_path)


    def print_networks(self):
        for name in self.net_names:
            if isinstance(name, str):
                print(getattr(self,'net_'+name))

    # load models
    def load_networks(self):
        """Load all the networks from the disk"""
        for name in self.net_names:
            if isinstance(name, str):
                filename = 'net_%s.path' % name
                path = os.path.join(self.net_save_dir, filename)
                net = getattr(self, 'net_' + name)
                try:
                    net.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
                    print('load %s from %s' % (name, path))
                except FileNotFoundError:
                    print('no checkpoint in %s'%path)
                net.to(self.device)


