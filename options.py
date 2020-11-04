import argparse
import model
import os
import utils

class ProjectOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('-name', type=str, default='final', help='name of the experiment.')
        self.parser.add_argument('-checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('-model',type=str,default='reco',help='model name')
        self.parser.add_argument('-phase',type=str,default='train',help='train or test')
        self.parser.add_argument('-gpu_idx', type=int, default=0, help='gpu index, -1 for CPU')
        self.parser.add_argument('-data_root',  type=str,default='./images', help='data root')
        self.parser.add_argument('-result_root', type=str, default='./results', help='root of saving results ')
        self.parser.add_argument('-print_net',action='store_true', help='bool for printing net')

    def add_train_options(self):
        self.parser.add_argument('-patch_size', type=int, default=-1, help='crop size ,<1 for no crop')
        self.parser.add_argument('-batch_size',type=int,default=4,help='train batch size')
        self.parser.add_argument('-print_freq',  type=int, default=10, help='loss print frequency')
        self.parser.add_argument('-net_save_freq', type=int, default=1,help='epochs for saving net')
        self.parser.add_argument('-im_save_freq', type=int, default=2000, help=' saving image frequency for train phase')
        self.parser.add_argument('-epochs',  type=int, default=100, help='epochs of train')

    def add_test_options(self):
        self.parser.add_argument('-patch_size', type=int, default=-1, help='crop size, <1 for no crop')
        self.parser.add_argument('-batch_size',type=int,default=1,help='test batch size')

    def gather_options(self):
        """Add additional model-specific options"""

        if not self.initialized:
            self.initialize()
        # get basic options
        opt, _ = self.parser.parse_known_args()
        if opt.phase=='train':
            self.isTrain=True
            self.add_train_options()
        else:
            self.isTrain=False
            self.add_test_options()
        # modify the options for different models
        model_option_set = model.get_option_setter(opt.model)
        parser = model_option_set(self.parser)
        opt = parser.parse_args()
        return opt

    def get_opt(self):
        self.opt = self.gather_options()
        self.opt.isTrain=self.isTrain
        if self.opt.isTrain:
            self.opt.im_save_dir = './%s/train_%s_%s_%s'%(self.opt.result_root,self.opt.model,self.opt.name,self.opt.dir_in)
        else:
            self.opt.im_save_dir = './%s/test_%s_%s_%s'%(self.opt.result_root,self.opt.model,self.opt.name,self.opt.dir_in)
        if self.opt.patch_size<1:
            self.opt.patch_size=False
        return self.opt

    @staticmethod
    def print_options(opt):
        """print and save options"""

        print('--------------Options--------------')
        for k, v in sorted(vars(opt).items()):
            print('%s: %s' % (str(k), str(v)))
        print('----------------End----------------')

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir,opt.model+'_'+opt.name)
        utils.init_folder(expr_dir)
        if opt.isTrain:
            file_name = os.path.join(expr_dir, 'train_opt.txt')
        else:
            file_name = os.path.join(expr_dir, 'test_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('--------------Options--------------\n')
            for k, v in sorted(vars(opt).items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('----------------End----------------\n')
            opt_file.close()