from data_cfg import Project_Dataset
from tensorboardX import SummaryWriter
from utils import init_folder
from options import ProjectOptions
from model import create_model
import time 
import os 
import torch

if __name__ == '__main__':
    opt=ProjectOptions().get_opt()
    ProjectOptions.print_options(opt)
    nets_path=os.path.join(opt.checkpoints_dir,opt.model+'_'+opt.name)
    init_folder(nets_path,opt.im_save_dir)
    data_set = Project_Dataset(opt)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=opt.batch_size, shuffle=True)
    length=len(data_loader)
    model=create_model(opt)
    model.load_networks()
    if opt.print_net:
        model.print_networks()
    if opt.phase=='train':
        log_dir=os.path.join(nets_path,'./log')
        init_folder(log_dir)
        print('Start train')
        writer = SummaryWriter(log_dir)
        for e in range(opt.epochs):
            epoch=e+1
            model.clear_sumloss()
            for i,data in enumerate(data_loader,0):
                model.set_input(data)
                model.train()
                model.write(writer,e*length+i+1)
                if (i+1)%opt.print_freq==0:
                    model.print_loss(opt.name,epoch,i,length)
                if (e*length+i+1)%opt.im_save_freq==0:
                    print('save img')
                    model.save_results()
            if epoch%opt.net_save_freq==0:
                model.save_networks()
        print('End')
    if opt.phase=='test':
        print('Start test')
        for i,data in enumerate(data_loader,0):
            model.set_input(data)
            model.forward()
            model.save_results()
        print('End')


