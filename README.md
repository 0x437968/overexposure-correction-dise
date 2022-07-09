# Over-exposure Correction via Exposure and Scene Information Disentanglement

Pytorch implementation of the paper [Over-exposure Correction via Exposure and Scene Information Disentanglement](https://openaccess.thecvf.com/content/ACCV2020/papers/Cao_Over-exposure_Correction_via_Exposure_and_Scene_Information_Disentanglement_ACCV_2020_paper.pdf).
contact yuhuicao@pku.edu.cn

## Usage 

### Data Preperation

In our experiments, outdoor images and portrait images are from [Place365 dataset](http://places2.csail.mit.edu/download.html). To adjust the images exposure, we use the method proposed in [LECRM](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w43/Ying_A_New_Low-Light_ICCV_2017_paper.pdf). The numpy implementation of LECRM can be found in the file LECRM.py of this repository.

### Train 

To train the model, run the following command line in the source code directory. For calculating style loss, VGG19 model can be downloaded in [here](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth). You may set other parameters based on your experiment setting.

For disentanglement model, the exposure adjust process has be embedded into data_cfg.py and you can place your original data into your data directory to train the model:
```bash
python main.py -model dise -name experiment_name -phase train -data_root yourdataroot --dir_in yourdatadir    
```

For recovery model, you can run LECRM.py to generate overexposed images and run main.py to train the model:
```bash
python main.py -model reco -name experiment_name -phase train -data_root yourdataroot --dir_in overdir --dir_gt gtdir   
```

## Citation

If you find the code helpful in your research or work, please kindly cite our paper.

```
@InProceedings{Cao_2020_ACCV,
    author    = {Cao, Yuhui and Ren, Yurui and Li, Thomas H. and Li, Ge},
    title     = {Over-exposure Correction via Exposure and Scene Information Disentanglement},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {November},
    year      = {2020}
}
```