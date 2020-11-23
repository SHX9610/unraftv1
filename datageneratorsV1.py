"""
*Preliminary* pytorch implementation.
data generators for voxelmorph
"""

import numpy as np
import sys
import glob
import os
import random
import torch

import torch.utils.data as data
import SimpleITK as sitk



def example_gen_gz(moving_dirs,fixed_names,device):
    """
    generate examples
    Parameters:
        vol_names: a list or tuple of filenames
        batch_size: the size of the batch (default: 1)
        The following are fairly specific to our data structure, please change to your own
        return_segs: logical on whether to return segmentations
        seg_dir: the segmentations directory.
    """
    deal_dataset = []
    moving_names = glob.glob(os.path.join(moving_dirs, '*.gz'))
    for i in range(len(moving_names)):
        moving_img =  sitk.ReadImage(moving_names[i])
        moving_img = sitk.GetArrayFromImage(moving_img)[np.newaxis,  np.newaxis, ...]
        fixed_img = sitk.ReadImage(fixed_names)
        fixed_img = sitk.GetArrayFromImage(fixed_img)[np.newaxis, np.newaxis, ...]
        input_fixed = torch.from_numpy(fixed_img).to(device).float()
        input_moving = torch.from_numpy(moving_img).to(device).float()
        deal_dataset += data.TensorDataset(input_moving,input_fixed)
    return deal_dataset

def fetch_dataloader_gz(movingdirs,fixednames,batchsize,device):

    dealdataset = example_gen_gz(movingdirs,fixednames,device)
    train_loader = data.DataLoader(dealdataset,batch_size=batchsize,pin_memory=False,shuffle=True,num_workers=0,drop_last=True)
    return train_loader

def example_gen_npy(moving_dirs,fixed_dirs,device):
    """
    generate examples
    Parameters:
        vol_names: a list or tuple of filenames
        batch_size: the size of the batch (default: 1)
        The following are fairly specific to our data structure, please change to your own
        return_segs: logical on whether to return segmentations
        seg_dir: the segmentations directory.
    """
    deal_dataset = []
    moving_names = glob.glob(os.path.join(moving_dirs, '*.npy'))
    fixed_names = glob.glob(os.path.join(fixed_dirs,'*.npy'))
    for i in range(len(moving_names)):
        moving_img =  np.load(moving_names[i])
        moving_img = moving_img[np.newaxis,  np.newaxis, ...]
        fixed_img = np.load(fixed_names[i])
        fixed_img = fixed_img[np.newaxis, np.newaxis, ...]
        input_fixed = torch.from_numpy(fixed_img).to(device).float()
        input_moving = torch.from_numpy(moving_img).to(device).float()
        deal_dataset += data.TensorDataset(input_moving,input_fixed)
    return deal_dataset

def fetch_dataloader_npy(movingdirs,fixednames,batchsize,device):

    dealdataset = example_gen_npy(movingdirs,fixednames,device)
    train_loader = data.DataLoader(dealdataset,batch_size=batchsize,pin_memory=False,shuffle=True,num_workers=0,drop_last=True)
    return train_loader

if __name__ =='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = "cuda"
    moving_dir= 'E:\peizhunsd\LPBA40\\train\\'
    fixed_name = 'E:\peizhunsd\LPBA40\\fixed.nii.gz'
    train_loader = fetch_dataloader_gz(moving_dir,fixed_name,batchsize=2,device=device)
    for i,data_blob in enumerate(train_loader):
        # print(i,data_blob)
        image1, image2 = [x.cuda() for x in data_blob]
        print(image1.shape,image2.shape)







