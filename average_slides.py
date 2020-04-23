import numpy as np
import cv2
import sys
import os
import glob

import pdb

data_folder = sys.argv[1]
input_folder = sys.argv[2]
noise = sys.argv[3]
"""
print('Data folder: {}'.format(data_folder))
pngs = glob.glob(os.path.join(data_folder,'*SR.png'))
inds = list(map(lambda x: int(x.split('/')[-1].split('_')[2]),pngs))
png_files = [files for _,files in sorted(zip(inds,pngs))]
#print(os.path.join(data_folders,'*.png'),pngs)
slide = 10
nslide = 31
full_img = np.zeros((90,390,82),dtype=float)
full_img_int = np.zeros((90,390,82),dtype=int)
for nimg, pngf in enumerate(png_files):
    img = np.array(cv2.resize(cv2.imread(pngf),(90,90)))[:,:,0]
    #img = (img - img.min()) / (img.max() - img.min())
    nframe = int(nimg / nslide)
    ns = nimg - nframe * nslide
    full_img[:,(ns*slide):(ns*slide+90),nframe] = full_img[:,(ns*slide):(ns*slide+90),nframe] + img
    full_img_int[:,(ns*slide):(ns*slide+90),nframe] = full_img_int[:,(ns*slide):(ns*slide+90),nframe] + np.ones((90,90),dtype=int)
    print(pngf, nframe, ns, img.shape)
avg_img = full_img / full_img_int
"""
print('Input folder: {}'.format(input_folder))
input_jpgs = sorted(glob.glob(os.path.join(input_folder,'*_'+noise+'_*input.jpg')))
target_jpgs = sorted(glob.glob(os.path.join(input_folder,'*_'+noise+'_*target.jpg')))

input_vol = []
target_vol = []
for inp,tar in zip(input_jpgs,target_jpgs):
    ind = int(inp.split('/')[-1].split('_')[0]) - 1
    input_vol.append(cv2.imread(inp))
    target_vol.append(cv2.imread(tar))
    #print(ind,inp,tar)

input_vol = np.transpose(input_vol,[1,2,0,3])[:,:,:,0]
target_vol = np.transpose(target_vol,[1,2,0,3])[:,:,:,0]
np.save('ucsd_vol_'+noise+'.npy',input_vol)
np.save('ucsd_vol_target.npy',target_vol)
