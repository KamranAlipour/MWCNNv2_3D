import glob
import ipywidgets
import ipyvolume
import cv2
import numpy as np
import os
import time

noises = ['1k'] # '1k' '4k' '2k'
sample = "ucsdLED2sm"

input_folder = '/home/data500/'+sample+'/'
print('Input folder: {}'.format(input_folder))
target_jpgs = sorted(glob.glob(os.path.join(input_folder,'*_'+noises[0]+'_*target.jpg')))
target_vol = []
for tar in target_jpgs:
    target_vol.append(cv2.imread(tar))
target_vol = np.transpose(target_vol,[1,2,0,3])[:,:,:,0]
np.save('result_vol_npys/target_'+sample+'_vol.npy',target_vol)
for noise in noises:
    input_jpgs = sorted(glob.glob(os.path.join(input_folder,'*_'+noise+'_*input.jpg')))
    input_vol = []
    for inp,tar in zip(input_jpgs,target_jpgs):
        ind = int(inp.split('/')[-1].split('_')[0]) - 1
        input_vol.append(cv2.imread(inp))
        #print(ind,inp,tar)
    input_vol = np.transpose(input_vol,[1,2,0,3])[:,:,:,0]
    np.save('result_vol_npys/input_'+sample+'_vol_'+noise+'.npy',input_vol)

width = 390
height = 90
slide = 10 # number of pixels slided
nslide = int((width - height)/slide) + 1 # number of slided frames within each input frame (390 - 90)/10 + 1

folder_prefix = 'experiment/MWCNN_DeNoising_norm/'+sample+'_results'
#noises = list(map(lambda x: x.split('/')[-1], glob.glob(folder_prefix+'/*')))
for n in noises:
    data_folder = os.path.join(folder_prefix,noises[0]) # folder_prefix+n+'/'
    print('Data folder: {}'.format(data_folder))
    pngs = glob.glob(os.path.join(data_folder,'model/*SR.png'))
    nframes = int(len(pngs)/nslide)
    inds = list(map(lambda x: int(x.split('/')[-1].split('_')[2]),pngs))
    png_files = [files for _,files in sorted(zip(inds,pngs))]
    #print(os.path.join(data_folders,'*.png'),pngs)
    full_img = np.zeros((height,width,nframes),dtype=float)
    full_img_int = np.zeros((height,width,nframes),dtype=int)
    for nimg, pngf in enumerate(png_files):
        img = np.array(cv2.resize(cv2.imread(pngf),(height,height)))[:,:,0]
        #img = (img - img.min()) / (img.max() - img.min())
        nframe = int(nimg / nslide)
        ns = nimg - nframe * nslide
        full_img[:,(ns*slide):(ns*slide+height),nframe] = full_img[:,(ns*slide):(ns*slide+height),nframe] + img
        full_img_int[:,(ns*slide):(ns*slide+height),nframe] = full_img_int[:,(ns*slide):(ns*slide+height),nframe] + np.ones((height,height),dtype=int)
        #print(pngf, nframe, ns, img.shape)
    avg_img = full_img / full_img_int
    np.save("result_vol_npys/output_"+sample+"_vol_"+n+".npy", avg_img)
