import glob
import ipyvolume
import cv2
import numpy as np
import os
import pythreejs

levels = {      
           'UCSD900':
          {  'SR': { '10': [0.10, 0.35], '15': [0.10, 0.35], '20': [0.10, 0.35], '25': [0.10, 0.35], '30': [0.10, 0.35], '40': [0.10, 0.35]},
             'LR': { '10': [0.20, 0.45], '15': [0.20, 0.60], '20': [0.20, 0.65], '25': [0.20, 0.75], '30': [0.20, 0.80], '40': [0.20, 0.80]},
             'HR': [1.00, 0.75]
          },
           'S1':
          {  'SR': { '10': [0.35, 0.00], '15': [0.35, 0.00], '20': [0.35, 0.00], '25': [0.35, 0.00] },
             'LR': { '10': [0.50, 0.00], '15': [0.50, 0.00], '20': [0.50, 0.00], '25': [0.50, 0.00] },
             'HR': [1.00, 0.75]
          }  
}
opacities = {
           'UCSD900':
             { 'SR': { '10': [0.00,0.20,0.00], '15': [0.00,0.20,0.00], '20': [0.00,0.20,0.00], '25': [0.00,0.20,0.00]},
               'LR': { '10': [0.00,0.20,0.00], '15': [0.00,0.20,0.00], '20': [0.00,0.20,0.00], '25': [0.00,0.20,0.00]},
               'HR': [0.70,0.73,0.00]
             },
           'S1':
             { 'SR': { '10': [0.20,0.00,0.00], '15': [0.20,0.00,0.00], '20': [0.20,0.00,0.00], '25': [0.20,0.00,0.00]},
               'LR': { '10': [0.15,0.00,0.00], '15': [0.15,0.00,0.00], '20': [0.15,0.00,0.00], '25': [0.15,0.00,0.00]},
               'HR': [0.70,0.73,0.00]
             }
}

fig_res = {} # dictionary of result figures
fig_inp = {} # dictionary of input figures

figcam = pythreejs.PerspectiveCamera(aspect=0.8, fov=46.0, matrixWorldNeedsUpdate=True, position=(1.785528946985704, 1.3757523415625494, 2.412892214337929), projectionMatrix=(2.9448154572796907, 0.0, 0.0, 0.0, 0.0, 2.3558523658237527, 0.0, 0.0, 0.0, 0.0, -1.00010000500025, -1.0, 0.0, 0.0, -0.200010000500025, 0.0), quaternion=(-0.19796316984919116, 0.30892794387643674, 0.08046673488216134, 0.9267681551784899), rotation=(-0.5181948251143776, 0.5713287660544955, 0.3285944197012996, 'XYZ'), scale=(1.0, 1.0, 1.0), up=(-0.27146072492949946, 0.9086713759213286, -0.31721507751364897))

noises = ['10','15','20','25']
sample = 'S1'
frame_range = [200,509,0,509]


import time
for noise in ['10','15']:
    print(noise)
    avg_img = np.load('result_vol_npys/output_'+sample+'_vol_'+noise+'.npy')
    avg_img = avg_img[:,frame_range[0]:frame_range[1],frame_range[2]:frame_range[3]]
    mins = avg_img.min(axis=(0,1))
    maxs = avg_img.max(axis=(0,1))
    avg_img_norm = (avg_img - mins) / (maxs - mins)
    fig_res[noise] = ipyvolume.figure(camera = figcam)
    ipyvolume.volshow(avg_img_norm[:,:,:],level=levels[sample]['SR'][noise],opacity=opacities[sample]['SR'][noise])
    ipyvolume.selector_default()
    fig_res[noise].volumes[0].opacity_scale = 5.00
    fig_res[noise].volumes[0].brightness = 2.00
    ipyvolume.pylab.xyzlabel(' ', ' ', ' ')
    ipyvolume.show()
    time.sleep(5)
    ipyvolume.pylab.savefig('output_3dfig_'+sample+'_'+noise+'.png',width=1280,height=720)
    print(fig_res[noise].volumes )
    fig_res[noise].volumes = None;
    ipyvolume.show()
    input_vol = np.load('input_'+sample+'_vol_'+noise+'.npy')
    input_vol = input_vol[:,frame_range[0]:frame_range[1],frame_range[2]:frame_range[3]]
    mins = input_vol.min(axis=(0,1))
    maxs = input_vol.max(axis=(0,1))
    input_vol_norm = (input_vol - mins) / (maxs - mins)
    #fig_inp[noise] = 
    #ipyvolume.figure(camera = figcam)
    ipyvolume.volshow(input_vol_norm[:,:,:],level=levels[sample]['LR'][noise],opacity=opacities[sample]['LR'][noise],fig=ipyvolume.gcf())
    #ipyvolume.selector_default()
    #fig_inp[noise].volumes[0].opacity_scale = 5.00
    #fig_inp[noise].volumes[0].brightness = 2.00
    ipyvolume.pylab.xyzlabel(' ', ' ', ' ')
    ipyvolume.show()
    time.sleep(5)
    ipyvolume.pylab.savefig('input_3dfig_'+sample+'_'+noise+'.png',width=1280,height=720)
    ipyvolume.pylab.clear()
