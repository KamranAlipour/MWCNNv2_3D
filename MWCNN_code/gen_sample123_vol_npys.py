import numpy as np
import glob
import cv2
samples = ['S1','S2','S3']
noises = ['10','15','20','25']

data_dir = '/home/phacou/MWCNNv2/MWCNN_code/experiment/MWCNN_DeNoising/'
for sample in samples:
    for ni,noise in enumerate(noises):
        model_addr = data_dir+sample+'_results/'
        print(ni,model_addr)
        case_addr = model_addr + noise
        if ni == 0:
            HRpngs = sorted(glob.glob(case_addr+'/GT/*HR.png')) 
        LRpngs = sorted(glob.glob(case_addr+'/noisy/*LR.png')) 
        SRpngs = sorted(glob.glob(case_addr+'/model/*SR.png'))
        hrdata = []
        lrdata = []
        srdata = []
        for (hr,lr,sr) in zip(HRpngs,LRpngs,SRpngs):
            if ni == 0:
                hrf = cv2.imread(hr, 0)
            lrf = cv2.imread(lr, 0)
            srf = cv2.imread(sr, 0)
            lrf = (lrf - lrf.min())/ (lrf.max() - lrf.min())
            srf = (srf - srf.min())/ (srf.max() - srf.min())
            if ni == 0:
                hrdata.append(hrf)
            lrdata.append(lrf)
            srdata.append(srf)
        if ni == 0:
            hrdata = np.array(hrdata)
            np.save('gt_'+sample+'_vol.npy',hrdata)
        lrdata = np.array(lrdata)
        np.save('input_'+sample+'_vol_'+noise+'.npy',lrdata)
        srdata = np.array(srdata)
        np.save('output_'+sample+'_vol_'+noise+'.npy',srdata)
