import os

from data import common

import numpy as np
import scipy.misc as misc
import scipy.io as sio
from scipy.misc import imresize

import torch
import torch.utils.data as data
import h5py

import glob

import cv2
import pdb

class SRData(data.Dataset):
    def __init__(self, args, train=True, benchmark=False):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0
        self.ext = args.ext
        self.slide = args.slide # number of pixels to slide in width
        if train:
            data_dir = args.train_data_dir
            hr_flist = sorted(glob.glob(os.path.join(data_dir,'train*_target.'+self.ext)))
            lr_flist = sorted(glob.glob(os.path.join(data_dir,'train*_input.'+self.ext)))
            #mat = h5py.File('../MWCNN/imdb_gray.mat')
            #self.args.ext = 'mat'
            hrd = []
            lrd = []
            for hrf in hr_flist:
               #print('loading HR file: '+hrf)
               if args.ext == 'npy': 
                   hrd.append(np.load(hrf))
               elif args.ext == 'jpg':
                   hrd.append(np.array(cv2.imread(hrf),dtype=np.float))
               else:
                   raise ValueError('The training dataset has unknown extension.')                                    
            for lrf in lr_flist:
               #print('loading LR file: '+lrf)
               if args.ext == 'npy':
                   lrd.append(np.load(lrf))
               elif args.ext == 'jpg':
                   lrd.append(np.array(cv2.imread(lrf),dtype=np.float))
               else:
                   raise ValueError('The training dataset has unknown extension.')                 
               #lrd.append(np.load(lrf))
            hrd = np.expand_dims(np.array(hrd),axis=3)
            lrd = np.expand_dims(np.array(lrd),axis=3)
            #self.hr_data = mat['images']['labels'][:,:,:,:]
            self.hr_data = hrd
            self.lr_data = lrd
            self.num = self.hr_data.shape[0]
            #print(self.hr_data.shape)

        if self.split == 'test':
            hr_flist = sorted(glob.glob('../data/npy_img/test*_target.'+self.ext))
            lr_flist = sorted(glob.glob('../data/npy_img/test*_input.'+self.ext))
            #self._set_filesystem(args.dir_data)
        if args.generate:
            hr_flist = sorted(glob.glob(args.gen_set+'_*_target.'+self.ext))
            lr_flist = sorted(glob.glob(args.gen_set+'_*_input.'+self.ext))
        self.images_hr = hr_flist #self._scan()
        self.list_len = len(self.images_hr)
        self.img_count = self.list_len
        self.images_lr = lr_flist
        nslides = 1
        if args.slide > 0:
           if args.ext == 'jpg':
               data_height = cv2.imread(self.images_hr[0]).shape[0]
               data_width = cv2.imread(self.images_hr[0]).shape[1]
               nslides = int((data_width - data_height) / args.slide) + 1
        self.nslides = nslides
        self.list_len = nslides * self.list_len
        

    def _scan(self):
        raise NotImplementedError
    #
    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    # def _name_hrbin(self):
    #     raise NotImplementedError

    # def _name_lrbin(self, scale):
    #     raise NotImplementedError

    def __getitem__(self, idx):
        if self.nslides == 1:
            hr, lr = self._load_file(idx)
        else:
            imgid = int(idx / self.nslides)
            hr, lr = self._load_file(imgid)
            slideid = idx - imgid * self.nslides 
            #print('getting item {} means slide {} in frame {}'.format(idx,slideid,imgid))
            height = hr.shape[0]
            hr = hr[:,(slideid*self.slide):(slideid*self.slide+height)]
            lr = lr[:,(slideid*self.slide):(slideid*self.slide+height)]
            if hr.shape[0] != 512 or hr.shape[1] != 512:
                hr = cv2.resize(hr,(512,512))
                lr = cv2.resize(lr,(512,512))
                if hr.max() > hr.min():
                   hr = (hr - hr.min()) / (hr.max() - hr.min())
                if lr.max() > lr.min():
                   lr = (lr - lr.min()) / (lr.max() - lr.min())
                hr = np.expand_dims(hr,axis=0)
                lr = np.expand_dims(lr,axis=0)
        #print ('HR: {}x{} {} {} '.format(hr.shape[0],hr.shape[1],hr.min(),hr.max()))
        #print ('LR: {}x{} {} {} '.format(lr.shape[0],lr.shape[1],lr.min(),lr.max()))
        if self.train:
            lr, hr, scale = self._get_patch(hr,lr)
            #print(hr.shape)
            tensors = common.np2Tensor([lr, hr], self.args.rgb_range)
            lr_tensor = tensors[0]
            hr_tensor = tensors[1]
            #print(hr_tensor.shape)
            return lr_tensor, hr_tensor, '_'
        else:
            #scale = 2
            # scale = self.scale[self.idx_scale]
            lr, hr, _ = self._get_patch(hr, lr)
            tensors = common.np2Tensor([lr, hr], self.args.rgb_range)
            lr_tensor, hr_tensor = tensors[0], tensors[1]
            return lr_tensor, hr_tensor


    def __len__(self):
        #print('get length {}'.format(self.list_len))
        return self.list_len

    def _get_index(self, idx):
        #print('get index {}'.format(idx))
        return idx

    def _load_file(self, idx):
        #print('loading file idx {}'.format(idx))
        idx = self._get_index(idx)
        # lr = self.images_lr[self.idx_scale][idx]
        hr = self.images_hr[idx]
        lr = self.images_lr[idx]
        if self.args.ext == 'npy':
            hr = np.expand_dims(np.load(hr),axis=0)
            lr = np.expand_dims(np.load(lr),axis=0)
        elif self.args.ext == 'jpg':
            hr = np.array(cv2.imread(hr)[:,:,0])
            lr = np.array(cv2.imread(lr)[:,:,0])
        elif self.args.ext == 'img' or self.benchmark:
            filename = hr
            hr = misc.imread(hr)
        elif self.args.ext.find('sep') >= 0:
            filename = hr
            # lr = np.load(lr)
            hr = np.load(hr)
        elif self.args.ext == 'mat' or self.train:
            hr = self.hr_data[idx, :, :, :]
            hr = np.squeeze(hr.transpose((1, 2, 0)))
            filename = str(idx) + '.png'
        else:
            filename = str(idx + 1)

        #filename = os.path.splitext(os.path.split(filename)[-1])[0]
        return hr, lr

    def _get_patch(self, hr, lr):
        patch_size = self.args.patch_size
        #print(hr.shape)
        if self.train:
            scale = self.scale[0]
            if self.args.task_type == 'denoising':
                lr, hr = lr, hr #common.get_patch_noise( hr, patch_size, scale)
            if self.args.task_type == 'SISR':
                lr, hr = common.get_patch_bic( hr, patch_size, scale)
            if self.args.task_type == 'JIAR':
                lr, hr = common.get_patch_compress( hr, patch_size, scale)
            #lr, hr = common.augment([lr, hr])
            return lr, hr, scale
        else:
            scale = self.scale[0]
            if self.args.task_type == 'denoising':
                lr, hr = lr, hr #common.get_img_noise( hr, patch_size, scale)
            if self.args.task_type == 'SISR':
                lr, hr = self._get_patch_test( hr, patch_size, scale)
            if self.args.task_type == 'JIAR':
                lr, hr = common.get_img_compress( hr, patch_size, scale)
            return lr, hr, scale
            # lr = common.add_noise(lr, self.args.noise)


    def _get_patch_test(self, hr, scale):

        ih, iw = hr.shape[0:2]
        lr = imresize(imresize(hr, [int(ih/scale), int(iw/scale)], 'bicubic'), [ih, iw], 'bicubic')
        ih = ih // 8 * 8
        iw = iw // 8 * 8
        hr = hr[0:ih, 0:iw, :]
        lr = lr[0:ih, 0:iw, :]

        return lr, hr




    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

