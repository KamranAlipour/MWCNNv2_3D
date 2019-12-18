import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import scipy.io as sio
from data import common
import numpy as np
# import model
import logging

import imageio

from skimage.metrics import structural_similarity as ssim

logging.basicConfig(filename='training.log', level=logging.INFO)

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        #if (torch.cuda.device_count() > 1):
        #   self.model = nn.DataParallel(self.model)

        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)


        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e5
        self.generate = args.generate # whether we want to generate videos



    def train(self):
        self.scheduler.step()

        self.loss.step()


        epoch = self.scheduler.last_epoch + 1


        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        # self.model_NLEst.train()
        # self.model_KMEst.train()


        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr,_) in enumerate(self.loader_train):
            #print(lr.shape)
            #print(hr.shape)
            lr, hr = self.prepare([lr, hr])
            # print(scale_factor[0,0,0,0])
            timer_data.hold()
            timer_model.tic()
            # _, _, hei, wid = hr.data.size()
            self.optimizer.zero_grad()
            idx_scale = 0

            sr = self.model(lr, idx_scale)
            loss = self.loss(sr, hr)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()


            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        # kernel_test = sio.loadmat('data/Compared_kernels_JPEG_noise_x234.mat')
        scale_list = self.scale #[2,3,4,8]
        self.ckp.add_log(torch.zeros(1, len(scale_list)))
        self.model.eval()
        no_eval = 0
        # self.model_NLEst.eval()
        # self.model_KMEst.eval()
        #print(self.args.gen_set,self.args.gen_set.split('/')[-1])
        result_dir = 'experiment/'+self.args.save+'/results_'+self.args.gen_set.split('/')[-1]
        file_prefix = 'test_case'
        if (self.generate):
           if not os.path.exists(result_dir):
                os.makedirs(result_dir)

        timer_test = utility.timer()
        
        with torch.no_grad():

            for idx_scale, scale in enumerate(scale_list):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                lr_im_list = []
                sr_im_list = []
                hr_im_list = []
                psnr_vals = []
                ssim_vals = []
                mse_vals = []
                psnr_input_vals = []
                ssim_input_vals = []
                mse_input_vals = []
                #tqdm_test = tqdm(self.loader_test, ncols=120)
                tqdm_test = tqdm(self.loader_test)
                for idx_img, (lr, hr) in enumerate(tqdm_test):
                    np.random.seed(seed=0)
                    filename = file_prefix+'_'+str(idx_img).zfill(3)
                    # sz = lr.size()
                    # scale_tensor = torch.ones([1, 1, sz[2], sz[3]]).float() * (scale / 80.0)
                    if not no_eval:
                        lr, hr = self.prepare([lr, hr])
                    else:
                        lr = self.prepare([lr])[0]
                    
                    #sz = lr.size()
                    #scale_tensor = torch.ones([1, 1, sz[2], sz[3]]).float() * (2.0 / 80)
                    
                    # print(lr.size())
                    # hr_ = torch.squeeze(hr_)
                    # hr_ = hr_.numpy()
                    # lr = hr

                    sr = self.model(lr, idx_scale)

                    sr = utility.quantize(sr, self.args.rgb_range)
                    #lr_im_list.append(np.array(lr.cpu())[0,0,:,:])
                    #sr_im_list.append(np.array(sr.cpu())[0,0,:,:])
                    #hr_im_list.append(np.array(hr.cpu())[0,0,:,:])

                    save_list = [sr]
                    psnr_val = utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    ssim_val = ssim(hr[0,0].cpu().data.numpy(), sr[0,0].cpu().data.numpy(), data_range=np.max(sr.cpu().data.numpy()) - np.min(sr.cpu().data.numpy()))
                    if (self.generate):
                        psnr_input_val = utility.calc_psnr(
                            lr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        psnr_input_vals.append(psnr_input_val)
                        psnr_vals.append(psnr_val)
                        ssim_input_val = ssim(hr[0,0].cpu().data.numpy(), lr[0,0].cpu().data.numpy(), data_range=np.max(lr.cpu().data.numpy()) - np.min(lr.cpu().data.numpy()))
                        ssim_vals.append(ssim_val)
                        ssim_input_vals.append(ssim_input_val)
                        
                    eval_acc += psnr_val
                    save_list.extend([lr, hr])
                    # # if not no_eval:
                    # #     eval_acc += utility.calc_psnr(
                    # #         sr, hr, scale, self.args.rgb_range,
                    # #         benchmark=self.loader_test.dataset.benchmark
                    # #     )
                    # #     save_list.extend([lr, hr])
                    #
                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, idx_img, scale)
                if (self.generate):
                #    lr_writer = imageio.get_writer('lr_s2.mp4', fps=10)
                #    sr_writer = imageio.get_writer('sr_s2.mp4', fps=10)
                #    hr_writer = imageio.get_writer('hr_s2.mp4', fps=10)
                #    for (lr_im,sr_im,hr_im) in zip(lr_im_list,sr_im_list,hr_im_list):
                #        lr_writer.append_data(lr_im)
                #        sr_writer.append_data(sr_im)
                #        hr_writer.append_data(hr_im)
                #    lr_writer.close()
                #    sr_writer.close()
                #    hr_writer.close()
                    np.save(result_dir+'/'+self.args.gen_set.split('/')[-1]+'_psnr_vals.npy',np.array(psnr_vals))
                    np.save(result_dir+'/'+self.args.gen_set.split('/')[-1]+'_psnr_input_vals.npy',np.array(psnr_input_vals))
                    np.save(result_dir+'/'+self.args.gen_set.split('/')[-1]+'_ssim_vals.npy',np.array(ssim_vals))
                    np.save(result_dir+'/'+self.args.gen_set.split('/')[-1]+'_ssim_input_vals.npy',np.array(ssim_input_vals))
                   
                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

