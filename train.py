import os
import copy

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms

from utils import Utils
from metrics import Metrics
from data import DF2KTrainDataset, DIV2KValDataset, Flickr2KTestDataset
from diffusion import GaussianDiffusion

import torch.distributed as dist

class DiffTrainer(Utils):
    # register objects needed for both training and testing
    def _setup_exec_env(self, virtual_device, ngpus_per_node, settings):

        settings['virtual_device'] = virtual_device 
        settings['master'] = virtual_device == 0
        settings['device'] = settings['user_set_devices'][virtual_device] if settings['mgpu'] and settings['user_set_devices'] != None else virtual_device
        
        net = settings['model'](*settings['args'], steps=settings['steps'])
        net = GaussianDiffusion(net, steps=settings['steps'], sample_steps=settings['sample_steps'])

        settings['net'] = self.define_model(
            net=net, \
            ngpus_per_node=ngpus_per_node, \
            virtual_device=virtual_device, \
            device=settings['device'], \
            master=settings['master'], \
            mgpu=settings['mgpu'], \
            addr=settings['addr'], \
            port=settings['port'] \
        )

        settings['ema_net'] = copy.deepcopy(net)
        settings['upsample'] = nn.Upsample(scale_factor=4.0, mode='bicubic').to(settings['device'])
        
        for k, v in settings.items():
            setattr(self, k, v)

        return

    # external call function to do trainging
    def setup_and_train(self, virtual_device, ngpus_per_node, settings, resume):
        self.train_batch_size=settings['train_batch_size']
        self.eval_batch_size=settings['eval_batch_size']
        self.workers=settings['workers']
        self.report_img_idxs=settings['report_img_idx']
        self.report_img_per=settings['report_img_per']
        
        self._setup_exec_env(virtual_device, ngpus_per_node, settings)
        self._setup_train_env()

        # do train
        self._train_network(resume)
        return
   
    # register objects needed to perform training
    def _setup_train_env(self):
        # define train dataloader
        train_dataset = DF2KTrainDataset()
        self.train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True) if self.mgpu else None
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, num_workers=self.workers, sampler=self.train_sampler, pin_memory=True)

        # register train variables 
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.iter_per_epoch = len(self.train_dataloader)
        self.epochs = (self.iters + self.iter_per_epoch - 1) // self.iter_per_epoch
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs*self.iter_per_epoch, eta_min=1e-7)

        # define valid dataloader
        self.valid_dataset = DIV2KValDataset()
        valid_sampler = DistributedSampler(self.valid_dataset, shuffle=False, drop_last=False) if self.mgpu else None
        self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.eval_batch_size, num_workers=self.workers, sampler=valid_sampler, pin_memory=True)

        # generate sample x_T
        generator = torch.Generator().manual_seed(2147483647) # to generate same sample x_T 
        img_size = self.valid_dataset.crop_size
        scale_up = self.valid_dataset.scale_factor
        report_img_size = int(img_size * scale_up)
        self.sample_x_T = torch.randn((len(self.report_img_idxs), 3, report_img_size, report_img_size), generator=generator).to(self.device)

        self.writer = SummaryWriter(self.log_path) if self.master else None
        return
    
    # store current training state
    def _store_train_env(self, epoch):
        last_path, old_path = self._gen_path(self.point_path)
        
        # extract last state
        last_states = {
            'net': self.net.module.state_dict() if self.mgpu else self.net.state_dict(),
            'ema_net': self.ema_net.module.state_dict() if self.mgpu else self.ema_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch+1,
        }
        
        # update old result for backup
        if os.path.exists(old_path):
            os.remove(old_path)
        if os.path.exists(last_path):
            os.rename(last_path, old_path)

        # store last state
        torch.save(last_states, last_path)
        
        return

    # load stored training state
    def _load_train_env(self, resume):
        if resume:
            states = self.get_states(self.point_path, self.mgpu, self.user_set_devices)
            self.load_model(self.net, states['net'], self.mgpu)
            self.load_model(self.ema_net, states['ema_net'], self.mgpu)
            self.optimizer.load_state_dict(states['optimizer'])
            self.scheduler.load_state_dict(states['scheduler'])
            epoch = states['epoch']
        else:
            epoch = 0
        
        return epoch

    # train and validate all epochs and report results
    def _train_network(self, resume):
        # load train env
        epoch = self._load_train_env(resume)

        # define process bar
        pbar = tqdm(total=self.epochs, desc=f'[Train]', smoothing=1.0)

        # restore process bar
        if epoch > 0:
            pbar.update(epoch)
            pbar.refresh()
        
        # perform training
        while epoch < self.epochs:
            t_loss =  self._train(epoch)
            v_loss, img, lbl, sample = self._valid(epoch)
            # summary
            if self.master:
                self._store_train_env(epoch)
                self.writer.add_scalar('Train Loss', t_loss, epoch)
                self.writer.add_scalar('Valid Loss', v_loss, epoch)
                if sample != None:
                    self.writer.add_images('Valid Images/A. Bicubic', img, epoch)
                    self.writer.add_images('Valid Images/B. Sample', sample, epoch)
                    self.writer.add_images('Valid Images/C. GT', lbl, epoch)
            
            epoch += 1
            pbar.update(1)
            pbar.refresh()
        
        pbar.close()

        if self.master:
            self.writer.close()

        return
    
    # train one epoch
    def _train(self, epoch):
        
        self.net.train()
        self.ema_net.train()

        if self.train_sampler != None:
            self.train_sampler.set_epoch(epoch)
        
        t_loss_tot = 0.0
        for img, lbl in tqdm(self.train_dataloader, desc=f'Epoch {epoch+1}/{self.epochs} [Train]', leave=False):
            
            self.optimizer.zero_grad()

            img = torch.clip(self.upsample(img.to(self.device)), 0.0, 1.0)
            lbl = lbl.to(self.device)

            t_loss = self.net(lbl, img)
            t_loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1) # prevent gradient exploding
            self.optimizer.step()
            self.scheduler.step() 

            t_loss_tot += t_loss.item()

            self.exec_ema(self.net, self.ema_net)        
        
        if self.mgpu:
            dist.all_reduce(t_loss_tot, op=dist.ReduceOp.SUM)
            N = dist.get_world_size()
            t_loss_tot /= N

        t_loss_avg = t_loss_tot/len(self.train_dataloader)

        return t_loss_avg
    
    # validate one epoch
    def _valid(self, epoch):
        self.net.eval()
        self.ema_net.eval()
        
        v_loss_tot = 0.0
        with torch.no_grad():
            for img, lbl in tqdm(self.valid_dataloader, desc=f'Epoch {epoch+1}/{self.epochs} [Valid]', leave=False):

                img = torch.clip(self.upsample(img.to(self.device)), 0.0, 1.0)
                lbl = lbl.to(self.device)

                v_loss = self.ema_net(lbl, img)
                v_loss_tot += v_loss.item()

            if self.mgpu:
                N = dist.get_world_size()
                dist.all_reduce(v_loss_tot, op=dist.ReduceOp.SUM)
                v_loss_tot /= N

            v_loss_avg = v_loss_tot/len(self.valid_dataloader)

            img, lbl, sample = self._sample_img(self.valid_dataset, random_sample=False) if epoch%self.report_img_per == 0 else (None, None, None)

        return v_loss_avg, img, lbl, sample
    
    # external call to perform testing
    def setup_and_test(self, virtual_device, ngpus_per_node, settings, resume):
        self.train_batch_size=settings['train_batch_size']
        self.eval_batch_size=settings['eval_batch_size']
        self.workers=settings['workers']
        self.report_img_idxs=settings['report_img_idx']
        self.report_img_per=settings['report_img_per']
        
        if not hasattr(self, 'ema_net'):
            self._setup_exec_env(virtual_device, ngpus_per_node, settings)
        self._setup_test_env(virtual_device, ngpus_per_node)

        # do test
        self._test_network(resume)
        return

    # register objects needed to perform tesing
    def _setup_test_env(self, virtual_device, ngpus_per_node):
        # define dataloader
        self.test_dataset = Flickr2KTestDataset()
        test_sampler = DistributedSampler(self.test_dataset, shuffle=False, drop_last=False) if self.mgpu and self.is_divisible(self.test_dataset, ngpus_per_node) else None
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.eval_batch_size, num_workers=self.workers, sampler=test_sampler, pin_memory=True)
        
        # define scores
        self.scores = self.define_model(
            net=Metrics(), \
            ngpus_per_node=ngpus_per_node, \
            virtual_device=virtual_device, \
            device=self.device, \
            master=self.master, \
            mgpu=self.mgpu, \
            addr=self.addr, \
            port=self.port \
        )

        # setup scores
        self.scores.setup(self.test_dataloader, self.device)
        self.scores.clear()

        self.writer = SummaryWriter(self.log_path) if self.master else None
        return
    
    # store current test state
    def _store_test_env(self):
        self.scores.store_states(self.states_path, self.virtual_device)
        return

    # load trained model and stored test state
    def _load_test_env(self, resume):
        states = self.get_states(self.point_path, self.mgpu, self.user_set_devices)
        self.load_model(self.ema_net, states['ema_net'], self.mgpu)
        if resume:
            self.scores.load_states(self.states_path, self.virtual_device)
        else:
            self.scores.clear()
        start_idx = (self.scores.get_stored_samples_num()+self.eval_batch_size-1)//self.eval_batch_size
        return start_idx
    
    # test network and report results
    def _test_network(self, resume):
        # evaluate scores
        psnr, ssim, is_mean, is_std, fid, img, lbl, sample = self._test(resume)
        # summary
        if self.master:
            self.writer.add_scalars('Test IS', {'mean': is_mean, 'std':is_std}, 0)
            self.writer.add_scalar('Test FID', fid, 0)
            self.writer.add_scalar('Test PSNR', psnr, 0)
            self.writer.add_scalar('Test SSIM', ssim, 0)
            self.writer.add_images('Test Images/A. Bicubic', img, 0)
            self.writer.add_images('Test Images/B. Sample', sample, 0)
            self.writer.add_images('Test Images/C. GT', lbl, 0)
            print(f'> [Stats.] | IS: ({is_mean:.3f}, {is_std:.3f}) | FID: {fid:.3f} | PSNR: {psnr:.3f} | SSIM: {ssim:.3f}')

            self.writer.close()

        return

    # perform testing
    def _test(self, resume):

        start_idx = self._load_test_env(resume)
        
        self.ema_net.eval()

        with torch.no_grad():
            for idx, (img, lbl) in enumerate(tqdm(self.test_dataloader, desc=f'[Test]', smoothing=1.0)):
                if idx < start_idx:
                    continue
                else:
                    img = torch.clip(self.upsample(img.to(self.device)), 0.0, 1.0)
                    lbl = lbl.to(self.device)
                    
                    x_T = torch.randn_like(img).to(self.device)
                    x_0 = self.ema_net.sample(x_T, img)

                    self.scores.update(x_0, lbl)
                    self._store_test_env()
                    
            # gather all results
            if self.mgpu:
                N = dist.get_world_size()
                
                psnr_sum, ssim_sum, is_acts, fid_acts = self.scores.get_states()
                is_acts_gather = [torch.ones_like(is_acts) for _ in range(N)]
                fid_acts_gather = [torch.ones_like(fid_acts) for _ in range(N)]
                
                dist.all_gather(is_acts_gather, is_acts)
                dist.all_gather(fid_acts_gather, fid_acts)
                dist.all_reduce(psnr_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(ssim_sum, op=dist.ReduceOp.SUM)

                is_acts = torch.cat(is_acts_gather)
                fid_acts = torch.cat(fid_acts_gather)
                self.scores.set_states(psnr_sum, ssim_sum, is_acts, fid_acts)

            psnr, ssim, (is_mean, is_std), fid = self.scores()
            img, lbl, sample = self._sample_img(self.test_dataset, random_sample=True)
        
        return psnr, ssim, is_mean, is_std, fid, img, lbl, sample
    
    # sample image
    def _sample_img(self, dataloader, random_sample):
        imgs = list()
        lbls = list()
        for idx in self.report_img_idxs:
            img, lbl = dataloader.__getitem__(idx)
            imgs.append(img)
            lbls.append(lbl)
                
        img = torch.stack(imgs).to(self.device)
        lbl = torch.stack(lbls).to(self.device)

        img = torch.clip(self.upsample(img), 0.0, 1.0)

        if random_sample:
            x_T = torch.randn_like(img).to(self.device)
        else:
            x_T = self.sample_x_T
        
        sample = self.ema_net.sample(x_T, img)
    
        return img, lbl, sample
    
    
    

