import os

import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

from skimage.metrics import structural_similarity as ssim

class Metrics(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False) # use original Inception model and give [-1, 1] as a input
        self.model.eval()

        self.results = dict()
        self.model.fc.register_forward_hook(self._get_result('IS', self.results))
        self.model.avgpool.register_forward_hook(self._get_result('FID', self.results))

        self.upsample = nn.Upsample(size=(299,299), mode='bilinear')
        self.softmax = nn.Softmax(dim=-1) # for IS

        self.fid_activations = None
        self.is_activations = None

        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
    
        self.eps = 1e-8 # to prevent log(0)
        self.crop_out = 8
    
    # calculate GT dataset mean and sigma used to calculate FID score; to reduce execution time during test
    def setup(self, eval_dataloader, device):
        for _, lbl in eval_dataloader:
            lbl = lbl.to(device)
            self._update_acts(lbl)
        self.mean_gt, self.sigma_gt = self._calc_fid_stats()
        return
    
    def update(self, img, lbl):
        self._update_acts(img)
        self._update_scalars(img, lbl)
        return
    
    # calculate scores
    def forward(self):
        B = self.get_stored_samples_num()
        psnr = self.psnr_sum / B
        ssim = self.ssim_sum / B
        inception_score = self._calc_is()
        fid_score = self._calc_fid()

        return psnr, ssim, inception_score, fid_score
    
    def set_states(self, psnr_sum, ssim_sum, is_acts, fid_acts):
        self.psnr_sum = psnr_sum
        self.ssim_sum = ssim_sum
        self.is_activations = is_acts
        self.fid_activations = fid_acts
        return
    
    def store_states(self, path, virtual_device):
        last_states = {
            'psnr_sum': self.psnr_sum,
            'ssim_sum': self.ssim_sum,
            'is_activations': self.is_activations,
            'fid_activations': self.fid_activations,
        }

        last_path, old_path = self._gen_path(path, virtual_device)

        if os.path.exists(old_path):
            os.remove(old_path)
        if os.path.exists(last_path):
            os.rename(last_path, old_path)

        torch.save(last_states, last_path)
        
        return
    
    def load_states(self, path, virtual_device):
        last_path, _ = self._gen_path(path, virtual_device)
        states = torch.load(last_path)

        psnr_sum = states['psnr_sum']
        ssim_sum = states['ssim_sum']
        is_acts = states['is_activations']
        fid_acts = states['fid_activations']

        self.set_states(psnr_sum, ssim_sum, is_acts, fid_acts)

        return

    def get_stored_samples_num(self):
        return self.is_activations.shape[0] if self.is_activations != None else 0
    
    def get_states(self):
        return self.psnr_sum, self.ssim_sum, self.is_activations, self.fid_activations

    # clear caches; initialize internal variables
    def clear(self):
        self.set_states(0.0, 0.0, None, None)
    
    def _update_acts(self, img):
        with torch.no_grad():
            img = self.upsample(img)
            img = self._norm(img)
            _ = self.model(img)
            is_act = self.results['IS'].reshape(1, -1)
            fid_act = self.results['FID'].reshape(1, -1)
            self.is_activations = torch.cat((self.is_activations, is_act), dim=0) if self.is_activations != None else is_act
            self.fid_activations = torch.cat((self.fid_activations, fid_act), dim=0) if self.fid_activations != None else fid_act
        return
    
    def _update_scalars(self, img, lbl):
        with torch.no_grad():
            img_y = self._rgb2y(img)
            lbl_y = self._rgb2y(lbl)
            self.psnr_sum += torch.sum(self._calc_psnr_per_batch(img_y, lbl_y))
            self.ssim_sum += torch.sum(self._calc_ssim_per_batch(img_y, lbl_y))
        return
    
    def _calc_fid_stats(self):
        act = self.fid_activations
        mean = torch.mean(act, axis=0)
        sigma = torch.cov(act.t())
        return mean, sigma
    
    def _calc_fid(self):
        mean_gt, sigma_gt = self.mean_gt, self.sigma_gt
        mean_x0, sigma_x0 = self._calc_fid_stats()
        fid_score = torch.square(mean_gt-mean_x0).sum() + torch.trace(sigma_gt + sigma_x0 - 2*torch.sqrt(sigma_gt*sigma_x0))
        return fid_score

    def _calc_is(self):
        scores = list()
        for acts in torch.split(self.is_activations, len(self.fid_activations)//10):
            posteriors_yx = self.softmax(acts) # lisf of p(y|x); as pytorch inception_v3 does not have softmax, apply softmax to output; shape = (B, 1000)
            probability_y = torch.mean(posteriors_yx, dim=0) # marginal distribtion of y = p(y) = mean{ p(y|x) } on x; shape = (1000)
            entropy = torch.sum(posteriors_yx * torch.log(posteriors_yx + self.eps), dim=1) # sum{ p(y|x) * log( p(y|x) ) } on y; shape = (B)
            cross_entropy = torch.sum(posteriors_yx * torch.log(probability_y + self.eps), dim=1) # sum{ p(y|x) * log( p(y) ) } on y; shape = (B)
            log_sharpness = torch.mean(entropy, dim=0)
            log_diversity = -torch.mean(cross_entropy, dim=0)
            inception_score = torch.exp(log_sharpness + log_diversity) # sharpness x diversity
            scores.append(inception_score)
        
        scores = torch.stack(scores, dim=0)

        return torch.mean(scores), torch.std(scores)

    def _calc_psnr_per_batch(self, img_y, lbl_y):
        crop_out = self.crop_out
        diff = img_y - lbl_y
        mse = torch.mean(diff[:,crop_out:-crop_out,crop_out:-crop_out]**2, dim=(1,2))
        return -10*torch.log10(mse + 1e-10)

    def _calc_ssim_per_batch(self, img_y, lbl_y):
        crop_out = self.crop_out
        img_crop = img_y[:,crop_out:-crop_out, crop_out:-crop_out].to('cpu').numpy()
        lbl_crop = lbl_y[:,crop_out:-crop_out, crop_out:-crop_out].to('cpu').numpy()
        return torch.tensor([ssim(img_crop[i], lbl_crop[i], channel_axis=0, data_range=1.0) for i in range(img_crop.shape[0])])

    def _gen_path(self, path, virtual_device):
        com_path = f'{path}/virtual_device_{virtual_device}'

        if not os.path.exists(com_path):
            os.makedirs(com_path)

        last_path = com_path + '/last.stts'
        old_path = com_path + '/old.stts'
        return last_path, old_path

    # get features from Inception Net.
    def _get_result(self, score_name, container):
        def hook(model, input, output):
            container[score_name] = output.detach()
        return hook
        
    def _norm(self, img):
        return (img - 0.5) * 2.0
    
    def _rgb2y(self, img):
        return (16.0 + 65.481*img[:,0,:,:] + 128.553*img[:,1,:,:] + 24.966*img[:,2,:,:]) / 255.0

    

    
