from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianDiffusion(nn.Module):
    def __init__(self, model, steps, sample_steps):
        super().__init__()
        # register environment variables
        self.train_steps = steps
        self.sample_steps = sample_steps
        self.model = model

        # register computation variables for training
        betas = torch.linspace(start=1e-4, end=0.005, steps=self.train_steps)
        alphas = 1. - betas
        self.register_buffer("alphas_bar", torch.cumprod(alphas, dim=0))
        
        # register computation variables for sampling
        sample_betas = torch.linspace(start=1e-4, end=0.1, steps=self.sample_steps)
        sample_alphas = 1. - sample_betas
        self.register_buffer("sample_alphas_bar", torch.cumprod(sample_alphas, dim=0))
        self.register_buffer("sample_one_over_sqrt_alphas", torch.sqrt(1. / sample_alphas))
        self.register_buffer("sample_betas_over_sqrt_one_minus_alphas_bar", sample_betas / torch.sqrt(1. - self.sample_alphas_bar))
        self.register_buffer("sample_sigmas", torch.sqrt(sample_betas))

    def _extract(self, values, times, dimension_num):
        B, *_ = times.shape
        selected_values = torch.gather(values, dim=0, index=times)
        return selected_values.reshape((B, *[1 for _ in range(dimension_num-1)])) # to broadcast coefficients

    def sample(self, x_T, x_c):
        # Algorithm 2
        x_c = self._norm(x_c)
        B, *_ = x_T.shape
        dimension_num = len(x_T.shape)

        x_t = x_T
        for step in tqdm(reversed(range(self.sample_steps)), total=self.sample_steps, desc=f'[Img.]', leave=False):

            time = step * x_t.new_ones((B, ), dtype=torch.int64)
            gamma = self._extract(self.sample_alphas_bar, time, dimension_num)
            epsilon = self.model(torch.cat((x_t, x_c), dim=1), torch.sqrt(gamma))

            one_over_sqrt_alpha = self._extract(self.sample_one_over_sqrt_alphas, time, dimension_num)
            beta_over_sqrt_one_minus_alpha_bar = self._extract(self.sample_betas_over_sqrt_one_minus_alphas_bar, time, dimension_num)
            sigma = self._extract(self.sample_sigmas, time, dimension_num)
            z = torch.randn_like(x_t, device=x_t.device) if step > 0 else 0
            # get x_{t-1} from x_{t} and set to new x_{t}
            x_t = one_over_sqrt_alpha * ( x_t - beta_over_sqrt_one_minus_alpha_bar * epsilon ) + sigma * z

        x_0 = x_t
        return self._denorm(torch.clip(x_0, -1, 1))
    
    def forward(self, x_0, x_c):
        # Algorithm 1
        B, *_ = x_0.shape
        dimension_num = len(x_0.shape)
        x_0 = self._norm(x_0)
        x_c = self._norm(x_c)
        
        time = torch.randint(low=1, high=self.train_steps, size=(B, ), device=x_0.device)
        
        gamma_high = self._extract(self.alphas_bar, time-1, dimension_num)
        gamma_low = self._extract(self.alphas_bar, time, dimension_num)

        gamma = (gamma_high-gamma_low) * torch.rand_like(gamma_high, device=gamma_high.device) + gamma_low

        epsilon = torch.randn_like(x_0, device=x_0.device)

        x_t =  torch.sqrt(gamma) * x_0 + torch.sqrt(1-gamma) * epsilon

        loss = F.mse_loss(self.model(torch.cat((x_t, x_c), dim=1), torch.sqrt(gamma)), epsilon, reduction='mean')
        
        return loss
    
    def _norm(self, img):
        return (img - 0.5) * 2.0
    
    def _denorm(self, img):
        return img / 2.0 + 0.5
    




