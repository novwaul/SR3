import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module): 
    def __init__(self, C, steps, channel_expansions=[1,2,4,4,8,8], emb_expansion=4, resblock_per_down_stage=3, drp_rate=0.0): # from the original code; set 0.1 for CIFAR10 and 0.0 for the others.
        super().__init__()
        self.emb = GammaEmbedding(steps=steps, dim=C, exp=emb_expansion)
        self.conv1 = Conv2d(2*3, C, 3)
        att_depth = len(channel_expansions)-2

        depth = len(channel_expansions) 
        last_depth = depth-1
        resblock_per_up_stage = resblock_per_down_stage + 1 # to match block connections between up stage and down stage, where { Down_WideResBlock_1 -> Up_WideResBlock_1 }, { Down_WideResBlock_2 -> Up_WideResBlock_2 }, and { Down_Block -> Up_WideResBlock_3 }

        self.down = nn.ModuleList()
        channels = list()
        in_channel = C
        channels.append(in_channel)
        for d in range(depth):
            out_channel = channel_expansions[d] * C
            for _ in range(resblock_per_down_stage):
                res_block = WideResNetBlock(in_channel, out_channel, emb_dimension=emb_expansion*C, attention=d==att_depth, drp_rate=drp_rate)
                in_channel = out_channel
                self.down.append(res_block)
                channels.append(in_channel)

            if d < last_depth:
                self.down.append(DownBlock(in_channel, in_channel))
                channels.append(in_channel)

        self.mid = nn.ModuleList([
            WideResNetBlock(in_channel, in_channel, emb_dimension=emb_expansion*C, attention=True, drp_rate=drp_rate),
            WideResNetBlock(in_channel, in_channel, emb_dimension=emb_expansion*C, attention=False, drp_rate=drp_rate)
        ])

        self.up = nn.ModuleList()
        for d in reversed(range(depth)):
            out_channel = channel_expansions[d] * C
            for _ in reversed(range(resblock_per_up_stage)):
                res_block = WideResNetBlock(in_channel + channels.pop(), out_channel, emb_dimension=emb_expansion*C, attention=d==att_depth, drp_rate=drp_rate)
                in_channel = out_channel
                self.up.append(res_block)
            
            if d > 0:
                self.up.append(UpBlock(in_channel, in_channel))
        
        del channels

        self.gn = GroupNorm(channel_expansions[0]*C)
        self.silu = nn.SiLU()
        self.conv2 = Conv2d(channel_expansions[0]*C, 3, kernel=3, gain=1e-10)
        
    def forward(self, x, gamma):

        emb = self.emb(gamma)

        connections = list()

        z = self.conv1(x)
        connections.append(z)

        for module in self.down:
            z = module(z, emb) if isinstance(module, WideResNetBlock) else module(z)
            connections.append(z)
        
        for module in self.mid:
            z = module(z, emb)
        
        for module in self.up:
            z = module(torch.cat((z, connections.pop()), dim=1), emb) if isinstance(module, WideResNetBlock) else module(z)
        
        z = self.gn(z)
        z = self.silu(z)
        out = self.conv2(z)

        return out

class GammaEmbedding(nn.Module):
    def __init__(self, steps, dim, exp):
        super().__init__()
        self.linear1 = Linear(dim, exp*dim)
        self.silu = nn.SiLU()
        self.linear2 = Linear(exp*dim, exp*dim)
        self.dim=dim

        x = torch.log(torch.tensor(5000)) / (self.dim//2 - 1) # log( 5000^(1 / (d/2 - 1)) )
        x = torch.exp( torch.arange(0, dim//2) * -x ) # 1 / 5000^(i / (d/2 - 1)) 
        self.register_buffer('x', x.reshape((1, -1)))

    def forward(self, gamma):
        
        x = gamma.reshape((-1, 1)) * self.x # gamma / 5000^(i / (d/2 - 1))
        emb = torch.concat((torch.sin(x), torch.cos(x)), dim=1) # sin( gamma / 5000^(i / (d/2 - 1)) ) and cos( gamma / 10000^(i / (d/2 - 1)) )
        if self.dim % 2 != 0:
            emb = F.pad(emb, pad=(0, 1)) # add zero pad at last
        
        emb = self.linear1(emb)
        emb = self.silu(emb)
        emb = self.linear2(emb)

        return emb # shape = (batch, exp*dim)

class WideResNetBlock(nn.Module): # DDPM ResBlock
    def __init__(self, in_channel, out_channel, emb_dimension, attention, drp_rate):
        super().__init__()
        self.do_attention = attention
        self.is_match = in_channel == out_channel
        self.C = out_channel

        self.gn1 = GroupNorm(in_channel)
        self.silu1 = nn.SiLU()
        self.conv1 = Conv2d(in_channel, out_channel, kernel=3)
        
        self.silu2 = nn.SiLU()
        self.linear1= Linear(emb_dimension, out_channel)

        self.gn2 = GroupNorm(out_channel)
        self.silu3 = nn.SiLU()
        self.dropout = nn.Dropout(drp_rate) 
        self.conv2 = Conv2d(out_channel, out_channel, kernel=3, gain=1e-10)

        if not self.is_match: # to match 'channel' betweem 'x' and 'z'
            self.linear2 = Linear(in_channel, out_channel)
        
        if self.do_attention:
            self.att = SelfAttentionBlock(out_channel) 
    

    def forward(self, x, emb):
        z = self.gn1(x)
        z = self.silu1(z)
        z = self.conv1(z)

        B = x.shape[0]
        C = self.C
        emb = self.silu2(emb)
        z = self.linear1(emb).reshape(B, C, 1, 1) + z

        z = self.gn2(z)
        z = self.silu3(z)
        z = self.dropout(z) 
        z = self.conv2(z)

        if not self.is_match:
            x = x.permute(0, 2, 3, 1) # shape=(B,C,H,W) -> (B,H,W,C)
            x = self.linear2(x)
            x = x.permute(0, 3, 1, 2) # shape=(B,H,W,C) -> (B,C,H,W)

        out = x + z

        if self.do_attention:
            out = self.att(out)
        
        return out

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gn = GroupNorm(dim)
        self.qkv = Linear(dim, 3*dim)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = Linear(dim, dim, gain=1e-10)

    def forward(self, x):
        B, C, H, W = x.shape

        z = self.gn(x)
        z = z.permute(0, 2, 3, 1) # shape=(B,C,H,W) -> (B,H,W,C)
        qkv = self.qkv(z).view(B, H*W, 3, C).permute(2,0,1,3) # shape=(B,H,W,3*C) -> (3,B,H*W,C)
        q,k,v = qkv[0], qkv[1], qkv[2] # (B,H*W,C)

        w = torch.matmul(q, k.transpose(-2,-1)) / (C**0.5) # shape=(B,H*W,H*W)
        attention = self.softmax(w)
        z = torch.matmul(attention, v) # shape=(B,H*W,C)

        z = self.proj(z).permute(0, 3, 1, 2).reshape(B, C, H, W)
        return x + z

class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = Conv2d(in_channel, out_channel, kernel=3, stride=2)
    
    def forward(self, x): 
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = Conv2d(in_channel, out_channel, kernel=3)
    
    def forward(self, x): 
        x = self.upsample(x)
        x = self.conv(x)
        return x

class GroupNorm(nn.Module):
    def __init__(self, in_channel, num_groups=8):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channel) # same as the TensorFlow default

    def forward(self, x):
        return self.group_norm(x)

class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride=1, gain=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding=1)
        nn.init.xavier_uniform_(self.conv.weight, gain=torch.sqrt(torch.tensor(gain))) # the original code initialization
        nn.init.constant_(self.conv.bias, 0.0) # the original code initialization
    
    def forward(self, x):
        return self.conv(x)

class Linear(nn.Module):
    def __init__(self, in_feature, out_feature, gain=1.0):
        super().__init__()
        self.linear = nn.Linear(in_feature, out_feature)
        nn.init.xavier_uniform_(self.linear.weight, gain=torch.sqrt(torch.tensor(gain))) # the original code initialization
        nn.init.constant_(self.linear.bias, 0.0) # the original code initialization

    def forward(self, x):
        return self.linear(x)

