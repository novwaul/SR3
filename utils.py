import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

# Train utility functions
class Utils():

    def _gen_path(self, path):
        last_path = path + '/last.pt'
        old_path = path + '/old.pt'
        return last_path, old_path

    def exec_ema(self, train_net, sample_net, decay=0.9999):
        train_net_modules = train_net.state_dict()
        sample_net_modules = sample_net.state_dict()

        for name, train_net_module in train_net_modules.items():
            sample_net_module = sample_net_modules[name]
            old = sample_net_module.data
            new = train_net_module.data
            sample_net_modules[name].data.copy_(old*decay + new*(1-decay))

        return

    def get_states(self, path, mgpu=False, user_set_devices=None):
        last_path, _ = self._gen_path(path)
        map_location = {'cuda:%d'%user_set_devices[0]: 'cuda:%d'%user_set_devices[dist.get_rank()]} if mgpu else None
        return torch.load(last_path, map_location=map_location)
    
    def define_model(self, net, ngpus_per_node, virtual_device, device, master, mgpu, addr, port):
        if mgpu:
            torch.distributed.init_process_group(backend='gloo', store=dist.TCPStore(addr, port, ngpus_per_node, master), world_size=ngpus_per_node, rank=virtual_device)
            net = DistributedDataParallel(net.to(device), device_ids=device)
        else:
            net = net.to(device)
        return net
    
    def load_model(self, net, weight, mgpu=False):
        if mgpu:
            net.module.load_state_dict(weight, strict=False)
            dist.barrier()
        else:
            net.load_state_dict(weight, strict=False)
        return
    
    def is_divisible(self, dataset, ngpus_per_node):
        return len(dataset) % ngpus_per_node == 0