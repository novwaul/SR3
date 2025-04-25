import random
from PIL import Image
from os import listdir
from os.path import join

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class TrainDataset(Dataset):
    def __init__(self, img_path, lbl_path, crop_size):
        self.img_names = sorted([name for name in listdir(img_path)])
        self.lbl_names = sorted([name for name in listdir(lbl_path)])
        
        self.img_path = img_path
        self.lbl_path = lbl_path
        
        self.scale_factor = 4
        self.crop_size = crop_size
        self.tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.img_names) 

    def __getitem__(self, idx):
        img = self.tensor(Image.open(join(self.img_path, self.img_names[idx])))
        lbl = self.tensor(Image.open(join(self.lbl_path, self.lbl_names[idx])))

        # random crop
        params = transforms.RandomCrop(self.crop_size).get_params(img, (self.crop_size, self.crop_size))
        img = transforms.functional.crop(img, *params)
        lbl = transforms.functional.crop(lbl, *[self.scale_factor*p for p in params])

        # random flip
        if random.random() < 0.5:
            img = torch.flip(img, [2])
            lbl = torch.flip(lbl, [2])
        
        # random rotation
        angle = float(90 * random.randint(0, 3))
        img = transforms.functional.rotate(img, angle)
        lbl = transforms.functional.rotate(lbl, angle)

        return img, lbl
    
class EvalDataset(Dataset):
    def __init__(self, img_path, lbl_path, crop_size=None):
        self.img_names = sorted([name for name in listdir(img_path)])
        self.lbl_names = sorted([name for name in listdir(lbl_path)])
        
        self.img_path = img_path
        self.lbl_path = lbl_path
        
        self.scale_factor = 4
        self.crop_size = crop_size
        self.tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.img_names) 

    def __getitem__(self, idx):
        img = self.tensor(Image.open(join(self.img_path, self.img_names[idx])))
        lbl = self.tensor(Image.open(join(self.lbl_path, self.lbl_names[idx])))

        # crop
        if self.crop_size != None:
            img = transforms.CenterCrop(self.crop_size)(img)
            lbl = transforms.CenterCrop(self.scale_factor*self.crop_size)(lbl)

        return img, lbl

class DIV2kDataset(TrainDataset):
    def __init__(self,
                img_path,
                lbl_path,
                crop_size):
        super().__init__(
            img_path,
            lbl_path,
            crop_size
        )

class Flickr2KDataset(TrainDataset):
    def __init__(self,
                img_path,
                lbl_path,
                crop_size):
        super().__init__(
            img_path,
            lbl_path,
            crop_size
        )

class DF2KTrainDataset(Dataset):
    def __init__(self, div2k_lr_path, div2k_hr_path, flickr2k_lr_path, flickr2k_hr_path, crop_size):
        super().__init__()
        self.div2k = DIV2kDataset(div2k_lr_path, div2k_hr_path, crop_size)
        self.flickr2k = Flickr2KDataset(flickr2k_lr_path, flickr2k_hr_path, crop_size)
        self.div2k_len = len(self.div2k)
        self.flickr2k_len = len(self.flickr2k)
        self.total_imgs = self.div2k_len+self.flickr2k_len
    
    def __len__(self):
        return self.total_imgs
    
    def __getitem__(self, idx):
        if idx < self.div2k_len:
            return self.div2k.__getitem__(idx)
        else:
            return self.flickr2k.__getitem__(idx-self.div2k_len)

class DIV2KValDataset(EvalDataset):
    def __init__(self,
                img_path,
                lbl_path,
                crop_size):
        super().__init__(
            img_path,
            lbl_path,
            crop_size
        )

class Flickr2KTestDataset(EvalDataset):
    def __init__(self,
                img_path,
                lbl_path,
                crop_size):
        super().__init__(
            img_path,
            lbl_path,
            crop_size
        )
