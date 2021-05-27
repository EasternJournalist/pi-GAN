import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision
from pathlib import Path

class ImageDataset(Dataset):
    def __init__(self, folder=None, image_size=64, transparent:bool=False, aug_prob:bool=0., exts = ['jpg', 'jpeg', 'png']):
        super(ImageDataset, self).__init__()
        self.image_size = image_size
        
        if os.path.exists('/data/52WangRuicheng/FFHQ64pkl/all_imgs.pkl'):
            print('Load pickle')
            with open('/data/52WangRuicheng/FFHQ64pkl/all_imgs.pkl', 'rb') as f:
                self.all_imgs = pickle.load(f)
        else:
            print('Get Paths...')
            self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
            assert len(self.paths) > 0, f'No images were found in {folder} for training'
    
            self.all_imgs = []
            print('Read images...')
            for i, path in enumerate(self.paths):
                self.all_imgs.append(Image.open(path))
                print(f'[{i + 1:>5d}/{len(self.paths):>5d}]', end='\r')
            print()
            print('Save pickle...')
            with open('/data/52WangRuicheng/FFHQ64pkl/all_imgs.pkl', 'wb') as f:
                pickle.dump(self.all_imgs, f, 1)
        self.img_arr = None
        
        self.create_transform(image_size)
        
    def create_transform(self, image_size):
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
        self.img_arr = []
        print('Transform')
        for i, img in enumerate(self.all_imgs):
            self.img_arr.append(self.transform(img))
            print(f'[{i + 1:>5d}/{len(self.all_imgs):>5d}]', end='\r')
        self.img_arr = torch.stack(self.img_arr)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, index):
        return self.img_arr[index]

dataset = ImageDataset('/data/52WangRuicheng/FFHQ64/image64_rescale', 16)