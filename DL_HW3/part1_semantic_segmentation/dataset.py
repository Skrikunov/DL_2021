from torch.utils.data import Dataset, DataLoader
import albumentations as A
from torchvision.transforms import ToTensor
import os
from PIL import Image
import numpy as np
import torch



class FloodNet(Dataset):
    """
    Labels semantic:
    0: Background, 1: Building, 2: Road, 3: Water, 4: Tree, 5: Vehicle, 6: Pool, 7: Grass
    """
    def __init__(
        self,
        data_path: str,
        phase: str,
        augment: bool,
        img_size: int,
    ):
        self.num_classes = 8
        self.data_path = data_path
        self.phase = phase
        self.augment = augment
        self.img_size = img_size

        self.items = [filename.split('.')[0] for filename in os.listdir(f'{data_path}/{phase}/image')]
        
        # TODO: implement augmentations (3.5 points)
        if augment:
            # TODO:
            # Random resize
            # Random crop (within image borders, output size = img_size)
            # Random rotation
            # Random horizontal and vertical Flip
            # Random color augmentation
            # self.transform = None
            self.transform = A.Compose([
                                      # Random resize
                                      A.Resize(256, 256),
                                      # Random crop
                                      A.RandomCrop(width=256, height=256),
                                      # Random rotation
                                      A.RandomRotate90(),
                                      A.ShiftScaleRotate(shift_limit=0.2,
                                                        scale_limit=0.2,
                                                        rotate_limit=30,
                                                        p=0.5),
                                      # Random horizontal and vertical Flip
                                      A.HorizontalFlip(p=0.5),
                                      A.VerticalFlip(p=0.5),
                                      # Random color augmentation
                                      A.RandomBrightnessContrast(brightness_limit=0.3,
                                                                contrast_limit=0.3,
                                                                p=0.5),
                                      A.RandomGamma(p=0.5),    
                                      A.CLAHE(p=0.5)
                                      ],p=1.0)
        else:
            # TODO: random crop to img_size
            # self.transform = None
            self.transform = A.Compose([
                                      A.RandomCrop(width=256, height=256)
                                      ])
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image = np.asarray(Image.open(f'{self.data_path}/{self.phase}/image/{self.items[index]}.jpg'))
        mask = np.asarray(Image.open(f'{self.data_path}/{self.phase}/mask/{self.items[index]}.png'))
        
        if self.phase == 'train':
            # TODO: apply transform to both image and mask (0.5 points)
            transformed = self.transform(image=image,mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            pass
        
        image = self.to_tensor(image.copy())
        mask = torch.from_numpy(mask.copy()).long()

        if self.phase == 'train':
            assert isinstance(image, torch.FloatTensor) and image.shape == (3, self.img_size, self.img_size)
            assert isinstance(mask, torch.LongTensor) and mask.shape == (self.img_size, self.img_size)

        return image, mask