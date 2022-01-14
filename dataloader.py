import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms.transforms import RandomVerticalFlip

def make_data_path_list(offset=0):
    img_A_list = []
    img_B_list = []

    for i_th in range(300):
        img_A_path = './data/monet_jpg/img (' + str(i_th+1) + ').jpg'
        img_A_list.append(img_A_path)

        img_B_path = './data/photo_jpg/img (' + str(i_th + offset + 1) + ').jpg'
        img_B_list.append(img_B_path)

    return img_A_list, img_B_list

class ImageTransform():
    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
    
    def __call__(self, img):
        return self.data_transform(img)

class ImageTransformWithResize():
    def __init__(self, size):
        self.data_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor()
        ])
   
    def __call__(self, img):
        return self.data_transform(img)

class PictureLoader():
    def __init__(self, picture_path_list, transform=None, picture_size=256):
        self.picture_path_list = picture_path_list
        self.picture_tensor = torch.empty((len(picture_path_list), 3, picture_size, picture_size))
        self.data_transform = transform

    def __call__(self):
        for idx, picture_path in enumerate(self.picture_path_list):
            picture = Image.open(picture_path, mode='r')
            if self.data_transform is not None:
               picture = normalize(self.data_transform(picture))

            self.picture_tensor[idx] = picture

        return self.picture_tensor
        
class CycleGAN_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_A_list, img_B_list, transform):
        self.img_A_list = img_A_list
        self.img_B_list = img_B_list
        self.transform = transform

    def __len__(self):
        return len(self.img_A_list)

    def __getitem__(self, index):
        img_A_path = self.img_A_list[index]
        img_B_path = self.img_B_list[index]
        img_A = Image.open(img_A_path)
        img_B = Image.open(img_B_path)
        
        img_A_transformed = normalize(self.transform(img_A))
        img_B_transformed = normalize(self.transform(img_B))

        return img_A_transformed, img_B_transformed

def image_saver(gen_B, gen_A, img_A, img_B, batch_size, epoch):
    generated_B = gen_B(img_A)
    generated_A = gen_A(img_B)

    for idx in range(batch_size):
        save_path_A2B = './pictures/A2B/' + str(epoch*batch_size + idx) + '.jpeg'
        save_path_B2A = './pictures/B2A/' + str(epoch*batch_size + idx) + '.jpeg'

        img_B = generated_B[idx]
        img_A = generated_A[idx]
        
        img_B = img_B.cpu().detach().numpy().transpose((1, 2, 0))
        img_A = img_A.cpu().detach().numpy().transpose((1, 2, 0))
        '''
        img_B = np.clip(img_B, 0, 1) * 255
        img_A = np.clip(img_A, 0, 1) * 255
        '''
        img_B = (img_B / 2 + 0.5) * 255
        img_A = (img_A / 2 + 0.5) * 255
        
        img_B= img_B.astype(np.uint8)
        img_A= img_A.astype(np.uint8)
   
        img_B = Image.fromarray(img_B)
        img_A = Image.fromarray(img_A)

        img_B.save(fp=save_path_A2B)
        img_A.save(fp=save_path_B2A)
    
    return None

def epoch_image_saver(gen_A, img_path, epoch, device):
    img = PictureLoader([img_path], transform=ImageTransform())()
    img_nums = img.shape[0]

    generated_A = gen_A(img.to(device))

    for idx in range(img_nums):
        save_path_B2A = './pictures/epochs/' + str(epoch*img_nums + idx) + '.jpeg'

        img_A = generated_A[idx]
        
        img_A = img_A.cpu().detach().numpy().transpose((2, 1, 0))

        img_A = (img_A / 2 + 0.5) * 255
        
        img_A= img_A.astype(np.uint8)
        img_A = Image.fromarray(img_A)

        img_A.save(fp=save_path_B2A)
    
    return None

def unnormalize(x):
    return x / 2 + 0.5

def normalize(x):
    return 2 * x + - 1