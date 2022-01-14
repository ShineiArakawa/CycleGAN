import os
from glob import glob
import numpy as np
from PIL import Image
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from dataloader import PictureLoader, ImageTransform, unnormalize
from models import ResnetGenerator6, ResnetGenerator9

def monet(weight_time_info, gen_A):
    picture_path_list = glob('./pictures/*.jpeg')
    picture_tensors = PictureLoader(picture_path_list=picture_path_list, transform=ImageTransform())()


    # load parameters
    B2A_load_path = './weight/B2A/trained_weight_B2A_'+str(weight_time_info)+'.pth'

    B2A_load_weights = torch.load(B2A_load_path, map_location={'cuda:0': 'cpu'})
    gen_A.load_state_dict(B2A_load_weights)

    gen_A.eval()

    # forward
    generated_pictures = gen_A(picture_tensors)

    max_epoch = picture_tensors.size(0) // 5
    residue = picture_tensors.size(0) % 5

    for epoch in range(max_epoch):
        for i_th in range(5):
            idx = 5*epoch + i_th

            picture = picture_tensors[idx].detach().numpy().transpose((1, 2, 0))
            plt.subplot(2, 5, i_th+1)
            plt.title('Original')
            plt.imshow(unnormalize(picture))
            
            generated_picture = generated_pictures[idx].cpu().detach().numpy().transpose((1, 2, 0))

            picture_for_save = generated_picture

            plt.subplot(2, 5, 5+i_th+1)
            plt.title('CycleGAN')
            plt.imshow(unnormalize(generated_picture))  
            
            save_path_B2A = './pictures/generated_' + str(idx) + '.jpeg'

            generated_picture = (picture_for_save / 2 + 0.5) * 255
            generated_picture = unnormalize(generated_picture).astype(np.uint8)
            generated_picture = Image.fromarray(generated_picture)

            generated_picture.save(fp=save_path_B2A)
        
        plt.show()
    

    for i_th in range(residue):
        idx = 5*max_epoch + i_th

        picture = picture_tensors[idx].detach().numpy().transpose((1, 2, 0))
        plt.subplot(2, 5, i_th+1)
        plt.title('Original')
        plt.imshow(unnormalize(picture))
        
        generated_picture = generated_pictures[idx].detach().numpy().transpose((1, 2, 0))
        plt.subplot(2, 5, 5+i_th+1)
        plt.title('CycleGAN')
        plt.imshow(unnormalize(generated_picture))
        
        save_path_B2A = './pictures/generated_' + str(idx) + '.jpeg'

        generated_picture = (generated_picture / 2 + 0.5) * 255
        generated_picture = generated_picture.astype(np.uint8)
        generated_picture = Image.fromarray(generated_picture)

        generated_picture.save(fp=save_path_B2A)
        
    plt.show()

    
    return None


gen = ResnetGenerator6()
monet(weight_time_info='2021-6-30_22-24-12', gen_A=gen)