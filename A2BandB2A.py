import torch
import torch.nn as nn
import torch.utils.data as data

import random
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt 
from models import UnetGenerater, ResnetGenerator, Discriminater, Discriminater_1
from dataloader import make_data_path_list, ImageTransformWithResize, CycleGAN_Dataset, ImageTransform, image_saver

def unnormalize(x):
    return x / 2 + 0.5

def translate(weight_time_info, offset, gen_B, gen_A):
    batch_size = 5

    img_list_A, img_list_B = make_data_path_list(offset=offset)

    test_dataset = CycleGAN_Dataset(img_list_A, img_list_B, transform=ImageTransform())
    test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # load parameters
    A2B_load_path = './weight/A2B/trained_weight_A2B'+str(weight_time_info)+'.pth'
    B2A_load_path = './weight/B2A/trained_weight_B2A'+str(weight_time_info)+'.pth'

    A2B_load_weights = torch.load(A2B_load_path, map_location={'cuda:0': 'cpu'})
    B2A_load_weights = torch.load(B2A_load_path, map_location={'cuda:0': 'cpu'})

    gen_B.load_state_dict(A2B_load_weights)
    gen_A.load_state_dict(B2A_load_weights)

    gen_B.eval()
    gen_A.eval()

    # forward
    batch_iterator = iter(test_dataloader)

    max_int = random.randint(0, 60)

    for i_th in range(max_int):
        imges = next(batch_iterator)

    img_A = imges[0]
    img_B = imges[1]

    fake_A = gen_A(img_B)
    fake_B = gen_B(img_A)

    img_A_0 = img_A[0].detach().numpy().transpose((2, 1, 0))
    plt.imshow(img_A_0)
    plt.show()

    # Show A2B
    fig = plt.figure(figsize=(15, 6))
    for i_th in range(batch_size):
        img_transposed = img_A[i_th].detach().numpy().transpose((1, 2, 0))
        plt.subplot(2, 5, i_th+1)
        plt.title('Monet Original')
        plt.imshow(unnormalize(img_transposed))

        img_transposed = fake_B[i_th].detach().numpy().transpose((1, 2, 0))
        plt.subplot(2, 5, 5+i_th+1)
        plt.title('CycleGAN')
        plt.imshow(unnormalize(img_transposed))

    plt.show()

    # Show B2A
    fig = plt.figure(figsize=(15, 6))
    for i_th in range(batch_size):
        
        img_transposed = img_B[i_th].detach().numpy().transpose((1, 2, 0))
        plt.subplot(2, 5, i_th+1)
        plt.title('Picture 4')
        plt.imshow(unnormalize(img_transposed))

        img_transposed = fake_A[i_th].detach().numpy().transpose((1, 2, 0))
        plt.subplot(2, 5, 5+i_th+1)
        plt.title('CycleGAN')
        plt.imshow(unnormalize(img_transposed))

    plt.show()

    image_saver(gen_B=gen_B, gen_A=gen_A, img_A=img_A, img_B=img_B, batch_size=5, epoch=0)

    return None


gen_B = ResnetGenerator()
gen_A = ResnetGenerator()

translate(weight_time_info='_2021-3-18_22-20-25', offset=300, gen_B=gen_B, gen_A=gen_A)

