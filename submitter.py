import os
from tqdm import tqdm
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

def submitter(weight_time_info, gen_A, num_images=7000, batch_size=500):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # load parameters
    B2A_load_path = './weight/B2A/trained_weight_B2A_'+str(weight_time_info)+'.pth'

    B2A_load_weights = torch.load(B2A_load_path, map_location={'cuda:0': 'cpu'})
    gen_A.load_state_dict(B2A_load_weights)

    gen_A.eval()
    gen_A.to(device)

    epochs = num_images // batch_size
    if num_images % batch_size != 0:
        epochs += 1

    for i_th in tqdm(range(epochs)):
        picture_path_list = []

        for j_th in range(1, batch_size+1):   
            picture_path = './data/photo_jpg/img ('+str(batch_size*i_th+j_th)+').jpg'
            picture_path_list.append(picture_path)

        picture_tensors = PictureLoader(picture_path_list=picture_path_list, transform=ImageTransform())().to(device)

        # forward
        generated_pictures = gen_A(picture_tensors)

        # Transform and save

        for j_th in range(1, batch_size+1):
            save_path_B2A = './pictures/images/generated_' + str(batch_size*i_th+j_th) + '.jpg'

            generated_picture = generated_pictures[j_th-1].cpu().detach().numpy().transpose((1, 2, 0))
            
            generated_picture = unnormalize(generated_picture) * 255
            generated_picture = generated_picture.astype(np.uint8)
                
            
            generated_picture = Image.fromarray(generated_picture)
            generated_picture.save(fp=save_path_B2A)
        
    return None


gen = ResnetGenerator6()
submitter(weight_time_info='2021-4-14_18-4-5', gen_A=gen, batch_size=10)