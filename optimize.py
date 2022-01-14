import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import optuna

from models import UnetGenerator, ResnetGenerator3, ResnetGenerator6, ResnetGenerator9, Discriminator, Discriminator_1, weight_saver, log_show, train_model
from dataloader import make_data_path_list, ImageTransform, CycleGAN_Dataset, image_saver, epoch_image_saver

# Hyperparameters
num_epochs = 5
batch_size = 5
cycle_loss_rate = 0
identity_loss_rate = 0

# Build dataloaders
train_img_list_A, train_img_list_B = make_data_path_list()

train_dataset = CycleGAN_Dataset(train_img_list_A, train_img_list_B, transform=ImageTransform())
train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define generator and discriminator
gen_B = ResnetGenerator6()
gen_A = ResnetGenerator6()

dec_A = Discriminator()
dec_B = Discriminator()

# Train
def objective(trial: optuna.trial):
    suggestted_cycle_loss_rate = trial.suggest_uniform('cycle_loss_rate', 0.0, 10.0)
    suggestted_identity_loss_rate = trial.suggest_uniform('identity_loss_rate', 0.0, 10.0)
    print("suggestted_cycle_loss_rate= ", suggestted_cycle_loss_rate)
    print("suggestted_identity_loss_rate= ", suggestted_identity_loss_rate)
    gen_B_trained, gen_A_trained, dec_A_trained, dec_B_trained, logs, losses = train_model(gen_B, gen_A, dec_A, dec_B, dataloader=train_dataloader, \
    num_epochs=num_epochs, cycle_loss_rate=suggestted_cycle_loss_rate, identity_loss_rate=suggestted_identity_loss_rate, toReturnLoss=True)

    totalLoss = losses[0] + losses[1]
    return totalLoss

study = optuna.create_study()
study.optimize(objective, n_trials=20)