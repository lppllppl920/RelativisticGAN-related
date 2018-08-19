import models

import random
import os
import cv2
import numpy as np
import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD

from pathlib import Path
from torchsummary import summary

from dataset import RoboticsDataset
from dataset import get_split
from transforms import (DualCompose,
                        Resize,
                        ImageOnly,
                        Normalize,
                        HorizontalFlip,
                        VerticalFlip,
                        MaskOnly,
                        MaskErodeDilation,
                        MaskShiftScaleRotate)
import utils

## Put only one artificial object onto some nturl stuff to generte dtset

## Building Generator
img_size = 128
input_nc = 3
output_nc = 1
# ngf = 32
use_dropout = False
norm_layer = utils.get_norm_layer(norm_type='batch')
# netG = models.ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
# netG = models.LeakyResnetGenerator(input_nc, output_nc, ngf=6, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
netG = models.LeakyResnetGenerator(input_nc, output_nc, ngf=6, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
## Unet Generator
# netG = models.UnetGenerator(input_nc, output_nc, 7, ngf=32, norm_layer=norm_layer, use_dropout=use_dropout)
utils.init_net(netG, init_type='normal', init_gain=0.02)
summary(netG, input_size=(3, img_size, img_size))

## Building Discriminator
netD = models.Discriminator(input_nc=1, img_size=img_size)
utils.init_net(netD, init_type='normal', init_gain=0.02)
summary(netD, input_size=(1, img_size, img_size))

lr = 0.00002
G_optimizer = Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

# root = Path('/home/xingtong/ToolTrackingData/runs/debug_Leaky_Gaussian_input_fold_0')
# root = Path('/home/xingtong/ToolTrackingData/runs/debug_Leaky_Gaussian_input_fold_4') ## This validation set has no confusing non-target instruments
root = Path('/home/xingtong/ToolTrackingData/runs/debug_scale_1_0_1_5_ngf_6')
try:
    root.mkdir()
except OSError:
    print("directory exists!")
# root = Path('/home/xingtong/ToolTrackingData/runs/debug_Unet_fold_4')

adding_noise = True
gaussian_std = 0.05
n_epochs = 500
report_each = 1200

## Configuring data loader
# fold = 0
fold = 4
train_file_names, val_file_names = get_split(fold=fold)
# color image in range [0, 255] will become [-1, 1]
train_transform = DualCompose([
    Resize(size=img_size),
    HorizontalFlip(),
    VerticalFlip(),
    Normalize(normalize_mask=False),
    MaskOnly([MaskErodeDilation(kernel_size_lower=0, kernel_size_upper=6)])
])

valid_transform = DualCompose([
    Resize(size=img_size),
    Normalize(normalize_mask=False)
])

batch_size = 6
num_workers = 4
train_loader = DataLoader(dataset=RoboticsDataset(mode='train', file_names=train_file_names, transform=train_transform, problem_type='binary'),
            shuffle=True,
            num_workers=num_workers,
            batch_size=batch_size)
dataset_length = len(train_loader)
validation_loader = DataLoader(dataset=RoboticsDataset(mode='valid', file_names=val_file_names, transform=valid_transform, problem_type='binary'),
            shuffle=False,
            num_workers=num_workers,
            batch_size=batch_size)

## Training
G_model_path = root / 'G_model_{fold}.pt'.format(fold=fold)
D_model_path = root / 'D_model_{fold}.pt'.format(fold=fold)
if G_model_path.exists() and D_model_path.exists():
    state = torch.load(str(G_model_path))
    netG.load_state_dict(state['model'])

    state = torch.load(str(D_model_path))
    epoch = state['epoch']
    step = state['step']
    best_mean_dice_coeffs = state['dice']
    netD.load_state_dict(state['model'])

    print('Restored model, epoch {}, step {:,}'.format(epoch, step))
else:
    epoch = 1
    step = 0
    best_mean_dice_coeffs = 0.0

save = lambda ep, model, model_path, dice: torch.save({
    'model': model.state_dict(),
    'epoch': ep,
    'step': step,
    'dice': dice
}, str(model_path))


validate_each = 10
log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mse = nn.MSELoss()

## TODO: check overfitting
## high resolution generator
## validation
## try it on arbitrary object
## journal
## TMI
## ALWAYS prepare slides

# best_mean_dice_coeffs = 0.0
flag = False
for epoch in range(epoch, n_epochs + 1):

    if(epoch > n_epochs / 2.0 and flag == False):
        lr = lr * 0.001
        G_optimizer = Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
        D_optimizer = Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
        flag = True
        
    mean_D_loss = 0
    mean_G_loss = 0
    mean_dice_coeffs = 0
    D_losses = []
    G_losses = []
    netG.train()
    netD.train()
    random.seed()
    tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
    tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
    noise_model = models.DynamicGNoise(shape1=128, shape2=128, std=gaussian_std - (gaussian_std / n_epochs) * epoch)
    try:
        for i, (inputs, real_masks) in enumerate(train_loader):
            inputs, real_masks = inputs.to(device), real_masks.to(device)

            # # ## display
            # color = inputs.data.cpu().numpy()[0]
            # mask = real_masks.data.cpu().numpy()[0]
            # color = np.moveaxis(color, source=[0, 1, 2], destination=[2, 0, 1])
            # mask = np.moveaxis(mask, source=[0, 1, 2], destination=[2, 0, 1])
            # print(np.min(mask), np.max(mask))
            # cv2.imshow("rgb", color * 0.5 + 0.5)
            # cv2.imshow("mask", mask * 0.5 + 0.5)
            # cv2.waitKey()

            ## Updating Discriminator
            D_optimizer.zero_grad()

            if(adding_noise == True):
                C_real = netD(noise_model(real_masks))
                C_fake = netD(noise_model(netG(inputs).detach()))
            else:
                C_real = netD(real_masks)
                C_fake = netD(netG(inputs).detach())

            # mean_C_real = torch.mean(C_real, dim=0, keepdim=True).expand_as(C_real)
            # mean_C_fake = torch.mean(C_fake, dim=0, keepdim=True).expand_as(C_fake)
            mean_C_real = torch.mean(C_real, dim=0, keepdim=True).expand_as(C_real).detach()
            mean_C_fake = torch.mean(C_fake, dim=0, keepdim=True).expand_as(C_fake).detach()
            loss1 = mse(C_real - mean_C_fake, torch.tensor(1.0).cuda().expand_as(C_real))
            loss2 = mse(C_fake - mean_C_real, torch.tensor(-1.0).cuda().expand_as(C_fake))
            loss = loss1 + loss2
            loss.backward()
            D_losses.append(loss.item())
            D_optimizer.step()


            ## Updating Generator
            G_optimizer.zero_grad()
            C_real = netD(real_masks)
            fake_masks = netG(inputs)
            C_fake = netD(fake_masks)
            # mean_C_real = torch.mean(C_real, dim=0, keepdim=True).expand_as(C_real)
            # mean_C_fake = torch.mean(C_fake, dim=0, keepdim=True).expand_as(C_fake)
            mean_C_real = torch.mean(C_real, dim=0, keepdim=True).expand_as(C_real).detach()
            mean_C_fake = torch.mean(C_fake, dim=0, keepdim=True).expand_as(C_fake).detach()
            loss1 = mse(C_fake - mean_C_real, torch.tensor(1.0).cuda().expand_as(C_fake))
            loss2 = mse(C_real - mean_C_fake, torch.tensor(-1.0).cuda().expand_as(C_real))
            loss = loss1 + loss2
            loss.backward()
            G_losses.append(loss.item())
            G_optimizer.step()

            step += 1
            tq.update(batch_size)
            mean_D_loss = np.mean(D_losses[-report_each:])
            mean_G_loss = np.mean(G_losses[-report_each:])
            tq.set_postfix(loss=' D={:.5f}, G={:.5f}'.format(mean_D_loss, mean_G_loss))
            # if i and i % report_each == 0:
            #     utils.write_event(log, step, Dloss=mean_D_loss)
            #     utils.write_event(log, step, Gloss=mean_G_loss)

            if (i == dataset_length - 1):
                colors = []
                masks = []
                for i in range(batch_size):
                    color = inputs.data.cpu().numpy()[i]
                    mask = fake_masks.data.cpu().numpy()[i]
                    color = np.moveaxis(color, source=[0, 1, 2], destination=[2, 0, 1])
                    mask = np.moveaxis(mask, source=[0, 1, 2], destination=[2, 0, 1])
                    color = cv2.resize(color, dsize=(300, 300))
                    mask = cv2.resize(mask, dsize=(300, 300))
                    colors.append(color)
                    masks.append(mask)

                final_color = colors[0]
                final_mask = masks[0]
                for i in range(batch_size - 1):
                    final_color = cv2.hconcat((final_color, colors[i + 1]))
                    final_mask = cv2.hconcat((final_mask, masks[i + 1]))
                final_mask = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
                final = cv2.vconcat((final_color, final_mask))
                # cv2.imwrite(str(root / 'color_images.png'), np.uint8(255*(final_color * 0.5 + 0.5)))
                # cv2.imshow("rgb", final * 0.5 + 0.5)
                # cv2.imshow("mask", final_mask * 0.5 + 0.5)
                final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
                cv2.imwrite(str(root / 'generated_mask_{epoch}.png'.format(epoch=epoch)),
                            np.uint8(255 * (final * 0.5 + 0.5)))
                cv2.imshow("generated", final * 0.5 + 0.5)
                cv2.waitKey(10)

        if(epoch % validate_each == 0):
            dice_coeffs = []
            counter = 0
            for j, (inputs, gt_masks) in enumerate(validation_loader):
                netG.eval()
                inputs, gt_masks = inputs.to(device), gt_masks.to(device)
                pred_masks = netG(inputs)
                pred_masks_cpu = pred_masks.data.cpu().numpy()
                inputs_cpu = inputs.data.cpu().numpy()
                dice_coeff = utils.dice_coefficient(((pred_masks * 0.5 + 0.5) > 0.5).float(),
                                                    (gt_masks * 0.5 + 0.5).float())
                dice_coeffs.append(dice_coeff.item())
            mean_dice_coeffs = np.mean(np.array(dice_coeffs))

            tq.set_postfix(loss=' D={:.5f}, G={:.5f}, validation dice score={:.5f}'.format(np.mean(D_losses), np.mean(G_losses), mean_dice_coeffs))
            if(mean_dice_coeffs > best_mean_dice_coeffs):
                counter = 0
                for j, (inputs, gt_masks) in enumerate(validation_loader):
                    netG.eval()
                    inputs, gt_masks = inputs.to(device), gt_masks.to(device)
                    pred_masks = netG(inputs)
                    pred_masks_cpu = pred_masks.data.cpu().numpy()
                    inputs_cpu = inputs.data.cpu().numpy()
                    for idx in range(inputs_cpu.shape[0]):
                        color = inputs_cpu[idx]
                        mask = pred_masks_cpu[idx]
                        color = np.moveaxis(color, source=[0, 1, 2], destination=[2, 0, 1])
                        mask = np.moveaxis(mask, source=[0, 1, 2], destination=[2, 0, 1])

                        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        result = cv2.cvtColor(
                            cv2.hconcat((np.uint8(255 * (color * 0.5 + 0.5)), np.uint8(255 * (mask * 0.5 + 0.5)))),
                            cv2.COLOR_BGR2RGB)
                        cv2.imwrite(str(root / 'validation_{counter}.png'.format(counter=counter)), result)
                        counter += 1

                ## Save both models
                best_mean_dice_coeffs = mean_dice_coeffs
                save(epoch, netD, D_model_path, best_mean_dice_coeffs)
                save(epoch, netG, G_model_path, best_mean_dice_coeffs)
                print("Finding better model in terms of validation loss: {}".format(best_mean_dice_coeffs))

        else:
            tq.set_postfix(loss=' D={:.5f}, G={:.5f}'.format(np.mean(D_losses), np.mean(G_losses)))

        utils.write_event(log, step, Dice_coeff=mean_dice_coeffs)
        utils.write_event(log, step, Dloss=mean_D_loss)
        utils.write_event(log, step, Gloss=mean_G_loss)
        tq.close()



    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        tq.close()
        print('Ctrl+C, saving snapshot')
        # save(epoch, netD, D_model_path)
        # save(epoch, netG, G_model_path)
        print('done.')
        exit()