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

from dataset import ColorizationDataset
from dataset import get_split
from transforms import (DualCompose,
                        Resize,
                        ImageOnly,
                        HorizontalFlip,
                        VerticalFlip,
                        ColorizationNormalize )
import utils


## Building Generator
img_size = 128
input_nc = 3
output_nc = 3
use_dropout = False
norm_layer = utils.get_norm_layer(norm_type='batch')
netG = models.LeakyResnetGenerator(input_nc, output_nc, ngf=32, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
## Unet Generator
utils.init_net(netG, init_type='normal', init_gain=0.02)
summary(netG, input_size=(3, img_size, img_size))

## Building Discriminator
netD = models.Discriminator(input_nc=3, img_size=img_size)
# netD = models.Discriminator(input_nc=6, img_size=img_size)
utils.init_net(netD, init_type='normal', init_gain=0.02)
summary(netD, input_size=(3, img_size, img_size))
# summary(netD, input_size=(6, img_size, img_size))


## Training
lr = 0.00002
G_optimizer = Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

root = Path('/home/xingtong/ToolTrackingData/runs/colorization_ngf_32')
# root = Path('/home/xingtong/ToolTrackingData/runs/colorization')
try:
    root.mkdir()
except OSError:
    print("directory exists!")

adding_noise = True
gaussian_std = 0.05
n_epochs = 200
report_each = 1200

train_transform = DualCompose([
    Resize(size=img_size),
    HorizontalFlip(),
    VerticalFlip(),
    ColorizationNormalize()
])

valid_transform = DualCompose([
    Resize(size=img_size),
    ColorizationNormalize()
])

fold = 0
train_file_names, val_file_names = get_split(fold=fold)

batch_size = 6
num_workers = 4
train_loader = DataLoader(dataset=ColorizationDataset(file_names=train_file_names, transform=train_transform, to_augment=True),
            shuffle=True,
            num_workers=num_workers,
            batch_size=batch_size)
dataset_length = len(train_loader)
valid_loader = DataLoader(dataset=ColorizationDataset(file_names=val_file_names, transform=valid_transform, to_augment=True),
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
    best_mean_error = state['error']
    netD.load_state_dict(state['model'])

    print('Restored model, epoch {}, step {:,}'.format(epoch, step))
else:
    epoch = 1
    step = 0
    best_mean_error = 0.0


save = lambda ep, model, model_path, error: torch.save({
    'model': model.state_dict(),
    'epoch': ep,
    'step': step,
    'error': error
}, str(model_path))

validate_each = 10
log = root.joinpath('train_{fold}.log'.format(fold=fold)).open('at', encoding='utf8')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mse_gan = nn.MSELoss()
mse_L2 = nn.MSELoss()

flag = False

best_mean_rec_loss = 100
for epoch in range(epoch, n_epochs + 1):

    if (epoch > n_epochs / 2.0 and flag == False):
        lr = lr * 0.001
        G_optimizer = Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
        D_optimizer = Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
        flag = True

    mean_D_loss = 0
    mean_G_loss = 0
    mean_recover_loss = 0
    D_losses = []
    G_losses = []
    recover_losses = []

    netG.train()
    netD.train()
    random.seed()
    tq = tqdm.tqdm(total=(len(train_loader) * batch_size))
    tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
    noise_model = models.DynamicGNoise(shape1=128, shape2=128, std=gaussian_std - (gaussian_std / n_epochs) * epoch)


    try:
        for i, (color_inputs, gray_inputs) in enumerate(train_loader):
            color_inputs, gray_inputs = color_inputs.to(device), gray_inputs.to(device)

            # ## display
            # gray = gray_inputs.data.cpu().numpy()[0]
            # color = color_inputs.data.cpu().numpy()[0]
            # gray = np.moveaxis(gray, source=[0, 1, 2], destination=[2, 0, 1])
            # color = np.moveaxis(color, source=[0, 1, 2], destination=[2, 0, 1])
            # print(np.max(gray), np.max(color))
            # cv2.imshow("gray", gray * 0.5 + 0.5)
            # cv2.imshow("color", color * 0.5 + 0.5)
            # cv2.waitKey()

            ## Updating Discriminator
            D_optimizer.zero_grad()

            pred_colors = netG(gray_inputs)
            if(adding_noise == True):
                C_real = netD(noise_model(color_inputs))
                C_fake = netD(noise_model(pred_colors.detach()))
                # C_real = netD(torch.cat((noise_model(color_inputs), gray_inputs), dim=1))
                # C_fake = netD(torch.cat((noise_model(pred_colors.detach()), gray_inputs), dim=1))
            else:
                C_real = netD(color_inputs)
                C_fake = netD(pred_colors.detach())
                # C_real = netD(torch.cat((color_inputs, gray_inputs), dim=1))
                # C_fake = netD(torch.cat((pred_colors.detach(), gray_inputs), dim=1))

            # mean_C_real = torch.mean(C_real, dim=0, keepdim=True).expand_as(C_real)
            # mean_C_fake = torch.mean(C_fake, dim=0, keepdim=True).expand_as(C_fake)
            mean_C_real = torch.mean(C_real, dim=0, keepdim=True).expand_as(C_real).detach()
            mean_C_fake = torch.mean(C_fake, dim=0, keepdim=True).expand_as(C_fake).detach()
            loss1 = mse_gan(C_real - mean_C_fake, torch.tensor(1.0).cuda().expand_as(C_real))
            loss2 = mse_gan(C_fake - mean_C_real, torch.tensor(-1.0).cuda().expand_as(C_fake))
            # loss3 = mse_L2(pred_color, color)
            loss = 0.5 * (loss1 + loss2)
            loss.backward()
            D_losses.append(loss.item())
            D_optimizer.step()

            ## Updating Generator
            G_optimizer.zero_grad()

            pred_colors = netG(gray_inputs)
            C_real = netD(color_inputs)
            C_fake = netD(pred_colors)
            # C_real = netD(torch.cat((color_inputs, gray_inputs), dim=1))
            # C_fake = netD(torch.cat((pred_colors, gray_inputs), dim=1))

            # mean_C_real = torch.mean(C_real, dim=0, keepdim=True).expand_as(C_real)
            # mean_C_fake = torch.mean(C_fake, dim=0, keepdim=True).expand_as(C_fake)
            mean_C_real = torch.mean(C_real, dim=0, keepdim=True).expand_as(C_real).detach()
            mean_C_fake = torch.mean(C_fake, dim=0, keepdim=True).expand_as(C_fake).detach()
            loss1 = mse_gan(C_fake - mean_C_real, torch.tensor(1.0).cuda().expand_as(C_fake))
            loss2 = mse_gan(C_real - mean_C_fake, torch.tensor(-1.0).cuda().expand_as(C_real))
            loss3 = mse_L2(pred_colors, color_inputs)
            loss = 0.5* (loss1 + loss2) + loss3
            loss.backward()
            G_losses.append((0.5 * (loss1 + loss2)).item())
            recover_losses.append(loss3.item())
            G_optimizer.step()

            step += 1
            tq.update(batch_size)
            mean_D_loss = np.mean(D_losses)
            mean_G_loss = np.mean(G_losses)
            mean_recover_loss = np.mean(recover_losses)
            tq.set_postfix(loss=' D={:.5f}, G={:.5f}, Rec={:.5f}'.format(mean_D_loss, mean_G_loss, mean_recover_loss))

            if (i == dataset_length - 1):
                color_imgs = []
                pred_color_imgs = []
                for i in range(batch_size):
                    color_img = color_inputs.data.cpu().numpy()[i]
                    pred_color_img = pred_colors.data.cpu().numpy()[i]

                    color_img = np.moveaxis(color_img, source=[0, 1, 2], destination=[2, 0, 1])
                    pred_color_img = np.moveaxis(pred_color_img, source=[0, 1, 2], destination=[2, 0, 1])

                    color_img = cv2.resize(color_img, dsize=(300, 300))
                    pred_color_img = cv2.resize(pred_color_img, dsize=(300, 300))
                    color_imgs.append(color_img)
                    pred_color_imgs.append(pred_color_img)

                final_color = color_imgs[0]
                final_pred_color = pred_color_imgs[0]
                for i in range(batch_size - 1):
                    final_color = cv2.hconcat((final_color, color_imgs[i + 1]))
                    final_pred_color = cv2.hconcat((final_pred_color, pred_color_imgs[i + 1]))

                final = cv2.vconcat((final_color, final_pred_color))
                final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
                cv2.imwrite(str(root / 'generated_mask_{epoch}.png'.format(epoch=epoch)),
                            np.uint8(255 * (final * 0.5 + 0.5)))
                cv2.imshow("generated", final * 0.5 + 0.5)
                cv2.waitKey(10)

        if (epoch % validate_each == 0):
            rec_losses = []
            counter = 0
            for j, (color_inputs, gray_inputs) in enumerate(valid_loader):
                netG.eval()
                color_inputs, gray_inputs = color_inputs.to(device), gray_inputs.to(device)

                pred_color_inputs = netG(gray_inputs)
                pred_color_inputs_cpu = pred_color_inputs.data.cpu().numpy()
                color_inputs_cpu = color_inputs.data.cpu().numpy()
                with torch.no_grad():
                    rec_losses.append(mse_L2(pred_color_inputs, color_inputs).item())

            mean_rec_loss = np.mean(rec_losses)

            tq.set_postfix(
                loss='validation Rec={:.5f}'.format(mean_rec_loss))
            if (mean_rec_loss < best_mean_rec_loss):
                counter = 0
                for j, (color_inputs, gray_inputs) in enumerate(valid_loader):
                    netG.eval()
                    color_inputs, gray_inputs = color_inputs.to(device), gray_inputs.to(device)
                    pred_color_inputs = netG(gray_inputs)
                    pred_color_inputs_cpu = pred_color_inputs.data.cpu().numpy()
                    color_inputs_cpu = color_inputs.data.cpu().numpy()
                    for idx in range(color_inputs_cpu.shape[0]):
                        color = color_inputs_cpu[idx]
                        pred_color = pred_color_inputs_cpu[idx]
                        color = np.moveaxis(color, source=[0, 1, 2], destination=[2, 0, 1])
                        pred_color = np.moveaxis(pred_color, source=[0, 1, 2], destination=[2, 0, 1])

                        result = cv2.cvtColor(
                            cv2.hconcat((np.uint8(255 * (color * 0.5 + 0.5)), np.uint8(255 * (pred_color * 0.5 + 0.5)))),
                            cv2.COLOR_BGR2RGB)
                        cv2.imwrite(str(root / 'validation_{counter}.png'.format(counter=counter)), result)
                        counter += 1

                ## Save both models
                best_mean_rec_loss = mean_rec_loss
                save(epoch, netD, D_model_path, best_mean_rec_loss)
                save(epoch, netG, G_model_path, best_mean_rec_loss)
                print("Finding better model in terms of validation loss: {}".format(best_mean_rec_loss))

        utils.write_event(log, step, Rec_error=mean_recover_loss)
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




