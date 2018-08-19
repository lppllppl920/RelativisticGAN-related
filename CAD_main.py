import dataset
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
                        MaskOnly,
                        MaskShiftScaleRotate,
                        MaskShift,
                        Normalize,
                        HorizontalFlip,
                        VerticalFlip)
import utils


mask_file_names = dataset.get_file_names('/home/xingtong/CAD_models/knife1', 'knife_mask')
print(mask_file_names)

color_file_names = dataset.get_file_names('/home/xingtong/CAD_models/knife1', 'color_')
print(color_file_names)

img_size = 128
train_transform = DualCompose([
    Resize(size=img_size),
    HorizontalFlip(),
    VerticalFlip(),
    Normalize(normalize_mask=True),
    MaskOnly([MaskShiftScaleRotate(scale_upper=4.0), MaskShift(limit=50)])
])

num_workers = 4
batch_size = 6
train_loader = DataLoader(dataset=dataset.CADDataset(color_file_names=color_file_names, mask_file_names=mask_file_names, transform=train_transform),
           shuffle=True,
           num_workers=num_workers,
           batch_size=batch_size)

root = Path('/home/xingtong/ToolTrackingData/runs/debug_CAD_knife')
## Building Generator
img_size = 128
input_nc = 3
output_nc = 1
import nets
netG = nets.UNet11(num_classes=1, pretrained='vgg')

# # ngf = 32
# use_dropout = False
# norm_layer = utils.get_norm_layer(norm_type='batch')
# # netG = models.ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
# netG = models.LeakyResnetGenerator(input_nc, output_nc, ngf=6, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
# # netG = models.LeakyResnetGenerator(input_nc, output_nc, ngf=64, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
# ## Unet Generator
# # netG = models.UnetGenerator(input_nc, output_nc, 7, ngf=3, norm_layer=norm_layer, use_dropout=use_dropout)
# utils.init_net(netG, init_type='normal', init_gain=0.02)
summary(netG, input_size=(3, img_size, img_size))

## Building Discriminator
netD = models.Discriminator(input_nc=1, img_size=img_size)
utils.init_net(netD, init_type='normal', init_gain=0.02)
summary(netD, input_size=(1, img_size, img_size))

lr = 0.00002
G_optimizer = Adam(netG.parameters(), lr=lr, betas=(0.9, 0.999))
D_optimizer = Adam(netD.parameters(), lr=lr, betas=(0.9, 0.999))

G_model_path = root / 'G_model.pt'
D_model_path = root / 'D_model.pt'
if G_model_path.exists() and D_model_path.exists():
    state = torch.load(str(G_model_path))
    netG.load_state_dict(state['model'])

    state = torch.load(str(D_model_path))
    epoch = state['epoch']
    step = state['step']
    netD.load_state_dict(state['model'])

    print('Restored model, epoch {}, step {:,}'.format(epoch, step))
else:
    epoch = 1
    step = 0

save = lambda ep, model, model_path: torch.save({
    'model': model.state_dict(),
    'epoch': ep,
    'step': step
}, str(model_path))


adding_noise = False
gaussian_std = 0.1
n_epochs = 500
report_each = 10

validate_each = 1
log = root.joinpath('train.log').open('at', encoding='utf8')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mse = nn.MSELoss()

for epoch in range(epoch, n_epochs + 1):

    mean_D_loss = 0
    mean_G_loss = 0
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

            color = inputs.data.cpu().numpy()[0]
            mask = real_masks.data.cpu().numpy()[0]
            color = np.moveaxis(color, source=[0, 1, 2], destination=[2, 0, 1])
            mask = np.moveaxis(mask, source=[0, 1, 2], destination=[2, 0, 1])
            # cv2.imshow("rgb", color * 0.5 + 0.5)
            # cv2.imshow("mask", mask * 0.5 + 0.5)
            # cv2.waitKey(1)

            for _ in range(5):
                ## Updating Discriminator
                D_optimizer.zero_grad()

                if (adding_noise == True):
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

            if (i == 250 / batch_size):
                colors = []
                masks = []
                inputs_array = inputs.data.cpu().numpy()
                fake_masks_array = fake_masks.data.cpu().numpy()
                for i in range(inputs_array.shape[0]):
                    color = inputs_array[i]
                    mask = fake_masks_array[i]
                    color = np.moveaxis(color, source=[0, 1, 2], destination=[2, 0, 1])
                    mask = np.moveaxis(mask, source=[0, 1, 2], destination=[2, 0, 1])
                    color = cv2.resize(color, dsize=(300, 300))
                    mask = cv2.resize(mask, dsize=(300, 300))
                    colors.append(color)
                    masks.append(mask)

                final_color = colors[0]
                final_mask = masks[0]
                for i in range(inputs_array.shape[0] - 1):
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

        utils.write_event(log, step, Dloss=mean_D_loss)
        utils.write_event(log, step, Gloss=mean_G_loss)
        tq.close()

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        tq.close()
        print('Ctrl+C, saving snapshot')
        save(epoch, netD, D_model_path)
        save(epoch, netG, G_model_path)
        print('done.')
        exit()