import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import LAIDataset
from utils.transforms import Transformer
from torch.utils.data import DataLoader, random_split

dir_train = '../data-train-val-test-520-390/train_aug_bcgsFRSo/'
dir_val = '../data-train-val-test-520-390/val/'
dir_test = '../data-train-val-test-520-390/test/'
dir_test_hum = '../data-train-val-test-520-390/test-hum/'

subdir_img = 'img/'
subdir_mask = 'mask-veg/'
dir_checkpoint = './checkpoint/'


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              save_cp=True,
              img_scale=0.5,
              data_aug="",
              threshold_eval=0.5):
    if type(data_aug) is str:
        brange=(1,1)
        crange=(1,1)
        grange=(1,1)
        srange=(1,1)
        hbool=False
        crop=False
        rotate=0
        shear=0
        
        if 'b' in data_aug:
            brange=(0.75,1.5)
        if 'c' in data_aug:
            crange=(0.75,1.5)
        if 'g' in data_aug:
            grange=(0.75,2)
        if 's' in data_aug:
            srange=(0.75,2)
        if 'F' in data_aug:
            hbool=True
        if 'C' in data_aug:
            crop=True
        if 'R' in data_aug:
            rotate=20
        if 'S' in data_aug:
            shear=15
        transforms = Transformer(brightness_range=brange, contrast_range=crange, gamma_range=grange, saturation_range=srange, hflip=hbool, crop=crop, rotate=rotate, shear=shear)
    else:
        transforms = None
        
    dataset_train = LAIDataset(dir_train+subdir_img, dir_train+subdir_mask, img_scale, transform=transforms)
    dataset_val = LAIDataset(dir_val+subdir_img, dir_val+subdir_mask, img_scale, transform=None)
    dataset_test = LAIDataset(dir_test+subdir_img, dir_test+subdir_mask, img_scale, transform=None)
    dataset_test_hum = LAIDataset(dir_test_hum+subdir_img, dir_test_hum+subdir_mask, img_scale, transform=None)
    
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_hum_loader = DataLoader(dataset_test_hum, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}_AUG_{data_aug}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(dataset_train)}
        Validation size: {len(dataset_val)}
        Test size:       {len(dataset_test)}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Data augmentation: {data_aug}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=len(dataset_train), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (len(dataset_train) // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device, threshold_eval)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/val', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/val', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
                        
        test_score = eval_net(net, test_loader, device, threshold_eval)
        if net.n_classes > 1:
            logging.info('Test cross entropy: {}'.format(test_score))
            writer.add_scalar('Loss/test', test_score, global_step)
        else:
            logging.info('Test Dice Coeff: {}'.format(test_score))
            writer.add_scalar('Dice/test', test_score, global_step)
            
        test_hum_score = eval_net(net, test_hum_loader, device, threshold_eval)
        if net.n_classes > 1:
            logging.info('Test human cross entropy: {}'.format(test_hum_score))
            writer.add_scalar('Loss/testHuman', test_hum_score, global_step)
        else:
            logging.info('Test human Dice Coeff: {}'.format(test_hum_score))
            writer.add_scalar('Dice/testHuman', test_hum_score, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-a', '--data-augmentation', dest='dataaugmentation', type=str, default="",
                        help='Use image augmentation on the original dataset. "bcgsF" for all the transformations (b brightness, c contrast, g gamma, s saturation, F horizontal flip)')
    parser.add_argument('-t', '--threshold-eval', dest='thresholdeval', type=float, default=0.5,
                        help='Threshold applied to the confident score of the predicted mask in order to evaluate the Dice coefficient')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  data_aug=args.dataaugmentation,
                  threshold_eval=args.thresholdeval)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)