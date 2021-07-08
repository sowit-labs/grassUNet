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
from utils.dataset import PAIDataset
from utils.transforms import Transformer
from torch.utils.data import DataLoader, random_split

dir_train = 'train/'
dir_val = 'val/'
dir_test = 'test/'
dir_test_hum = 'test-hum/'

subdir_img = 'img/'
subdir_mask = 'mask-veg/'
dir_checkpoint = './checkpoint/'


def train_net(net,
              device,
              datadir='./data/',
              epochs=5,
              batch_size=1,
              lr=0.001,
              save_cp=True,
              img_scale=0.5,
              humantest=True):
    """Function to train the U-Net segmentation model from a dataset"""

    transforms = None
        
    dataset_train = PAIDataset(os.path.join(datadir,dir_train,subdir_img), os.path.join(datadir,dir_train,subdir_mask), img_scale, transform=transforms)
    dataset_val = PAIDataset(os.path.join(datadir,dir_val,subdir_img), os.path.join(datadir,dir_val,subdir_mask), img_scale, transform=None)
    dataset_test = PAIDataset(os.path.join(datadir,dir_test,subdir_img), os.path.join(datadir,dir_test,subdir_mask), img_scale, transform=None)
    if humantest:
        dataset_test_hum = PAIDataset(os.path.join(datadir,dir_test_hum,subdir_img), os.path.join(datadir,dir_test_hum,subdir_mask), img_scale, transform=None)
    
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    if humantest:
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
        Human test size: {len(dataset_test_hum) if humantest else 0}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
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
                    val_score = eval_net(net, val_loader, device)
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
        
        if humantest:
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
    """Arguments for the CLI"""
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', metavar='D', type=str, default="data",
                        help='Dataset directory path', dest='datadir')
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
    parser.add_argument('-t', '--human-test', action='store_true', default=False,
                        help='Test with humanly-annotated data if True')

    return parser.parse_args()


if __name__ == '__main__':
    """Process call from the CLI"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

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
                  datadir=args.datadir,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  humantest=args.human_test)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)