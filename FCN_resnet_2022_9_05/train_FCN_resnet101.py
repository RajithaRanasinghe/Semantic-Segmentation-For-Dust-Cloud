import argparse
import os
import time
import imageio
import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from datetime import datetime
from distutils.dir_util import copy_tree
from data import get_loader
from model.Unet import Unet
from utils import get_logger, create_dir
import torchvision.transforms as T
import torchvision.models as M
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=5e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
parser.add_argument('--dataset', type=str, default='dust', help='dataset name')
opt = parser.parse_args()

CE = torch.nn.BCELoss()


def dice_loss(pred_mask, true_mask):
    loss = 1 - dice(pred_mask, true_mask)

    return loss


def calc_loss(pred, target, bce_weight=0.2):
    bce = CE(pred, target)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss


def dice(pred, target):
    intersection = (abs(target - pred) < 0.05).sum()
    cardinality = (target >= 0).sum() + (pred >= 0).sum()

    return 2.0 * intersection / cardinality


class Network(object):
    def __init__(self):
        self.save_best = False
        self.best_mIoU, self.best_dice_coeff = 0, 0
        self.recall_array = [0]
        self.fallout_array = [0]
        self._init_configure()
        self._init_logger()

    def _init_configure(self):
        with open('configs/config.yml') as fp:
            self.cfg = yaml.safe_load(fp)

    def _init_logger(self):

        self.model_name = 'FCN/FCN_resnet101'
        #self.model_name = 'FCN/fcn_resnet50'
        

        log_dir = 'logs/' + self.model_name + '/' + opt.dataset + '/train' + '/{}'.format(
            time.strftime('%Y%m%d-%H%M%S'))

        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))

        self.save_path = log_dir
        self.image_save_path = log_dir + "/saved_images"
        create_dir(self.image_save_path)
        self.save_tbx_log = self.save_path + '/tbx_log'
        self.writer = SummaryWriter(self.save_tbx_log)

    def visualize_gt(self, var_map, i):
        count = i
        for kk in range(var_map.shape[0]):
            pred_edge_kk = var_map[kk, :, :, :]
            pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
            pred_edge_kk *= 255.0
            pred_edge_kk = pred_edge_kk.astype(np.uint8)
            name = '{:02d}_gt.png'.format(count)
            imageio.imwrite(self.image_save_path + "/train_" + name, pred_edge_kk)

    def visualize_prediction(self, var_map, i):
        count = i
        for kk in range(var_map.shape[0]):
            pred_edge_kk = var_map[kk, :, :, :]
            pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
            pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
            pred_edge_kk *= 255.0
            pred_edge_kk = pred_edge_kk.astype(np.uint8)
            name = '{:02d}_pred.png'.format(count)
            imageio.imwrite(self.image_save_path + "/train_" + name, pred_edge_kk)



    def visualize_val_gt(self, var_map, i):
        count = i
        for kk in range(var_map.shape[0]):
            pred_edge_kk = var_map[kk, :, :, :]
            pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
            pred_edge_kk *= 255.0
            pred_edge_kk = pred_edge_kk.astype(np.uint8)
            name = '{}_gt.png'.format(count)
            imageio.imwrite(self.image_save_path + "/val_" + name, pred_edge_kk)

    def visualize_val_prediction(self, var_map, i):
        count = i
        for kk in range(var_map.shape[0]):
            pred_edge_kk = var_map[kk, :, :, :]
            pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
            pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
            pred_edge_kk *= 255.0
            pred_edge_kk = pred_edge_kk.astype(np.uint8)
            name = '{}_pred.png'.format(count)
            imageio.imwrite(self.image_save_path + "/val_" + name, pred_edge_kk)

    def run(self):

        # build models
        model = M.segmentation.fcn_resnet101(pretrained=False, progress=True, num_classes=1, aux_loss=None)
        #model = M.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=1, aux_loss=None)
        model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), opt.lr_gen)

        image_root = self.cfg[opt.dataset]['image_dir']
        gt_root = self.cfg[opt.dataset]['mask_dir']
        val_image_root = self.cfg[opt.dataset]['val_image_dir']
        val_gt_root = self.cfg[opt.dataset]['val_mask_dir']

        train_loader, val_loader = get_loader(image_root, gt_root, val_image_root, val_gt_root, batchsize=opt.batchsize,
                                              trainsize=opt.trainsize)

        total_step = len(train_loader)
        val_total_step = len(val_loader)

        print("Let's go!")
        for epoch in range(1, opt.epoch):
            running_dice = 0.0
            running_loss = 0.0

            for i, pack in enumerate(train_loader, start=1):
                optimizer.zero_grad()

                images, gts = pack
                images = Variable(images)
                gts = Variable(gts)
                images = images.cuda()
                gts = gts.cuda()


                pred = torch.sigmoid(model(images)['out']) # This is for DeepLab
                loss = calc_loss(pred, gts)

                loss.backward()
                optimizer.step()

                dice_coe = dice(pred, gts)
                running_dice += dice_coe
                running_loss += loss


                if i % 10 == 0 or i == total_step:
                    self.logger.info(
                        '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:.4f}, dice_coe: {:.4f}'.
                            format(datetime.now(), epoch, opt.epoch, i, total_step, loss.item(), dice_coe))

            epoch_dice = running_dice / len(train_loader)
            epoch_loss = running_loss / len(train_loader)
            self.logger.info('Train dice coeff: {}'.format(epoch_dice))
            self.writer.add_scalar('Train/DSC', epoch_dice, epoch)
            self.writer.add_scalar('Train/Loss', epoch_loss, epoch)

            #Comented to test deeplab
            val_running_dice = 0.0
            val_running_loss = 0.0


            #for i, pack in enumerate(val_loader, start=1):
            for i, pack in enumerate(val_loader, start=1):
                with torch.no_grad():
                    #images, gts, name = pack
                    images, gts = pack
                    images = Variable(images)
                    gts = Variable(gts)

                    images = images.cuda()

                    gts = gts.cuda()


                    #print(images)
                    #print(model(images))
                    pred = torch.sigmoid(model(images)['out']) # This is for DeepLab

                val_loss = calc_loss(pred, gts)
                self.visualize_val_gt(gts, 'gt{}'.format(i))
                self.visualize_val_prediction(pred, 'pd{}'.format(i))



                val_dice_coe = dice(pred, gts)
                val_running_dice += val_dice_coe
                val_running_loss += val_loss
            

                if i % 10 == 0 or i == total_step:
                    self.logger.info(
                        '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Validation loss: {:.4f}, dice_coe: {:.4f}'.
                            format(datetime.now(), epoch, opt.epoch, i, val_total_step, val_loss.item(), val_dice_coe))


            val_epoch_dice = val_running_dice / len(val_loader)
            val_epoch_loss = val_running_loss / len(val_loader)
            self.logger.info('Validation dice coeff: {}'.format(val_epoch_dice))
            self.writer.add_scalar('Validation/DSC', val_epoch_dice, epoch)
            self.writer.add_scalar('Validation/Loss', val_epoch_loss, epoch)

            mdice_coeff = val_epoch_dice

            if self.best_dice_coeff < mdice_coeff:
                self.best_dice_coeff = mdice_coeff
                self.save_best = True

                if not os.path.exists(self.image_save_path):
                    os.makedirs(self.image_save_path)

                copy_tree(self.image_save_path, self.save_path + '/best_model_predictions')
                self.patience = 0
            else:
                self.save_best = False
                self.patience += 1

            # adjust_lr(optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)
            Checkpoints_Path = self.save_path + '/Checkpoints'
            if not os.path.exists(Checkpoints_Path):
                os.makedirs(Checkpoints_Path)

            if self.save_best:
                torch.save(model.state_dict(), Checkpoints_Path + '/Model_897_t1.pth')

            self.logger.info('current best dice coef {}'.format(self.best_dice_coeff))
            self.logger.info('current patience :{}'.format(self.patience))


if __name__ == '__main__':
    train_network = Network()
    train_network.run()
