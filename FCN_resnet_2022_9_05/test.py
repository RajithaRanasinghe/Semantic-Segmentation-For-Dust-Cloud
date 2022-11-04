import argparse
import os
import matplotlib.pyplot as plt
import imageio
import time
import numpy as np
import torch
import yaml
from torch.autograd import Variable

from data import get_loader
from metrics import calculate_all_measures, confusion

import torchvision.models as M
from model.Unet import Unet

from utils import adjust_lr, get_logger, create_dir
from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=10, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=5e-4, help='learning rate')
parser.add_argument('--lr_dis', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--is_ResNet', type=bool, default=False, help='Unet or ResNet backbone')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
parser.add_argument('-beta1_dis', type=float, default=0.5, help='beta of Adam for descriptor')
parser.add_argument('--dataset', type=str, default='dust', help='dataset name')
opt = parser.parse_args()

class Test(object):
    def __init__(self):
        self.recall_array = [0,1]
        self.fallout_array = [0,1]
        self._init_configure()
        self._init_logger()

    def _init_configure(self):
        with open('configs/config.yml') as fp:
            self.cfg = yaml.safe_load(fp)

    def _init_logger(self):

        #self.model_name = 'FCN/fcn_resnet50'
        self.model_name = 'FCN/FCN_resnet101'
        

        #Dataset = "dataset_100"
        Dataset = "dataset_897"

        log_dir = 'logs/' + self.model_name + '/' + opt.dataset + '/test' + '/' + Dataset + '/{}'.format(
            time.strftime('%Y%m%d-%H%M%S'))

        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))

        self.save_path = log_dir
        self.image_save_path = log_dir + "/saved_images"
        create_dir(self.image_save_path)
        self.save_tbx_log = self.save_path + '/tbx_log'
        self.writer = SummaryWriter(self.save_tbx_log)
        
        #self.model_load_path = 'logs/FCN/FCN_resnet50/dust/train/20221023-125717/Checkpoints/Model_897_t1.pth' #Resnet_50_dataset897
        #self.model_load_path = 'logs/FCN/FCN_resnet50/dust/train/20221023-153853/Checkpoints/Model_100_t1.pth' #Resnet_50_dataset100
        #self.model_load_path = 'logs/FCN/FCN_resnet101/dust/train/20221023-162254/Checkpoints/Model_100_t1.pth' #Resnet_101_dataset100
        self.model_load_path = 'logs/FCN/FCN_resnet101/dust/train/20221023-170726/Checkpoints/Model_897_t1.pth' #Resnet_101_dataset897
        

    def dice(self, pred, target):
        intersection = (abs(target - pred) < 0.05).sum()
        cardinality = (target >= 0).sum() + (pred >= 0).sum()

        return 2.0 * intersection / cardinality

    def visualize_val_gt(self, var_map, i):
        count = i
        for kk in range(var_map.shape[0]):
            pred_edge_kk = var_map[kk, :, :, :]
            pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
            pred_edge_kk *= 255.0
            pred_edge_kk = pred_edge_kk.astype(np.uint8)
            name = '{:02d}_gt.png'.format(count)
            imageio.imwrite(self.image_save_path + "/val_" + name, pred_edge_kk)

    def visualize_val_prediction(self, var_map, i):
        count = i
        for kk in range(var_map.shape[0]):
            pred_edge_kk = var_map[kk, :, :, :]
            pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
            pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
            pred_edge_kk *= 255.0
            pred_edge_kk = pred_edge_kk.astype(np.uint8)
            name = '{:02d}_pred.png'.format(count)
            imageio.imwrite(self.image_save_path + "/val_" + name, pred_edge_kk)

    def compute_roc(self, recall_a, fallout_a):
        roc = 0.0

        a = np.column_stack((recall_a, fallout_a))
        ind = np.argsort(a[:, 1])
        sorted_array = a[ind]
        n_recall_array = []
        n_fallout_array = []

        for i in range(len(sorted_array) - 1):
            n_recall_array.append(sorted_array[i][0])
            n_fallout_array.append(sorted_array[i][1])
            roc = roc + ((sorted_array[i + 1][0] + sorted_array[i][0]) * abs(
                sorted_array[i + 1][1] - sorted_array[i][1])) / 2

        n_fallout_array.append(1)
        n_recall_array.append(1)
        plt.plot(n_fallout_array, n_recall_array, 'b', label='AUC = %0.2f' % roc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Fallout (False Positive Rate)')
        plt.ylabel('Recall (True Positive Rate)')
        plt.title('Receiver Operating Characteristic')

        # save the figure
        plt.savefig(self.save_path + '/ROC.png', dpi=300, bbox_inches='tight')

        return roc

    def run(self):

        # build models
        #model = M.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=1, aux_loss=None)
        model = M.segmentation.fcn_resnet101(pretrained=False, progress=True, num_classes=1, aux_loss=None)
        
        
        
        model.load_state_dict(torch.load(self.model_load_path))
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(pytorch_total_params)
        model.cuda()


        image_root = self.cfg[opt.dataset]['image_dir']
        gt_root = self.cfg[opt.dataset]['mask_dir']
        val_image_root = self.cfg[opt.dataset]['val_image_dir']
        val_gt_root = self.cfg[opt.dataset]['val_mask_dir']

        val_running_dice = 0.0
        tot_TP, tot_TN, tot_FN, tot_FP, tot_recall, tot_fallout = 0, 0, 0, 0, 0, 0
        train_loader, val_loader = get_loader(image_root, gt_root, val_image_root, val_gt_root, batchsize=opt.batchsize,
                                              trainsize=opt.trainsize)

        for i, pack in enumerate(val_loader, start=1):
            with torch.no_grad():
                images, gts = pack
                images = Variable(images)
                gts = Variable(gts)
                images = images.cuda()
                gts = gts.cuda()
                prediction = model(images)['out']
                pred = torch.sigmoid(prediction)

            self.visualize_val_gt(gts, i)
            self.visualize_val_prediction(pred, i)

            val_dice_coe = self.dice(pred, gts)
            val_running_dice += val_dice_coe

            TP, FP, TN, FN, precision_v, recall_v, fallout_v, f1 = confusion(gts, pred)

            tot_TP += TP * val_loader.batch_size
            tot_FP += FP * val_loader.batch_size
            tot_TN += TN * val_loader.batch_size
            tot_FN += FN * val_loader.batch_size

            self.recall_array.append(recall_v)
            self.fallout_array.append(fallout_v)
            self.logger.info('recall: {} fall-out: {}'.format(recall_v, fallout_v))

        roc = self.compute_roc(self.recall_array, self.fallout_array)
        self.logger.info("ROC : {}".format(roc))


if __name__ == '__main__':
    Test_network = Test()
    Test_network.run()