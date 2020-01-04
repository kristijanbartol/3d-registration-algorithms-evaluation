#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import gc
import argparse
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data import ModelNet40
from model import DCP
from util import transform_point_cloud, npmat2euler
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from os.path import join

import copy
import time
import json

#DATA_DIR = '/home/kristijan/phd/datasets/Stanford3DDataset/'
ITEMS = ['bunny', 'horse', 'hand', 'dragon', 'happy']#, 'blade']

VOXEL_SIZE = 0.05   # means 5cm for the dataset

result_dict = {}


parser = argparse.ArgumentParser(description='Point Cloud Registration')
parser.add_argument('--exp_name', type=str, default='dcp_v1', metavar='N',
                    help='Name of the experiment')
parser.add_argument('--model', type=str, default='dcp', metavar='N',
                    choices=['dcp'],
                    help='Model to use, [dcp]')
parser.add_argument('--emb_nn', type=str, default='dgcnn', metavar='N',
                    choices=['pointnet', 'dgcnn'],
                    help='Embedding nn to use, [pointnet, dgcnn]')
parser.add_argument('--pointer', type=str, default='identity', metavar='N',
                    choices=['identity', 'transformer'],
                    help='Attention-based pointer generator to use, [identity, transformer]')
parser.add_argument('--head', type=str, default='svd', metavar='N',
                    choices=['mlp', 'svd', ],
                    help='Head to use, [mlp, svd]')
parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                    help='Dimension of embeddings')
parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                    help='Num of blocks of encoder&decoder')
parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                    help='Num of heads in multiheadedattention')
parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                    help='Num of dimensions of fc in transformer')
parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                    help='Dropout ratio in transformer')
parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--test_batch_size', type=int, default=10, metavar='batch_size',
                    help='Size of batch)')
parser.add_argument('--epochs', type=int, default=250, metavar='N',
                    help='number of episode to train ')
parser.add_argument('--use_sgd', action='store_true', default=False,
                    help='Use SGD')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--eval', action='store_true', default=False,
                    help='evaluate the model')
parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                    help='Whether to use cycle consistency')
parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                    help='Wheter to add gaussian noise')
parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                    help='Wheter to test on unseen category')
parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                    help='Num of points to use')
parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'], metavar='N',
                    help='dataset to use')
parser.add_argument('--factor', type=float, default=4, metavar='N',
                    help='Divided factor for rotations')
parser.add_argument('--model_path', type=str, default='pretrained/dcp_v1.t7', metavar='N',
                    help='Pretrained model path')


# Part of the code is referred from: https://github.com/floodsung/LearningToCompare_FSL

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def test_single_registration(net, src, tgt):
    net.eval()
    src = src.astype(np.float32)
    tgt = tgt.astype(np.float32)
    src = np.swapaxes(src, 0, 1)
    tgt = np.swapaxes(tgt, 0, 1)
    src = np.expand_dims(src, axis=0)
    tgt = np.expand_dims(tgt, axis=0)
    src = torch.from_numpy(src).cuda()
    tgt = torch.from_numpy(tgt).cuda()
    rot_ab_pred, trans_ab_pred, rot_ba_pred, trans_ba_pred = net(src, tgt)

    return rot_ab_pred, trans_ab_pred
    

def test_one_epoch(args, net, test_loader):
    net.eval()
    mse_ab = 0
    mae_ab = 0
    mse_ba = 0
    mae_ba = 0

    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    rotations_ba = []
    translations_ba = []
    rotations_ba_pred = []
    translations_ba_pred = []

    eulers_ab = []
    eulers_ba = []

    for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba in tqdm(test_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()
        rotation_ba = rotation_ba.cuda()
        translation_ba = translation_ba.cuda()

        batch_size = src.size(0)
        num_examples += batch_size
        print(src.shape)
        rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(src, target)

        ## save rotation and translation
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        eulers_ab.append(euler_ab.numpy())
        ##
        rotations_ba.append(rotation_ba.detach().cpu().numpy())
        translations_ba.append(translation_ba.detach().cpu().numpy())
        rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
        translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())
        eulers_ba.append(euler_ba.numpy())

        transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)

        transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)

        ###########################
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
               + F.mse_loss(translation_ab_pred, translation_ab)
        if args.cycle:
            rotation_loss = F.mse_loss(torch.matmul(rotation_ba_pred, rotation_ab_pred), identity.clone())
            translation_loss = torch.mean((torch.matmul(rotation_ba_pred.transpose(2, 1),
                                                        translation_ab_pred.view(batch_size, 3, 1)).view(batch_size, 3)
                                           + translation_ba_pred) ** 2, dim=[0, 1])
            cycle_loss = rotation_loss + translation_loss

            loss = loss + cycle_loss * 0.1

        total_loss += loss.item() * batch_size

        if args.cycle:
            total_cycle_loss = total_cycle_loss + cycle_loss.item() * 0.1 * batch_size

        mse_ab += torch.mean((transformed_src - target) ** 2, dim=[0, 1, 2]).item() * batch_size
        mae_ab += torch.mean(torch.abs(transformed_src - target), dim=[0, 1, 2]).item() * batch_size

        mse_ba += torch.mean((transformed_target - src) ** 2, dim=[0, 1, 2]).item() * batch_size
        mae_ba += torch.mean(torch.abs(transformed_target - src), dim=[0, 1, 2]).item() * batch_size

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    rotations_ba = np.concatenate(rotations_ba, axis=0)
    translations_ba = np.concatenate(translations_ba, axis=0)
    rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
    translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)

    eulers_ab = np.concatenate(eulers_ab, axis=0)
    eulers_ba = np.concatenate(eulers_ba, axis=0)

    return total_loss * 1.0 / num_examples, total_cycle_loss / num_examples, \
           mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples, \
           mse_ba * 1.0 / num_examples, mae_ba * 1.0 / num_examples, rotations_ab, \
           translations_ab, rotations_ab_pred, translations_ab_pred, rotations_ba, \
           translations_ba, rotations_ba_pred, translations_ba_pred, eulers_ab, eulers_ba


def train_one_epoch(args, net, train_loader, opt):
    net.train()

    mse_ab = 0
    mae_ab = 0
    mse_ba = 0
    mae_ba = 0

    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    rotations_ba = []
    translations_ba = []
    rotations_ba_pred = []
    translations_ba_pred = []

    eulers_ab = []
    eulers_ba = []

    for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba in tqdm(train_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()
        rotation_ba = rotation_ba.cuda()
        translation_ba = translation_ba.cuda()

        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size
        rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(src, target)

        ## save rotation and translation
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        eulers_ab.append(euler_ab.numpy())
        ##
        rotations_ba.append(rotation_ba.detach().cpu().numpy())
        translations_ba.append(translation_ba.detach().cpu().numpy())
        rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
        translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())
        eulers_ba.append(euler_ba.numpy())

        transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)

        transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)
        ###########################
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
               + F.mse_loss(translation_ab_pred, translation_ab)
        if args.cycle:
            rotation_loss = F.mse_loss(torch.matmul(rotation_ba_pred, rotation_ab_pred), identity.clone())
            translation_loss = torch.mean((torch.matmul(rotation_ba_pred.transpose(2, 1),
                                                        translation_ab_pred.view(batch_size, 3, 1)).view(batch_size, 3)
                                           + translation_ba_pred) ** 2, dim=[0, 1])
            cycle_loss = rotation_loss + translation_loss

            loss = loss + cycle_loss * 0.1

        loss.backward()
        opt.step()
        total_loss += loss.item() * batch_size

        if args.cycle:
            total_cycle_loss = total_cycle_loss + cycle_loss.item() * 0.1 * batch_size

        mse_ab += torch.mean((transformed_src - target) ** 2, dim=[0, 1, 2]).item() * batch_size
        mae_ab += torch.mean(torch.abs(transformed_src - target), dim=[0, 1, 2]).item() * batch_size

        mse_ba += torch.mean((transformed_target - src) ** 2, dim=[0, 1, 2]).item() * batch_size
        mae_ba += torch.mean(torch.abs(transformed_target - src), dim=[0, 1, 2]).item() * batch_size

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    rotations_ba = np.concatenate(rotations_ba, axis=0)
    translations_ba = np.concatenate(translations_ba, axis=0)
    rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
    translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)

    eulers_ab = np.concatenate(eulers_ab, axis=0)
    eulers_ba = np.concatenate(eulers_ba, axis=0)

    return total_loss * 1.0 / num_examples, total_cycle_loss / num_examples, \
           mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples, \
           mse_ba * 1.0 / num_examples, mae_ba * 1.0 / num_examples, rotations_ab, \
           translations_ab, rotations_ab_pred, translations_ab_pred, rotations_ba, \
           translations_ba, rotations_ba_pred, translations_ba_pred, eulers_ab, eulers_ba


def test(args, net, test_loader, boardio, textio):

    test_loss, test_cycle_loss, \
    test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, test_rotations_ab, test_translations_ab, \
    test_rotations_ab_pred, \
    test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
    test_translations_ba_pred, test_eulers_ab, test_eulers_ba = test_one_epoch(args, net, test_loader)
    test_rmse_ab = np.sqrt(test_mse_ab)
    test_rmse_ba = np.sqrt(test_mse_ba)

    test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
    test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)) ** 2)
    test_r_rmse_ab = np.sqrt(test_r_mse_ab)
    test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)))
    test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
    test_t_rmse_ab = np.sqrt(test_t_mse_ab)
    test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))

    test_rotations_ba_pred_euler = npmat2euler(test_rotations_ba_pred, 'xyz')
    test_r_mse_ba = np.mean((test_rotations_ba_pred_euler - np.degrees(test_eulers_ba)) ** 2)
    test_r_rmse_ba = np.sqrt(test_r_mse_ba)
    test_r_mae_ba = np.mean(np.abs(test_rotations_ba_pred_euler - np.degrees(test_eulers_ba)))
    test_t_mse_ba = np.mean((test_translations_ba - test_translations_ba_pred) ** 2)
    test_t_rmse_ba = np.sqrt(test_t_mse_ba)
    test_t_mae_ba = np.mean(np.abs(test_translations_ba - test_translations_ba_pred))

    textio.cprint('==FINAL TEST==')
    textio.cprint('A--------->B')
    textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                  % (-1, test_loss, test_cycle_loss, test_mse_ab, test_rmse_ab, test_mae_ab,
                     test_r_mse_ab, test_r_rmse_ab,
                     test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))
    textio.cprint('B--------->A')
    textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                  'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                  % (-1, test_loss, test_mse_ba, test_rmse_ba, test_mae_ba, test_r_mse_ba, test_r_rmse_ba,
                     test_r_mae_ba, test_t_mse_ba, test_t_rmse_ba, test_t_mae_ba))


def train(args, net, train_loader, test_loader, boardio, textio):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)


    best_test_loss = np.inf
    best_test_cycle_loss = np.inf
    best_test_mse_ab = np.inf
    best_test_rmse_ab = np.inf
    best_test_mae_ab = np.inf

    best_test_r_mse_ab = np.inf
    best_test_r_rmse_ab = np.inf
    best_test_r_mae_ab = np.inf
    best_test_t_mse_ab = np.inf
    best_test_t_rmse_ab = np.inf
    best_test_t_mae_ab = np.inf

    best_test_mse_ba = np.inf
    best_test_rmse_ba = np.inf
    best_test_mae_ba = np.inf

    best_test_r_mse_ba = np.inf
    best_test_r_rmse_ba = np.inf
    best_test_r_mae_ba = np.inf
    best_test_t_mse_ba = np.inf
    best_test_t_rmse_ba = np.inf
    best_test_t_mae_ba = np.inf

    for epoch in range(args.epochs):
        scheduler.step()
        train_loss, train_cycle_loss, \
        train_mse_ab, train_mae_ab, train_mse_ba, train_mae_ba, train_rotations_ab, train_translations_ab, \
        train_rotations_ab_pred, \
        train_translations_ab_pred, train_rotations_ba, train_translations_ba, train_rotations_ba_pred, \
        train_translations_ba_pred, train_eulers_ab, train_eulers_ba = train_one_epoch(args, net, train_loader, opt)
        test_loss, test_cycle_loss, \
        test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, test_rotations_ab, test_translations_ab, \
        test_rotations_ab_pred, \
        test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
        test_translations_ba_pred, test_eulers_ab, test_eulers_ba = test_one_epoch(args, net, test_loader)
        train_rmse_ab = np.sqrt(train_mse_ab)
        test_rmse_ab = np.sqrt(test_mse_ab)

        train_rmse_ba = np.sqrt(train_mse_ba)
        test_rmse_ba = np.sqrt(test_mse_ba)

        train_rotations_ab_pred_euler = npmat2euler(train_rotations_ab_pred)
        train_r_mse_ab = np.mean((train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)) ** 2)
        train_r_rmse_ab = np.sqrt(train_r_mse_ab)
        train_r_mae_ab = np.mean(np.abs(train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)))
        train_t_mse_ab = np.mean((train_translations_ab - train_translations_ab_pred) ** 2)
        train_t_rmse_ab = np.sqrt(train_t_mse_ab)
        train_t_mae_ab = np.mean(np.abs(train_translations_ab - train_translations_ab_pred))

        train_rotations_ba_pred_euler = npmat2euler(train_rotations_ba_pred, 'xyz')
        train_r_mse_ba = np.mean((train_rotations_ba_pred_euler - np.degrees(train_eulers_ba)) ** 2)
        train_r_rmse_ba = np.sqrt(train_r_mse_ba)
        train_r_mae_ba = np.mean(np.abs(train_rotations_ba_pred_euler - np.degrees(train_eulers_ba)))
        train_t_mse_ba = np.mean((train_translations_ba - train_translations_ba_pred) ** 2)
        train_t_rmse_ba = np.sqrt(train_t_mse_ba)
        train_t_mae_ba = np.mean(np.abs(train_translations_ba - train_translations_ba_pred))

        test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
        test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)) ** 2)
        test_r_rmse_ab = np.sqrt(test_r_mse_ab)
        test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)))
        test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
        test_t_rmse_ab = np.sqrt(test_t_mse_ab)
        test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))

        test_rotations_ba_pred_euler = npmat2euler(test_rotations_ba_pred, 'xyz')
        test_r_mse_ba = np.mean((test_rotations_ba_pred_euler - np.degrees(test_eulers_ba)) ** 2)
        test_r_rmse_ba = np.sqrt(test_r_mse_ba)
        test_r_mae_ba = np.mean(np.abs(test_rotations_ba_pred_euler - np.degrees(test_eulers_ba)))
        test_t_mse_ba = np.mean((test_translations_ba - test_translations_ba_pred) ** 2)
        test_t_rmse_ba = np.sqrt(test_t_mse_ba)
        test_t_mae_ba = np.mean(np.abs(test_translations_ba - test_translations_ba_pred))

        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            best_test_cycle_loss = test_cycle_loss

            best_test_mse_ab = test_mse_ab
            best_test_rmse_ab = test_rmse_ab
            best_test_mae_ab = test_mae_ab

            best_test_r_mse_ab = test_r_mse_ab
            best_test_r_rmse_ab = test_r_rmse_ab
            best_test_r_mae_ab = test_r_mae_ab

            best_test_t_mse_ab = test_t_mse_ab
            best_test_t_rmse_ab = test_t_rmse_ab
            best_test_t_mae_ab = test_t_mae_ab

            best_test_mse_ba = test_mse_ba
            best_test_rmse_ba = test_rmse_ba
            best_test_mae_ba = test_mae_ba

            best_test_r_mse_ba = test_r_mse_ba
            best_test_r_rmse_ba = test_r_rmse_ba
            best_test_r_mae_ba = test_r_mae_ba

            best_test_t_mse_ba = test_t_mse_ba
            best_test_t_rmse_ba = test_t_rmse_ba
            best_test_t_mae_ba = test_t_mae_ba

            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)

        textio.cprint('==TRAIN==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss:, %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, train_loss, train_cycle_loss, train_mse_ab, train_rmse_ab, train_mae_ab, train_r_mse_ab,
                         train_r_rmse_ab, train_r_mae_ab, train_t_mse_ab, train_t_rmse_ab, train_t_mae_ab))
        textio.cprint('B--------->A')
        textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, train_loss, train_mse_ba, train_rmse_ba, train_mae_ba, train_r_mse_ba, train_r_rmse_ba,
                         train_r_mae_ba, train_t_mse_ba, train_t_rmse_ba, train_t_mae_ba))

        textio.cprint('==TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, test_loss, test_cycle_loss, test_mse_ab, test_rmse_ab, test_mae_ab, test_r_mse_ab,
                         test_r_rmse_ab, test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))
        textio.cprint('B--------->A')
        textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, test_loss, test_mse_ba, test_rmse_ba, test_mae_ba, test_r_mse_ba, test_r_rmse_ba,
                         test_r_mae_ba, test_t_mse_ba, test_t_rmse_ba, test_t_mae_ba))

        textio.cprint('==BEST TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, best_test_loss, best_test_cycle_loss, best_test_mse_ab, best_test_rmse_ab,
                         best_test_mae_ab, best_test_r_mse_ab, best_test_r_rmse_ab,
                         best_test_r_mae_ab, best_test_t_mse_ab, best_test_t_rmse_ab, best_test_t_mae_ab))
        textio.cprint('B--------->A')
        textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, best_test_loss, best_test_mse_ba, best_test_rmse_ba, best_test_mae_ba,
                         best_test_r_mse_ba, best_test_r_rmse_ba,
                         best_test_r_mae_ba, best_test_t_mse_ba, best_test_t_rmse_ba, best_test_t_mae_ba))

        boardio.add_scalar('A->B/train/loss', train_loss, epoch)
        boardio.add_scalar('A->B/train/MSE', train_mse_ab, epoch)
        boardio.add_scalar('A->B/train/RMSE', train_rmse_ab, epoch)
        boardio.add_scalar('A->B/train/MAE', train_mae_ab, epoch)
        boardio.add_scalar('A->B/train/rotation/MSE', train_r_mse_ab, epoch)
        boardio.add_scalar('A->B/train/rotation/RMSE', train_r_rmse_ab, epoch)
        boardio.add_scalar('A->B/train/rotation/MAE', train_r_mae_ab, epoch)
        boardio.add_scalar('A->B/train/translation/MSE', train_t_mse_ab, epoch)
        boardio.add_scalar('A->B/train/translation/RMSE', train_t_rmse_ab, epoch)
        boardio.add_scalar('A->B/train/translation/MAE', train_t_mae_ab, epoch)

        boardio.add_scalar('B->A/train/loss', train_loss, epoch)
        boardio.add_scalar('B->A/train/MSE', train_mse_ba, epoch)
        boardio.add_scalar('B->A/train/RMSE', train_rmse_ba, epoch)
        boardio.add_scalar('B->A/train/MAE', train_mae_ba, epoch)
        boardio.add_scalar('B->A/train/rotation/MSE', train_r_mse_ba, epoch)
        boardio.add_scalar('B->A/train/rotation/RMSE', train_r_rmse_ba, epoch)
        boardio.add_scalar('B->A/train/rotation/MAE', train_r_mae_ba, epoch)
        boardio.add_scalar('B->A/train/translation/MSE', train_t_mse_ba, epoch)
        boardio.add_scalar('B->A/train/translation/RMSE', train_t_rmse_ba, epoch)
        boardio.add_scalar('B->A/train/translation/MAE', train_t_mae_ba, epoch)

        ############TEST
        boardio.add_scalar('A->B/test/loss', test_loss, epoch)
        boardio.add_scalar('A->B/test/MSE', test_mse_ab, epoch)
        boardio.add_scalar('A->B/test/RMSE', test_rmse_ab, epoch)
        boardio.add_scalar('A->B/test/MAE', test_mae_ab, epoch)
        boardio.add_scalar('A->B/test/rotation/MSE', test_r_mse_ab, epoch)
        boardio.add_scalar('A->B/test/rotation/RMSE', test_r_rmse_ab, epoch)
        boardio.add_scalar('A->B/test/rotation/MAE', test_r_mae_ab, epoch)
        boardio.add_scalar('A->B/test/translation/MSE', test_t_mse_ab, epoch)
        boardio.add_scalar('A->B/test/translation/RMSE', test_t_rmse_ab, epoch)
        boardio.add_scalar('A->B/test/translation/MAE', test_t_mae_ab, epoch)

        boardio.add_scalar('B->A/test/loss', test_loss, epoch)
        boardio.add_scalar('B->A/test/MSE', test_mse_ba, epoch)
        boardio.add_scalar('B->A/test/RMSE', test_rmse_ba, epoch)
        boardio.add_scalar('B->A/test/MAE', test_mae_ba, epoch)
        boardio.add_scalar('B->A/test/rotation/MSE', test_r_mse_ba, epoch)
        boardio.add_scalar('B->A/test/rotation/RMSE', test_r_rmse_ba, epoch)
        boardio.add_scalar('B->A/test/rotation/MAE', test_r_mae_ba, epoch)
        boardio.add_scalar('B->A/test/translation/MSE', test_t_mse_ba, epoch)
        boardio.add_scalar('B->A/test/translation/RMSE', test_t_rmse_ba, epoch)
        boardio.add_scalar('B->A/test/translation/MAE', test_t_mae_ba, epoch)

        ############BEST TEST
        boardio.add_scalar('A->B/best_test/loss', best_test_loss, epoch)
        boardio.add_scalar('A->B/best_test/MSE', best_test_mse_ab, epoch)
        boardio.add_scalar('A->B/best_test/RMSE', best_test_rmse_ab, epoch)
        boardio.add_scalar('A->B/best_test/MAE', best_test_mae_ab, epoch)
        boardio.add_scalar('A->B/best_test/rotation/MSE', best_test_r_mse_ab, epoch)
        boardio.add_scalar('A->B/best_test/rotation/RMSE', best_test_r_rmse_ab, epoch)
        boardio.add_scalar('A->B/best_test/rotation/MAE', best_test_r_mae_ab, epoch)
        boardio.add_scalar('A->B/best_test/translation/MSE', best_test_t_mse_ab, epoch)
        boardio.add_scalar('A->B/best_test/translation/RMSE', best_test_t_rmse_ab, epoch)
        boardio.add_scalar('A->B/best_test/translation/MAE', best_test_t_mae_ab, epoch)

        boardio.add_scalar('B->A/best_test/loss', best_test_loss, epoch)
        boardio.add_scalar('B->A/best_test/MSE', best_test_mse_ba, epoch)
        boardio.add_scalar('B->A/best_test/RMSE', best_test_rmse_ba, epoch)
        boardio.add_scalar('B->A/best_test/MAE', best_test_mae_ba, epoch)
        boardio.add_scalar('B->A/best_test/rotation/MSE', best_test_r_mse_ba, epoch)
        boardio.add_scalar('B->A/best_test/rotation/RMSE', best_test_r_rmse_ba, epoch)
        boardio.add_scalar('B->A/best_test/rotation/MAE', best_test_r_mae_ba, epoch)
        boardio.add_scalar('B->A/best_test/translation/MSE', best_test_t_mse_ba, epoch)
        boardio.add_scalar('B->A/best_test/translation/RMSE', best_test_t_rmse_ba, epoch)
        boardio.add_scalar('B->A/best_test/translation/MAE', best_test_t_mae_ba, epoch)

        if torch.cuda.device_count() > 1:
            torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        else:
            torch.save(net.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        gc.collect()


def main(args):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    _init_(args)

    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))

    if args.dataset == 'modelnet40':
        train_loader = DataLoader(
            ModelNet40(num_points=args.num_points, partition='train', gaussian_noise=args.gaussian_noise,
                       unseen=args.unseen, factor=args.factor),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            ModelNet40(num_points=args.num_points, partition='test', gaussian_noise=args.gaussian_noise,
                       unseen=args.unseen, factor=args.factor),
            batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    else:
        raise Exception("not implemented")

    if args.model == 'dcp':
        net = DCP(args).cuda()
        if args.eval:
            if args.model_path is '':
                model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
            else:
                model_path = args.model_path
                print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            net.load_state_dict(torch.load(model_path), strict=False)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        raise Exception('Not implemented')
    if args.eval:
        test(args, net, test_loader, boardio, textio)
    else:
        train(args, net, train_loader, test_loader, boardio, textio)


    print('FINISH')
    boardio.close()


def draw_registration_result(source, target, transformation=None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    if transformation is not None:
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def save_results(source, target, result, data_root, object_name, algo):
    metrics_path = join(data_root, '{}_{}.json'.format(object_name, algo))
    metrics = { 
                'rmse': result.inlier_rmse, 
                'fitness': result.fitness
              }
    with open(metrics_path, 'w') as mp:
        json.dump(metrics, mp)
    result_dict['{}_{}'.format(object_name, algo)] = metrics


def preprocess_point_cloud(pcd, voxel_size, object_name):
    start_time = time.time()
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    start_time = time.time()
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    start_time = time.time()
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    print('[{}] Compute FPFH feature with search radius {:.3f} ({:.2f}s).'.format(
        object_name, radius_feature, time.time() - start_time))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size, data_root, object_name):
    source_path = join(data_root, '{}.{}'.format(object_name, 'ply'))
    target_path = join(data_root, '{}_trans.{}'.format(object_name, 'ply'))
    source = o3d.io.read_point_cloud(source_path)
    #target = source.translate(np.array([0.1, 0.1, 0.1])).rotate(np.array([0.5, 0.5, 0.5]))
    #o3d.io.write_point_cloud(target_path, target, write_ascii=True)
    target = o3d.io.read_point_cloud(target_path)
    #trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                         [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    #source.transform(trans_init)
    #draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size, object_name)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size, object_name)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    result = o3d.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac, icp_type):
    distance_threshold = voxel_size * 0.4
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPoint() if icp_type == 'point' \
            else o3d.registration.TransformationEstimationPointToPlane())
    return result


def coarse_fine_matching(name, icp_type):
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(VOXEL_SIZE, DATA_DIR, name)

    start_time = time.time()
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                VOXEL_SIZE)
    print('[{}] Result RANSAC: {} ({:.2f}s)'.format(name, result_ransac, time.time() - start_time))
    draw_registration_result(source, target,
                            result_ransac.transformation)
    save_results(source, target, result_ransac, DATA_DIR, name, 'fpfh-ransac')

    start_time = time.time()
    result_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh,
                                    VOXEL_SIZE, result_ransac, icp_type)
    print('[{}] Result ICP: {} ({:.2f}s)'.format(name, result_icp, time.time() - start_time))
    draw_registration_result(source, target, result_icp.transformation)
    
    save_results(source, target, result_icp, DATA_DIR, name, 'fpfh-ransac-{}-icp'.format(icp_type))


def fast_global_matching(name, icp_type):
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(VOXEL_SIZE, DATA_DIR, name)

    start_time = time.time()
    result_fast = execute_fast_global_registration(source_down, target_down,
                                                   source_fpfh, target_fpfh,
                                                   VOXEL_SIZE)
    result_fast = o3d.registration.evaluate_registration(
        source, target, 0.01, transformation=result_fast.transformation)                                     
    print('[{}] Result FGR: {} ({:.2f}s)'.format(name, result_fast, time.time() - start_time))
    draw_registration_result(source, target,
                             result_fast.transformation)
    save_results(source, target, result_fast, DATA_DIR, name, 'fgr')

    start_time = time.time()
    result_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh,
                                    VOXEL_SIZE, result_fast, icp_type)
    print('[{}] Result ICP: {} ({:.2f}s)'.format(name, result_icp, time.time() - start_time))
    draw_registration_result(source, target, result_icp.transformation)
    save_results(source, target, result_icp, DATA_DIR, name, 'fgr-{}-icp'.format(icp_type))


def loadPointCloud(filename):
    pcloud = np.loadtxt(filename, skiprows=1)
    plist = pcloud.tolist()
    p3dlist = []
    for x, y, z in plist:
        pt = POINT3D(x, y, z)
        p3dlist.append(pt)
    return pcloud.shape[0], p3dlist


def goicp_to_o3d(points):
    pc_list = []
    for point in points:
        pc_list.append([point.x, point.y, point.z])
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(np.array(pc_list))
    return o3d_pc


def get_transformation(rotation, translation):
    return np.array([[rotation[0][0], rotation[0][1], rotation[0][2],
        translation[0]], [rotation[1][0], rotation[1][1], rotation[1][2],
        translation[1]], [rotation[2][0], rotation[2][1], rotation[2][2],
        translation[2]], [0., 0., 0., 1.]])


def numpy_to_point_cloud(np_point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_point_cloud)
    return pcd


def go_icp(_goicp, name):
    src_path = join(DATA_DIR, '{}.{}'.format(name, 'txt'))
    tgt_path = join(DATA_DIR, '{}_trans.{}'.format(name, 'txt'))
    N_src, src_points = loadPointCloud(src_path)
    N_tgt, tgt_points = loadPointCloud(tgt_path)
    _goicp.loadModelAndData(N_src, src_points, N_tgt, tgt_points)
    _goicp.setDTSizeAndFactor(300, 2.0)

    start_time = time.time()
    _goicp.BuildDT()
    _goicp.Register()
    goicp_time = time.time() - start_time

    source = _goicp(src_points)
    target = _goicp(tgt_points)
    draw_registration_result(source, target)

    goicp_transformation = get_transformation(
        _goicp.optimalRotation(), 
        _goicp.optimalTranslation())
    draw_registration_result(target, source, goicp_transformation)

    result_goicp = o3d.registration.evaluate_registration(
        target, source, 0.001, transformation=goicp_transformation)
    print(goicp_transformation) 
    print('[{}] Result Go-ICP: {} ({:.2f})'.format(name, result_goicp, goicp_time))

    save_results(source, target, result_goicp, DATA_DIR, name, 'goicp')


class DCPEval():

    def __init__(self):
        self.args = parser.parse_args()
        self.net = DCP(self.args)
        self.model_path = self.args.model_path
        self.net.load_state_dict(torch.load(self.model_path), strict=False)

    def run(self, src, tgt):
        return test_single_registration(self.net, src, tgt)


def inference(args, name):
    net = DCP(args).cuda()
    model_path = args.model_path
    net.load_state_dict(torch.load(model_path), strict=False)

    #src_path = join('data/{}.{}'.format(name, 'ply'))
    #tgt_path = join('data/{}_trans.{}'.format(name, 'ply'))
    #src = o3d.io.read_point_cloud(src_path)
    #tgt = o3d.io.read_point_cloud(tgt_path)

    source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(VOXEL_SIZE, 'data/', name)

    _voxel_size = 0.001
    num_points = np.asarray(source.points).shape[0]
    while True:
        source = source.voxel_down_sample(voxel_size=_voxel_size)
        target = target.voxel_down_sample(voxel_size=_voxel_size)
        down_points = np.asarray(source.points).shape[0]
        print('[{}] Number of points: {} -> {}'.format(
                name, num_points, down_points))
        if down_points > 20000:
            _voxel_size += 0.0002
        else:
            break

    source = np.asarray(source.points)
    target = np.asarray(target.points)

    print(source.shape)

    start_time = time.time()
    result = test_single_registration(net, source, target)
    total_time = time.time() - start_time
    transformation = get_transformation(
        result[0][0].cpu().detach().numpy(), 
        result[1][0].cpu().detach().numpy())

    source = numpy_to_point_cloud(source.astype(np.float64))
    target = numpy_to_point_cloud(target.astype(np.float64))
    result_dcp = o3d.registration.evaluate_registration(
        source, target, 0.1, transformation=transformation)

    print('[{}] Result DCP: {} ({:.2f})'.format(name, result_dcp, total_time))

    result_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh,
                                    VOXEL_SIZE, result_dcp, 'point')
    print('[{}] Result ICP: {} ({:.2f})'.format(name, result_icp, 0.))


if __name__ == '__main__':
    args = parser.parse_args()
    #main(args)

    for name in ITEMS:
        inference(args, name)

    with open(join('data/', 'results.json'), 'w') as fr:
        json.dump(result_dict, fr)
