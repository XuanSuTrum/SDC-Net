# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 21:53:27 2023

@author: Administrator
"""
import numpy as np
import torch.utils.data as Data
import torch
# import ctypes
# ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
from torch.nn import init
import os
import random
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD, RMSprop
from typing import Optional
import scipy.io as scio
from torch.optim.optimizer import Optimizer
from sklearn import preprocessing
import torch.nn.functional as F

def augmentation(feature_seqence,label_seqence,video_time,alpha=0.5):
    augment_data=[]
    augment_label=[]
    flag=0
    if len(feature_seqence)==0:
        return feature_seqence,label_seqence
    for i in range(len(video_time)):
        video_feature=feature_seqence[flag:flag+video_time[i],:]
        video_label=label_seqence[flag:flag+video_time[i],:]
        for j in range(len(video_feature)):
            index=np.random.randint(0,len(video_feature),2)
            weight_sequence=0.5*np.ones(2).reshape((1,2))
            lam=np.random.beta(alpha,alpha)
            weight_sequence[0,0]=lam
            weight_sequence[0,1]=1-lam
            augment_data.append(np.dot(weight_sequence,video_feature[index,:]))
            augment_label.append(video_label[j,:])
        flag+=video_time[i]
    return np.vstack(augment_data),np.vstack(augment_label)

# def get_dataset_aug(test_id, BATCH_SIZE,session,parameter):
#     video = 15
#     alpha = parameter['alpha']
#     # path='F:\\zhourushuang\\transfer_learning\\feature_for_net_session'+str(session)+'_LDS_de'
#     path = 'F:\\Emotion_datasets\\SEED\\feature_for_net_session' + str(session) + '_LDS_de'
#     os.chdir(path)
#     feature_list_source_labeled = []
#     label_list_source_labeled = []
#     feature_list_source_unlabeled = []
#     label_list_source_unlabeled = []
#     feature_list_target = []
#     label_list_target = []
#     feature_list_source_labeled_aug = []
#     label_list_source_labeled_aug = []
#     feature_list_source_unlabeled_aug = []
#     label_list_source_unlabeled_aug = []
#     feature_list_target_aug = []
#     label_list_target_aug = []
#     ## our label:0 negative, label:1 :neural,label:2:positive, seed original label: -1,0,1, our label= seed label+1
#     min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
#     video_time = [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]
#     index = 0
#     for info in os.listdir(path):
#         domain = os.path.abspath(path)
#         info_ = os.path.join(domain, info)
#         if session == 1:
#             feature = scio.loadmat(info_)['dataset_session1']['feature'][0, 0]
#             label = scio.loadmat(info_)['dataset_session1']['label'][0, 0]
#         elif session == 2:
#             feature = scio.loadmat(info_)['dataset_session2']['feature'][0, 0]
#             label = scio.loadmat(info_)['dataset_session2']['label'][0, 0]
#         else:
#             feature = scio.loadmat(info_)['dataset_session3']['feature'][0, 0]
#             label = scio.loadmat(info_)['dataset_session3']['label'][0, 0]
#         feature = min_max_scaler.fit_transform(feature).astype('float32')
#         one_hot_label_mat = np.zeros((len(label), 3))
#         for i in range(len(label)):
#             if label[i] == -1:
#                 one_hot_label = [1, 0, 0]
#                 one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
#                 one_hot_label_mat[i, :] = one_hot_label
#             if label[i] == 0:
#                 one_hot_label = [0, 1, 0]
#                 one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
#                 one_hot_label_mat[i, :] = one_hot_label
#             if label[i] == 1:
#                 one_hot_label = [0, 0, 1]
#                 one_hot_label = np.hstack(one_hot_label).reshape(1, 3)
#                 one_hot_label_mat[i, :] = one_hot_label
#         if index != test_id:
#             ## source labeled data
#             feature_labeled = feature[0:np.cumsum(video_time[0:video])[-1], :]
#             label_labeled = one_hot_label_mat[0:np.cumsum(video_time[0:video])[-1], :]
#             feature_list_source_labeled.append(feature_labeled)
#             label_list_source_labeled.append(label_labeled)
#             ## the origin EEG data for augmentation
#             feature_labeled_origin, label_labeled_origin = np.copy(feature_labeled), np.copy(label_labeled)
#             ## source labeled data and the augdata
#             feature_labeled_aug, label_labeled_aug = augmentation(feature_labeled_origin, label_labeled_origin,
#                                                                   video_time[0:video], alpha)
#             feature_labeled = np.row_stack((feature_labeled, feature_labeled_aug)).astype('float32')
#             label_labeled = np.row_stack((label_labeled, label_labeled_aug)).astype('float32')
#             feature_list_source_labeled_aug.append(feature_labeled)
#             label_list_source_labeled_aug.append(label_labeled)
#             ## source unlabeled data
#             feature_unlabeled = feature[np.cumsum(video_time[0:video])[-1]:len(feature), :]
#             label_unlabeled = one_hot_label_mat[np.cumsum(video_time[0:video])[-1]:len(feature), :]
#             feature_list_source_unlabeled.append(feature_unlabeled)
#             label_list_source_unlabeled.append(label_unlabeled)
#             ## source unlabeled data and aug data
#             ## the origin EEG data for augmentation
#             feature_unlabeled_origin, label_unlabeled_origin = np.copy(feature_unlabeled), np.copy(label_unlabeled)
#             feature_unlabeled_aug, label_unlabeled_aug = augmentation(feature_unlabeled_origin, label_unlabeled_origin,
#                                                                       video_time[video:len(video_time)], alpha)
#             feature_unlabeled = np.row_stack((feature_unlabeled, feature_unlabeled_aug)).astype('float32')
#             label_unlabeled = np.row_stack((label_unlabeled, label_unlabeled_aug)).astype('float32')
#             feature_list_source_unlabeled_aug.append(feature_unlabeled)
#             label_list_source_unlabeled_aug.append(label_unlabeled)
#         else:
#             ## target labeled data
#             feature_list_target.append(feature)
#             label_list_target.append(one_hot_label_mat)
#             label = one_hot_label_mat
#             ## target labeled data and aug data
#             feature_origin, label_origin = np.copy(feature), np.copy(label)
#             feature_aug, label_aug = augmentation(feature_origin, label_origin, video_time, alpha)
#             feature = np.row_stack((feature, feature_aug)).astype('float32')
#             label = np.row_stack((label, label_aug)).astype('float32')
#             feature_list_target_aug.append(feature)
#             label_list_target_aug.append(label)
#         index += 1
#
#     source_feature_labeled, source_label_labeled = np.vstack(feature_list_source_labeled), np.vstack(
#         label_list_source_labeled)
#     source_feature_unlabeled, source_label_unlabeled = np.vstack(feature_list_source_unlabeled), np.vstack(
#         label_list_source_unlabeled)
#     target_feature = feature_list_target[0]
#     target_label = label_list_target[0]
#
#     source_feature_labeled_aug, source_label_labeled_aug = np.vstack(feature_list_source_labeled_aug), np.vstack(
#         label_list_source_labeled_aug)
#     source_feature_unlabeled_aug, source_label_unlabeled_aug = np.vstack(feature_list_source_unlabeled_aug), np.vstack(
#         label_list_source_unlabeled_aug)
#     target_feature_aug = feature_list_target_aug[0]
#     target_label_aug = label_list_target_aug[0]
#
#     target_set = {'feature': target_feature, 'label': target_label, 'feature_aug': target_feature_aug,
#                   'label_aug': target_label_aug}
#     source_set_labeled = {'feature': source_feature_labeled, 'label': source_label_labeled,
#                           'feature_aug': source_feature_labeled_aug, 'label_aug': source_label_labeled_aug}
#
#     torch_dataset_source_labeled = Data.TensorDataset(torch.from_numpy(source_set_labeled['feature_aug']),torch.from_numpy(source_set_labeled['label_aug']))
#     torch_dataset_test = Data.TensorDataset(torch.from_numpy(target_set['feature_aug']),torch.from_numpy(target_set['label_aug']))
#     torch_dataset_test_tt = Data.TensorDataset(torch.from_numpy(target_set['feature']),torch.from_numpy(target_set['label']))
#
#     source_loader = Data.DataLoader(
#              dataset=torch_dataset_source_labeled,
#              batch_size=BATCH_SIZE,
#              shuffle=True,
#              num_workers=0,
#              drop_last=True
#              )
#     target_train_loader = Data.DataLoader(
#              dataset=torch_dataset_test,
#              batch_size=BATCH_SIZE,
#              shuffle=True,
#              num_workers=0,
#              drop_last=True
#              )
#     target_test_loader = Data.DataLoader(
#              dataset=torch_dataset_test_tt,
#              batch_size= target_feature.shape[0],
#              shuffle=True,
#              num_workers=0,
#              # drop_last=True
#              )
#     return source_loader, target_train_loader, target_test_loader

import os
import numpy as np
import scipy.io as scio
from sklearn import preprocessing
import torch
import torch.utils.data as Data


def augmentation(feature, label, video_time, alpha):
    # Implement your augmentation function here
    # This is a placeholder, replace with actual augmentation logic
    aug_feature = feature * alpha
    aug_label = label
    return aug_feature, aug_label


def get_dataset_aug(test_id, BATCH_SIZE, session, parameter):
    video = 15
    alpha = parameter['alpha']
    path = 'F:\\Emotion_datasets\\SEED\\feature_for_net_session' + str(session) + '_LDS_de'
    os.chdir(path)

    feature_list_source_labeled = []
    label_list_source_labeled = []
    feature_list_source_unlabeled = []
    label_list_source_unlabeled = []
    feature_list_target = []
    label_list_target = []
    feature_list_source_labeled_aug = []
    label_list_source_labeled_aug = []
    feature_list_source_unlabeled_aug = []
    label_list_source_unlabeled_aug = []
    feature_list_target_aug = []
    label_list_target_aug = []

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    video_time = [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]

    index = 0
    for info in os.listdir(path):
        domain = os.path.abspath(path)
        info_ = os.path.join(domain, info)

        if session == 1:
            feature = scio.loadmat(info_)['dataset_session1']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset_session1']['label'][0, 0]
        elif session == 2:
            feature = scio.loadmat(info_)['dataset_session2']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset_session2']['label'][0, 0]
        else:
            feature = scio.loadmat(info_)['dataset_session3']['feature'][0, 0]
            label = scio.loadmat(info_)['dataset_session3']['label'][0, 0]

        feature = min_max_scaler.fit_transform(feature).astype('float32')

        one_hot_label_mat = np.zeros((len(label), 3))
        for i in range(len(label)):
            if label[i] == -1:
                one_hot_label_mat[i, :] = [1, 0, 0]
            elif label[i] == 0:
                one_hot_label_mat[i, :] = [0, 1, 0]
            elif label[i] == 1:
                one_hot_label_mat[i, :] = [0, 0, 1]

        if index != test_id:
            # Source labeled data
            for trial in range(video):
                start_idx = sum(video_time[:trial])
                end_idx = sum(video_time[:trial + 1])

                feature_trial = feature[start_idx:end_idx, :]
                label_trial = one_hot_label_mat[start_idx:end_idx, :]

                feature_list_source_labeled.append(feature_trial)
                label_list_source_labeled.append(label_trial)

                feature_trial_origin, label_trial_origin = np.copy(feature_trial), np.copy(label_trial)
                feature_trial_aug, label_trial_aug = augmentation(feature_trial_origin, label_trial_origin,
                                                                  [video_time[trial]], alpha)

                feature_list_source_labeled_aug.append(
                    np.row_stack((feature_trial, feature_trial_aug)).astype('float32'))
                label_list_source_labeled_aug.append(np.row_stack((label_trial, label_trial_aug)).astype('float32'))

            # Source unlabeled data
            for trial in range(video, len(video_time)):
                start_idx = sum(video_time[:trial])
                end_idx = sum(video_time[:trial + 1])

                feature_trial = feature[start_idx:end_idx, :]
                label_trial = one_hot_label_mat[start_idx:end_idx, :]

                feature_list_source_unlabeled.append(feature_trial)
                label_list_source_unlabeled.append(label_trial)

                feature_trial_origin, label_trial_origin = np.copy(feature_trial), np.copy(label_trial)
                feature_trial_aug, label_trial_aug = augmentation(feature_trial_origin, label_trial_origin,
                                                                  [video_time[trial]], alpha)

                feature_list_source_unlabeled_aug.append(
                    np.row_stack((feature_trial, feature_trial_aug)).astype('float32'))
                label_list_source_unlabeled_aug.append(np.row_stack((label_trial, label_trial_aug)).astype('float32'))

        else:
            feature_list_target.append(feature)
            label_list_target.append(one_hot_label_mat)

            feature_origin, label_origin = np.copy(feature), np.copy(label)
            feature_aug, label_aug = augmentation(feature_origin, label_origin, video_time, alpha)

            feature_list_target_aug.append(np.row_stack((feature, feature_aug)).astype('float32'))
            label_list_target_aug.append(np.row_stack((label, label_aug)).astype('float32'))

        index += 1

    source_feature_labeled = np.vstack(feature_list_source_labeled)
    source_label_labeled = np.vstack(label_list_source_labeled)
    # source_feature_unlabeled = np.vstack(feature_list_source_unlabeled)
    # source_label_unlabeled = np.vstack(label_list_source_unlabeled)
    target_feature = feature_list_target[0]
    target_label = label_list_target[0]

    source_feature_labeled_aug = np.vstack(feature_list_source_labeled_aug)
    source_label_labeled_aug = np.vstack(label_list_source_labeled_aug)
    # source_feature_unlabeled_aug = np.vstack(feature_list_source_unlabeled_aug)
    # source_label_unlabeled_aug = np.vstack(label_list_source_unlabeled_aug)
    target_feature_aug = feature_list_target_aug[0]
    target_label_aug = label_list_target_aug[0]

    target_set = {'feature': target_feature, 'label': target_label, 'feature_aug': target_feature_aug,
                  'label_aug': target_label_aug}
    source_set_labeled = {'feature': source_feature_labeled, 'label': source_label_labeled,
                          'feature_aug': source_feature_labeled_aug, 'label_aug': source_label_labeled_aug}

    torch_dataset_source_labeled = Data.TensorDataset(torch.from_numpy(source_set_labeled['feature_aug']),
                                                      torch.from_numpy(source_set_labeled['label_aug']))
    torch_dataset_test = Data.TensorDataset(torch.from_numpy(target_set['feature_aug']),
                                            torch.from_numpy(target_set['label_aug']))
    torch_dataset_test_tt = Data.TensorDataset(torch.from_numpy(target_set['feature']),
                                               torch.from_numpy(target_set['label']))

    source_loader = Data.DataLoader(dataset=torch_dataset_source_labeled, batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=0, drop_last=True)
    target_train_loader = Data.DataLoader(dataset=torch_dataset_test, batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=0, drop_last=True)
    target_test_loader = Data.DataLoader(dataset=torch_dataset_test_tt, batch_size=target_feature.shape[0],
                                         shuffle=True, num_workers=0)

    return source_loader, target_train_loader, target_test_loader
