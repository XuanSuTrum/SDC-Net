# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:12:17 2023

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 20:32:41 2023

@author: Administrator
"""
# 没有多尺度 有cmmd
import torch.nn as nn
import mmd1
import backbone
import torch
import cmmd_1
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

GAMMA = 10 ^ 3

class Transfer_Net(nn.Module):
    def __init__(self, upper_threshold, lower_threshold, num_class, max_iter, base_net='CFE', transfer_loss='mmd', use_bottleneck=False, width=32,
                 confidence_threshold=0.65,):
        super(Transfer_Net, self).__init__()
        self.base_network = backbone.network_dict[base_net]()
        self.P = torch.randn(num_class, 64)
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        self.confidence_threshold = confidence_threshold
        classifier_layer_list = [nn.Linear(64, width), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(width, 3)]
        self.classifier = nn.Sequential(*classifier_layer_list)
        self.softmax = nn.Softmax(dim=1)

        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.max_iter = max_iter
        self.threshold = 0.9

    def forward(self, e, source, target, s_label):

        source_data = self.base_network(source)
        target_data = self.base_network(target)

        source_clf = self.classifier(source_data)
        target_clf = self.classifier(target_data)
        # 源域配对
        S_s_label = self.softmax(source_clf)
        sim_matrix = self.get_cos_similarity_distance(S_s_label)
        estimated_sim_truth = self.get_cos_similarity_distance(s_label)
        # 目标域配对
        S_t_label = self.softmax(target_clf)
        sim_matrix_target = self.get_cos_similarity_distance(S_t_label)
        estimated_sim_truth_target = self.get_cos_similarity_by_threshold(sim_matrix_target)

        # t_label_ = target_clf.data.max(1)[1]
        # source_clf = self.softmax(source_clf)
        target_clf = self.softmax(target_clf)
        target_probabilities = torch.max(target_clf, dim=1)[0]

        if 0 <= e < 10:
             confidence_threshold = 0
        elif 10 <= e < 40:
             confidence_threshold = 0.5
        elif 40 <= e <=85:
             confidence_threshold = 0.75
        else:
             # 如果 e 超出了上述范围，你可能需要考虑如何处理
             confidence_threshold = 1
        # # #
        # if 0 <= e < 10:
        #     confidence_threshold = 1
        # elif 10 <= e < 40:
        #     confidence_threshold = 0.75
        # elif 40 <= e <=85:
        #     confidence_threshold = 0.4
        # else:
        #     # 如果 e 超出了上述范围，你可能需要考虑如何处理
        #     confidence_threshold = 0

        # 根据概率分布计算置信度
        confidence_mask = target_probabilities > confidence_threshold

        # 仅保留置信度高于阈值的目标域数据
        confident_target_data = target_data[confidence_mask]
        confident_target_predictions = torch.argmax(target_clf[confidence_mask], dim=1)
        target_data_bao = target[confidence_mask]
        pseudo_labels = confident_target_predictions

        # t_label = target_clf.data.max(1)[1]
        s_label = torch.argmax(s_label, dim=1)

        transfer_loss = self.adapt_loss(source_data, target_data, self.transfer_loss)

        # cmmd_loss = torch.Tensor([0])
        # cmmd_loss = cmmd_loss.cpu()
        cmmd_loss = cmmd_1.cmmd(source_data, confident_target_data, s_label, pseudo_labels)
        # print('target_probabilities:',target_probabilities)
        return source_clf, transfer_loss, cmmd_loss, sim_matrix, estimated_sim_truth,sim_matrix_target, estimated_sim_truth_target

    def predict(self, x):
        features = self.base_network(x)
        clf = self.classifier(features)
        return clf

    def compute_indicator(self, cos_dist_matrix):
        device = cos_dist_matrix.device
        dtype = cos_dist_matrix.dtype
        selected = torch.tensor(1, dtype=dtype, device=device)
        not_selected = torch.tensor(0, dtype=dtype, device=device)
        w2 = torch.where(cos_dist_matrix < self.lower_threshold, selected, not_selected)
        w1 = torch.where(cos_dist_matrix > self.upper_threshold, selected, not_selected)
        w = w1 + w2
        nb_selected = torch.sum(w)
        return w, nb_selected

    def get_cos_similarity_distance(self, features):
        """Get distance in cosine similarity
        :param features: features of samples, (batch_size, num_clusters)
        :return: distance matrix between features, (batch_size, batch_size)
        """
        # (batch_size, num_clusters)
        features_norm = torch.norm(features, dim=1, keepdim=True)
        # (batch_size, num_clusters)
        features = features / features_norm
        # (batch_size, batch_size)
        cos_dist_matrix = torch.mm(features, features.transpose(0, 1))
        return cos_dist_matrix

    def get_cos_similarity_by_threshold(self, cos_dist_matrix):
        """Get similarity by threshold
        :param cos_dist_matrix: cosine distance in matrix,
        (batch_size, batch_size)
        :param threshold: threshold, scalar
        :return: distance matrix between features, (batch_size, batch_size)
        """
        device = cos_dist_matrix.device
        dtype = cos_dist_matrix.dtype
        similar = torch.tensor(1, dtype=dtype, device=device)
        dissimilar = torch.tensor(0, dtype=dtype, device=device)
        sim_matrix = torch.where(cos_dist_matrix > self.threshold, similar, dissimilar)
        return sim_matrix

    def update_threshold(self, epoch: int):
        """Update threshold
        :param threshold: scalar
        :param epoch: scalar
        :return: new_threshold: scalar
        """
        n_epochs = self.max_iter
        diff = self.upper_threshold - self.lower_threshold
        eta = diff / n_epochs
        #        eta=self.diff/ n_epochs /2
        # First epoch doesn't update threshold
        if epoch != 0:
            self.upper_threshold = self.upper_threshold - eta
            self.lower_threshold = self.lower_threshold + eta
        else:
            self.upper_threshold = self.upper_threshold
            self.lower_threshold = self.lower_threshold
        self.threshold = (self.upper_threshold + self.lower_threshold) / 2

    #        print(">>> new threshold is {}".format(new_threshold), flush=True)

    def adapt_loss(self, X, Y, adapt_loss):
        #loss = mmd.MMD_loss(X, Y)
        loss = mmd1.mmd_rbf_accelerate(X, Y)
        # loss,_,_ = mk_mmd.mix_rbf_mmd2_and_ratio(X, Y, [GAMMA])
        """Compute adaptation loss, currently we support mmd and coral

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix
            adapt_loss {string} -- loss type, 'mmd' or 'coral'. You can add your own loss

        Returns:
            [tensor] -- adaptation loss tensor
        """
        # if adapt_loss == 'mmd':
        #     mmd_loss = mmd.MMD_loss()
        #     loss = mmd_loss(X, Y)
        # elif adapt_loss == 'coral':
        #     loss = CORAL(X, Y)
        # else:
        #     loss = 0
        return loss

    # def visualization(self,target,target_labels,tsne=1):
    #     feature_target_f=self.base_network(target)
    #     target_feature=self.classifier(feature_target_f)
    #     #       target_feature=torch.nn.functional.softmax(target_feature, dim=1)
    #     target_feature=target_feature.cpu().detach().numpy()
    #     feature_target_f=feature_target_f.cpu().detach().numpy()
    #     target_labels=np.argmax(target_labels.cpu().detach().numpy(),axis=1)
    #     colors1 = '#00CED1' #点的颜色
    #     colors2 = '#DC143C'
    #     colors3 = '#008000'
    #     area = np.pi * 0.5**2  # 点面积
    #     if tsne==0:
    #         x0=target_feature[np.where(target_labels==0)[0]]
    #         x1=target_feature[np.where(target_labels==1)[0]]
    #         x2=target_feature[np.where(target_labels==2)[0]]
    #     # 画散点图
    #         fig = plt.figure()
    #         ax = Axes3D(fig)
    #         ax.scatter(x0[:,0],x0[:,1],x0[:,2], s=area, c=colors1, alpha=0.4)
    #         ax.scatter(x1[:,0],x1[:,1],x1[:,2], s=area, c=colors2, alpha=0.4)
    #         ax.scatter(x2[:,0],x2[:,1],x2[:,2], s=area, c=colors3, alpha=0.4)
    #         plt.show()
    #     else:
    #         target_feature = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000).fit_transform(feature_target_f.astype('float32'))
    #         x0=target_feature[np.where(target_labels==0)[0]]
    #         x1=target_feature[np.where(target_labels==1)[0]]
    #         x2=target_feature[np.where(target_labels==2)[0]]
    #         plt.scatter(x0[:,0],x0[:,1], s=20, c='none',  marker='o',edgecolors = colors1,alpha=0.5,label='Source Class 0')
    #         plt.scatter(x1[:,0],x1[:,1], s=20, c='none',  marker='*',edgecolors = colors2,alpha=0.5,label='Source Class 1')
    #         plt.scatter(x2[:,0],x2[:,1], s=20, c='none',  marker='s',edgecolors = colors3,alpha=0.5,label='Source Class 2')
    #         plt.show()

    def visualization(self, source, source_labels, target, target_labels, tsne=1):

        # 提取源域和目标域的特征
        feature_source_f = self.base_network(source)
        feature_target_f = self.base_network(target)
    
        # 将结果转为numpy数组
        feature_source_f = feature_source_f.cpu().detach().numpy()
        source_labels = np.argmax(source_labels.cpu().detach().numpy(), axis=1)
    
        feature_target_f = feature_target_f.cpu().detach().numpy()
        target_labels = np.argmax(target_labels.cpu().detach().numpy(), axis=1)
    
        colors1 = '#00CED1'  # 点的颜色蓝绿色
        colors2 = '#DC143C'  # 深红色
        colors3 = '#008000'  # 绿色
    
        area = 0.5 ** 2  # 点面积
        area_dict = {0: 1, 1: 2, 2: 3}
        if tsne == 0:
            # 绘制源域的散点图
            x0_source = feature_source_f[np.where(source_labels == 0)[0]]
            x1_source = feature_source_f[np.where(source_labels == 1)[0]]
            x2_source = feature_source_f[np.where(source_labels == 2)[0]]
    
            # 绘制目标域的散点图
            x0_target = feature_target_f[np.where(target_labels == 0)[0]]
            x1_target = feature_target_f[np.where(target_labels == 1)[0]]
            x2_target = feature_target_f[np.where(target_labels == 2)[0]]
    
            # 画散点图
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(x0_source[:, 0], x0_source[:, 1], x0_source[:, 2], s=area, c=colors1, alpha=0.1,
                        label='Source Class 0')
            ax.scatter(x1_source[:, 0], x1_source[:, 1], x1_source[:, 2], s=area, c=colors2, alpha=0.1,
                        label='Source Class 1')
            ax.scatter(x2_source[:, 0], x2_source[:, 1], x2_source[:, 2], s=area, c=colors3, alpha=0.4,
                        label='Source Class 2')
    
            ax.scatter(x0_target[:, 0], x0_target[:, 1], x0_target[:, 2], s=area, c=colors1, alpha=0.4, marker='^',
                        label='Target Class 0')
            ax.scatter(x1_target[:, 0], x1_target[:, 1], x1_target[:, 2], s=area, c=colors2, alpha=0.4, marker='^',
                        label='Target Class 1')
            ax.scatter(x2_target[:, 0], x2_target[:, 1], x2_target[:, 2], s=area, c=colors3, alpha=0.4, marker='^',
                        label='Target Class 2')
    
            ax.legend()
            plt.show()
    
        else:
            # 使用t-SNE进行降维
            source_feature_tsne = TSNE(perplexity=10, n_components=2, init='pca', n_iter=3000).fit_transform(
                feature_source_f.astype('float32'))
            target_feature_tsne = TSNE(perplexity=10, n_components=2, init='pca', n_iter=3000).fit_transform(
                feature_target_f.astype('float32'))
    
            # 绘制源域的散点图
            x0_source = source_feature_tsne[np.where(source_labels == 0)[0]]
            x1_source = source_feature_tsne[np.where(source_labels == 1)[0]]
            x2_source = source_feature_tsne[np.where(source_labels == 2)[0]]
    
            # 绘制目标域的散点图
            x0_target = target_feature_tsne[np.where(target_labels == 0)[0]]
            x1_target = target_feature_tsne[np.where(target_labels == 1)[0]]
            x2_target = target_feature_tsne[np.where(target_labels == 2)[0]]
    
            plt.scatter(x0_source[:, 0], x0_source[:, 1], c='none', marker='o', edgecolors=colors1, alpha=0.2,
                        label='Source Class 0')
            plt.scatter(x1_source[:, 0], x1_source[:, 1], c='none', marker='o', edgecolors=colors2, alpha=0.2,
                        label='Source Class 1')
            plt.scatter(x2_source[:, 0], x2_source[:, 1], c='none', marker='o', edgecolors=colors3, alpha=0.2,
                        label='Source Class 2')
    
            plt.scatter(x0_target[:, 0], x0_target[:, 1], c='none', marker='*', edgecolors=colors1, alpha=0.2,
                        label='Target Class 0')
            plt.scatter(x1_target[:, 0], x1_target[:, 1], c='none', marker='*', edgecolors=colors2, alpha=0.2,
                        label='Target Class 1')
            plt.scatter(x2_target[:, 0], x2_target[:, 1], c='none', marker='*', edgecolors=colors3, alpha=0.2,
                        label='Target Class 2')
    
            plt.legend().set_visible(False)
            plt.xticks([])
            plt.yticks([])
            #plt.savefig('E:\博士课题进展\paper1\Overleaf Projects (1 items)\Figure{:}.png'.format(e),dpi=300)
            plt.show()

    def visualization_pseudo(self, e, source, target, tsne=1):
        source_labels = []
        target_labels = []
        # 提取源域和目标域的特征
        feature_source_f = self.base_network(source)
        feature_target_f = self.base_network(target)

        # 提取源域和目标域的分类器输出
        source_feature = self.classifier(feature_source_f)
        source_max_indices = torch.argmax(source_feature, dim=1)
        num_classes = source_feature.shape[1]
        source_one_hot_encoded = torch.eye(num_classes)[source_max_indices.cpu()]
        source_labels.append(source_one_hot_encoded)
        source_labels = torch.cat(source_labels, dim=0)

        target_feature = self.classifier(feature_target_f)
        target_max_indices = torch.argmax(target_feature, dim=1)
        num_classes = target_feature.shape[1]
        target_one_hot_encoded = torch.eye(num_classes)[target_max_indices.cpu()]
        target_labels.append(target_one_hot_encoded)
        target_labels = torch.cat(target_labels, dim=0)
        # 将结果转为numpy数组
        source_feature = source_feature.cpu().detach().numpy()
        feature_source_f = feature_source_f.cpu().detach().numpy()
        source_labels = np.argmax(source_labels.cpu().detach().numpy(), axis=1)

        target_feature = target_feature.cpu().detach().numpy()
        feature_target_f = feature_target_f.cpu().detach().numpy()
        target_labels = np.argmax(target_labels.cpu().detach().numpy(), axis=1)

        colors1 = '#00CED1'  # 点的颜色蓝绿色
        colors2 = '#DC143C'  # 深红色
        colors3 = '#008000'  # 绿色

        area = 0.5 ** 2  # 点面积
        area_dict = {0: 1, 1: 2, 2: 3}
        if tsne == 0:
            # 绘制源域的散点图
            x0_source = feature_source_f[np.where(source_labels == 0)[0]]
            x1_source = feature_source_f[np.where(source_labels == 1)[0]]
            x2_source = feature_source_f[np.where(source_labels == 2)[0]]

            # 绘制目标域的散点图
            x0_target = feature_target_f[np.where(target_labels == 0)[0]]
            x1_target = feature_target_f[np.where(target_labels == 1)[0]]
            x2_target = feature_target_f[np.where(target_labels == 2)[0]]

            # 画散点图
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(x0_source[:, 0], x0_source[:, 1], x0_source[:, 2], s=area, c=colors1, alpha=0.1,
                        label='Source Class 0')
            ax.scatter(x1_source[:, 0], x1_source[:, 1], x1_source[:, 2], s=area, c=colors2, alpha=0.1,
                        label='Source Class 1')
            ax.scatter(x2_source[:, 0], x2_source[:, 1], x2_source[:, 2], s=area, c=colors3, alpha=0.4,
                        label='Source Class 2')

            ax.scatter(x0_target[:, 0], x0_target[:, 1], x0_target[:, 2], s=area, c=colors1, alpha=0.4, marker='^',
                        label='Target Class 0')
            ax.scatter(x1_target[:, 0], x1_target[:, 1], x1_target[:, 2], s=area, c=colors2, alpha=0.4, marker='^',
                        label='Target Class 1')
            ax.scatter(x2_target[:, 0], x2_target[:, 1], x2_target[:, 2], s=area, c=colors3, alpha=0.4, marker='^',
                        label='Target Class 2')

            ax.legend()
            plt.show()

        else:
            # 使用t-SNE进行降维
            source_feature_tsne = TSNE(perplexity=10, n_components=2, init='pca', n_iter=3000).fit_transform(
                feature_source_f.astype('float32'))
            target_feature_tsne = TSNE(perplexity=10, n_components=2, init='pca', n_iter=3000).fit_transform(
                feature_target_f.astype('float32'))

            # 绘制源域的散点图
            x0_source = source_feature_tsne[np.where(source_labels == 0)[0]]
            x1_source = source_feature_tsne[np.where(source_labels == 1)[0]]
            x2_source = source_feature_tsne[np.where(source_labels == 2)[0]]

            # 绘制目标域的散点图
            x0_target = target_feature_tsne[np.where(target_labels == 0)[0]]
            x1_target = target_feature_tsne[np.where(target_labels == 1)[0]]
            x2_target = target_feature_tsne[np.where(target_labels == 2)[0]]

            plt.scatter(x0_source[:, 0], x0_source[:, 1], c='none', marker='o', edgecolors=colors1, alpha=0.1,
                        label='Source Class 0')
            plt.scatter(x1_source[:, 0], x1_source[:, 1], c='none', marker='o', edgecolors=colors2, alpha=0.1,
                        label='Source Class 1')
            plt.scatter(x2_source[:, 0], x2_source[:, 1], c='none', marker='o', edgecolors=colors3, alpha=0.1,
                        label='Source Class 2')

            plt.scatter(x0_target[:, 0], x0_target[:, 1], c='none', marker='*', edgecolors=colors1, alpha=0.5,
                        label='Target Class 0')
            plt.scatter(x1_target[:, 0], x1_target[:, 1], c='none', marker='*', edgecolors=colors2, alpha=0.5,
                        label='Target Class 1')
            plt.scatter(x2_target[:, 0], x2_target[:, 1], c='none', marker='*', edgecolors=colors3, alpha=0.5,
                        label='Target Class 2')

            plt.legend().set_visible(False)
            plt.xticks([])
            plt.yticks([])
            #plt.savefig('E:\博士课题进展\paper1\Overleaf Projects (1 items)\Figure{:}.png'.format(e),dpi=300)
            plt.show()