# Copyright 2020 Noisy-FewShot-Learning Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import faiss
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pygcn.utils import *
from pygcn.layers import GraphConvolution
import pdb


class GCNcleaner(nn.Module):
    def __init__(self, input_dim, hidden_dim = 16, dropout = 0.5):
        super(GCNcleaner, self).__init__()
        self.gc_input = GraphConvolution(input_dim, hidden_dim)
        self.gc_output = GraphConvolution(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc_input(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc_output(x, adj)
        return x


def run_cleaner(cfg, clean_features, clean_labels, noisy_features, noisy_labels, faiss_gpu_id = 0):
    label_set = np.unique(clean_labels)
    weights = np.zeros((noisy_labels.shape[0],))

    for label in label_set: # loop over all classes

        clean_idx = np.where(clean_labels==label)[0]
        noisy_idx = np.where(noisy_labels==label)[0]
        if noisy_idx.size == 0:  continue  # 0 noisy examples
        clean_features_ = torch.Tensor(clean_features[clean_idx,:])
        noisy_features_ = torch.Tensor(noisy_features[noisy_idx,:])
        cur_features = torch.cat((clean_features_,noisy_features_)).cuda()
        pos_idx = np.arange(clean_features_.shape[0])
        neg_idx = np.arange(noisy_features_.shape[0]) + clean_features_.shape[0]

        # graph creation
        affinitymat = features2affinitymax(cur_features.data.cpu().numpy(), k = cfg['k'], gpu_id = faiss_gpu_id)
        affinitymat = affinitymat.minimum(affinitymat.T)
        affinitymat = graph_normalize(affinitymat+ sp.eye(affinitymat.shape[0]))
        affinitymat = sparse_mx_to_torch_sparse_tensor(affinitymat).cuda()

        # GCN training
        model = train_gcn(cur_features, affinitymat, pos_idx, neg_idx, cfg['gcnlambda'])

        # run the GCN model in eval mode to get the predicted relevance weights
        model.eval()
        output = torch.sigmoid(model(cur_features, affinitymat))
        cur_weights = output[neg_idx]
        cur_weights = cur_weights.cpu().detach().numpy().squeeze()
        weights[noisy_idx] = cur_weights

    return weights


def train_gcn(features, affinitymat, pos_idx, neg_idx, gcn_lambda):
    lr = 0.1
    gcniter = 100
    eps=1e-6

    model = GCNcleaner(input_dim=features.shape[1])
    model = model.cuda()
    model.train()

    params_set = [dict(params=model.parameters())]
    optimizer = optim.Adam(params_set,lr=lr, weight_decay=5e-4)
    for epoch in range(gcniter): 
        adjust_learning_rate(optimizer, epoch, lr)

        optimizer.zero_grad()        
        output = torch.sigmoid(model(features, affinitymat)) 
        loss_train = -(output.squeeze()[pos_idx]+eps).log().mean()  # loss for clean
        loss_train += -gcn_lambda*(1-output[neg_idx]+eps).log().mean()   # loss for noisy, treated as negative
        loss_train.backward()
        optimizer.step()
    
    return model


def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def features2affinitymax(features, k = 50, gpu_id = 0):
    knn, sim = knn_faiss(features, features, k = k+1, gpu_id = gpu_id)
    aff = knn2affinitymat(knn[:,1:], sim[:,1:]) # skip self-matches
    return aff


def knn2affinitymat(knn, sim):
    N, k = knn.shape[0], knn.shape[1]
    
    row_idx_rep = np.tile(np.arange(N),(k,1)).T
    sim_flatten = sim.flatten('F')
    row_flatten = row_idx_rep.flatten('F')
    knn_flatten = knn.flatten('F')

    # # Handle the cases where FAISS returns -1 as knn indices - FIX
    # invalid_idx = np.where(knn_flatten<0)[0]
    # if len(invalid_idx):
    #     sim_flatten = np.delete(sim_flatten, invalid_idx)
    #     row_flatten = np.delete(row_flatten, invalid_idx)
    #     knn_flatten = np.delete(knn_flatten, invalid_idx)

    W = sp.csr_matrix((sim_flatten, (row_flatten, knn_flatten)), shape=(N, N))
    return W


def graph_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def knn_faiss(X, Q, k, gpu_id = 0):
    D = X.shape[1]

    # CPU search if gpu_id = -1. GPU search otherwise.
    if gpu_id == -1:
        index = faiss.IndexFlatIP(D)
    else:
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = gpu_id
        index = faiss.GpuIndexFlatIP(res, D, flat_config)
    index.add(X)                  
    sim, knn = index.search(Q, min(k,X.shape[0])) 
    index.reset()
    del index

    return knn, sim