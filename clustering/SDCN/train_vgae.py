from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva
import scipy.sparse as sp
from collections import Counter
from models.vgae import VGAE, GAE

# torch.cuda.set_device(1)


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

def train_sdcn(dataset):

    # KNN Graph
    adj = load_graph(args.name, args.k).to(device)
    model = VGAE(args.n_input, 32, args.n_z, adj).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)


    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

  
    

    for epoch in range(150):
        if epoch % 5 == 0:
        # update_interval
            with torch.no_grad():
                model.eval()
                encoding = model.encode(data).clone()              
                kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
                y_pred = kmeans.fit_predict(encoding.data.cpu().numpy())
                eva(y, y_pred, 'vgae')
                print("\n")
        model.train()
        loss = model(data)
        print(np.mean(loss.cpu().detach().numpy()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    print("VGAE!\n")
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='cora')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=7, type=int)
    parser.add_argument('--n_z', default=16, type=int)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda:2" if args.cuda else "cpu")

    args.pretrain_path = 'data/{}.pkl'.format(args.name)
    dataset = load_data(args.name)

    if args.name == 'cora':
        args.k = None
        args.n_clusters = 7
        args.n_input = 1433


    print(args)
    train_sdcn(dataset)
