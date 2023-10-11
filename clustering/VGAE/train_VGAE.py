from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from sklearn.cluster import KMeans
from evaluation import eva
from model import GCNModelVAE
from optimizer import loss_function
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score

import matplotlib.pyplot as plt
import seaborn as sns
import umap
from matplotlib.pyplot import MultipleLocator

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=108, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
parser.add_argument('--n_clusters', type=int, default=7)

args = parser.parse_args()


def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    adj, features, y = load_data(args.dataset_str)
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, adj_all = mask_test_edges(adj)
    # print(adj_orig)
    # adj = adj_train
    adj = adj_all

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    hidden_emb = None
    for epoch in range(args.epochs):
        if epoch % 5 == 0:
        # update_interval
            with torch.no_grad():
                model.eval()
                _, mu, _ = model(features, adj_norm)
                # print(mu.shape)  
                # y_pred = mu.cpu().numpy().argmax(1)            
                kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
                y_pred = kmeans.fit_predict(mu.data.cpu().numpy())
                eva(y, y_pred, 'vgae')
                if epoch == 60:
                    subject = ['Rule_Learning', 'Neural_Networks', 'Probabilistic_Methods', 'Genetic_Algorithms', 'Reinforcement_Learning', 'Theory', 'Case_Based']

                    umap_model = umap.UMAP(random_state=42)
                    embedding = umap_model.fit_transform(mu.data.cpu().numpy())

                    unique_labels = np.unique([0, 1, 2, 3, 4, 5, 6]) # fixed order

                    # 使用 Seaborn 的 cubehelix_palette 调色板生成颜色
                    palette = sns.color_palette("hsv", n_colors=len(unique_labels))  # 使用hsv颜色空间为每个类别生成唯一颜色
                    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

                    plt.figure(figsize=(12, 8), dpi=300)
                    for label in unique_labels:
                        indices = np.where(y == label)
                        plt.scatter(embedding[indices, 0], embedding[indices, 1], c=[color_map[label]], label=subject[label], alpha=0.6, edgecolor='w', s=30)
                    # plt.legend(loc='upper right', ncol=2)
                    # 设置 axis刻度
                    x_major_locator=MultipleLocator(2)
                    y_major_locator=MultipleLocator(2)
                    ax=plt.gca()
                    ax.xaxis.set_major_locator(x_major_locator)
                    ax.yaxis.set_major_locator(y_major_locator)
                    plt.savefig("/home/sunyang/MCN_main/figures/vgae_umap_wo.png")
        t = time.time()
        model.train()
        optimizer.zero_grad()
        recovered, mu, logvar = model(features, adj_norm)
        loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.numpy()
        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t)
              )

    print("Optimization Finished!")

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))


if __name__ == '__main__':
    def seed_everything(seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    seed_everything(args.seed)
    gae_for(args)
