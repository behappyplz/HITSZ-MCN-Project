import numpy as np
import torch
from sklearn.preprocessing import normalize
import networkx as nx
from torch_geometric.data import Data


def get_dataset(dataset):
    # datasets = Planetoid('./dataset', dataset)
    # dataset: x, edge_index
    # TODO ours
    # 读取边缘信息
    edges = np.loadtxt(f'./data/{dataset}_graph.txt', dtype=int)
    # 去重并获取egde-list
    s = []
    for e in edges:
        if [e[0],e[1]] in s:
            # print(e)
            continue
        else:
            s.append([e[0],e[1]])
    edge_index = torch.tensor(s, dtype=torch.long).t().contiguous()

    # 读取节点特征
    features = np.loadtxt(f'./data/{dataset}.txt')
    x = torch.tensor(features, dtype=torch.float)

    # 读取标签
    labels = np.loadtxt(f'./data/{dataset}_label.txt', dtype=int)
    y = torch.tensor(labels, dtype=torch.long)

    # Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
    datasets = Data(x=x, edge_index=edge_index, y=y)
    return [datasets]

def data_preprocessing(dataset):
    dataset.adj = torch.sparse_coo_tensor(
        dataset.edge_index, torch.ones(dataset.edge_index.shape[1]), torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
    ).to_dense()
    dataset.adj_label = dataset.adj

    dataset.adj += torch.eye(dataset.x.shape[0])
    dataset.adj = normalize(dataset.adj, norm="l1")
    dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)

    return dataset

def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    # t_order
    t=2
    # 计算一阶/二阶转移概率矩阵
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)


