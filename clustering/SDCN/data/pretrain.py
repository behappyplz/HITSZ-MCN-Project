import numpy as np
import os
import random
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from evaluation import eva
from torch.nn import Linear

import matplotlib.pyplot as plt
import seaborn as sns
import umap
from matplotlib.pyplot import MultipleLocator


#torch.cuda.set_device(3)

device = "cuda:0"

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3) #self.dropout(enc_h3))

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3) #self.dropout(dec_h3))

        return x_bar, z


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pretrain_ae(model, dataset, y):
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=5e-4)
    
    for epoch in range(30):
        # adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)

            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            x = torch.Tensor(dataset.x).to(device).float()
            if epoch == 0:       
                kmeans = KMeans(n_clusters=7, n_init=20).fit(
                            x.data.cpu().numpy()
                        )
                eva(y, kmeans.labels_, "dir_kmeans")
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))           
            kmeans = KMeans(n_clusters=7, n_init=20).fit(z.data.cpu().numpy())
            eva(y, kmeans.labels_, epoch)
        
        if epoch == 20:
            # torch.save(model.state_dict(), 'cora_final_48.pkl')
            # 然后创建 UMAP 对象并进行进一步降维
            from matplotlib.pyplot import MultipleLocator
            subject = ['Rule_Learning', 'Neural_Networks', 'Probabilistic_Methods', 'Genetic_Algorithms', 'Reinforcement_Learning', 'Theory', 'Case_Based']

            umap_model = umap.UMAP(random_state=42)
            embedding = umap_model.fit_transform(z.data.cpu().numpy())

            unique_labels = np.unique([0, 1, 2, 3, 4, 5, 6]) # fixed order

            # 使用 Seaborn 的 cubehelix_palette 调色板生成颜色
            palette = sns.color_palette("hsv", n_colors=len(unique_labels))  # 使用hsv颜色空间为每个类别生成唯一颜色
            color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

            plt.figure(figsize=(12, 8), dpi=300)
            for label in unique_labels:
                indices = np.where(y == label)
                plt.scatter(embedding[indices, 0], embedding[indices, 1], c=[color_map[label]], label=subject[label], alpha=0.6, edgecolor='w', s=30)
            # plt.legend(loc='upper right', ncol=2)
            x_major_locator=MultipleLocator(2)
            y_major_locator=MultipleLocator(2)
            ax=plt.gca()
            #ax为两条坐标轴的实例
            ax.xaxis.set_major_locator(x_major_locator)
            ax.yaxis.set_major_locator(y_major_locator)
            plt.savefig("/home/sunyang/MCN_main/figures/ae_umap_wo.png")
            break
        
'''
model = AE(
        n_enc_1=768,
        n_enc_2=768,
        n_enc_3=2048,
        n_dec_1=2048,
        n_dec_2=768,
        n_dec_3=768,
        n_input=1433,
        n_z=16,).to(device)# n_enc_1必须是一个较小的数！！！ n_z=10效果显著好于16！！
'''
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(3407)

model = AE(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=2000,
        n_dec_1=2000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=1433,
        n_z=10,).to(device)
# Origin: cora.pkl: acc 0.4328 , nmi 0.2147 , ari 0.1688 , f1 0.3806
# to 2^x: acc 0.4298 , nmi 0.2617 , ari 0.1897 , f1 0.3847 + dropout top0.43 final 0.415
# 调参后top3: acc 0.4638 , nmi 0.2469 , ari 0.1975 , f1 0.4532
# 调参后pretrain top1: 20 :acc 0.4889 , nmi 0.2654 , ari 0.2177 , f1 0.4743
x = np.loadtxt('cora.txt', dtype=float)
y = np.loadtxt('cora_label.txt', dtype=int)

dataset = LoadDataset(x)
pretrain_ae(model, dataset, y)

print(model)   