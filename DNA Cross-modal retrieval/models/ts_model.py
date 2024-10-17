import pandas as pd
import torch
import torch.nn as nn
import numpy as np


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # 定义模型的层次结构
        self.p_model = nn.Sequential(nn.Conv2d(2, 2, (1,1), 1),
                                     nn.Flatten(),
            nn.Linear(100, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 10)
                                     )
        self.fc1 = nn.Linear(1024, 128)  # 全连接层，输入大小为784，输出大小为128
        self.relu = nn.ReLU()           # ReLU激活函数
        self.fc2 = nn.Linear(128, 10)   # 全连接层，输入大小为128，输出大小为10（分类数）

    def forward(self, x):
        # 定义前向传播逻辑
            # 第二层全连接层
        return self.p_model(x)
    def seq_pairs_to_onehots(self, seq_pairs):
        onehot_pairs = np.stack([
            self.seqs_to_onehots(seq_pairs.target_features.values),

            self.seqs_to_onehots(seq_pairs.query_features.values)
        ], axis = 1)
        return onehot_pairs

    def seqs_to_onehots(self,seqs):
        seq_array = np.array(list(map(list, seqs)))
        return np.array([(seq_array == b).T for b in self.bases]).T.astype(float)
device = torch.device('cuda:0')
predictor = SimpleModel().to(device)
batch_train_data = torch.randn(2,10,10).to(device)
pre = predictor(batch_train_data)
# class Encoder_dataset(Dataset):
#     def __init__(self, h5_path):
#         self.train_data = pd.read_hdf(h5_path,key='train_data').values
#     def __len__(self):
#         return len(self.train_data)
#     def __getitem__(self, item):
#         img_txt_pair = torch.tensor(self.train_data[item][:-1])
#         # img_txt_pair = img_txt_pair.reshape((2,512))
#         label = torch.tensor(self.train_data[item][-1])
#         return img_txt_pair,label

# device = torch.device('cuda')
# a = SimpleModel().to(device)
# b = Encoder_dataset('/home/cao/桌面/new_similarity_search/simi/Dataset/train_data/train_data.h5')
# c = DataLoader(b,batch_size=100,shuffle=True)
# for i,j in tqdm(c):
#     i = i.to(device)
#     a(i)
# chunks = pd.read_hdf('/home/cao/桌面/new_similarity_search/simi/Dataset/train_data/train_data.h5', chunksize=100000)
# count = 0
# for chunk in chunks:
#     print(count)
#     count+=1
# print(count)
# a = h5py.File('/home/cao/桌面/new_similarity_search/simi/Dataset/train_data/train_data.h5','r')
# print(a['train_data'])
# h5_path = '/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/val_data.h5'
# train_data = tables.open_file(h5_path, mode='r')
# print(train_data.root['train_data'][0])

txt_cat = torch.randn(10,512)
img_cat = torch.randn(10,512)
train_data_simi = torch.stack([img_cat,txt_cat],dim=1)
print(train_data_simi.shape)