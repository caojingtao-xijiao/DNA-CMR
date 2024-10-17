import pandas as pd
import torch
import json
import random
import tables
import torch
# -*- coding:utf-8 -*-

import h5py
import numpy as np



feature_file = '/home/cao/桌面/new_similarity_search/simi/Dataset/train_data/train_feature.h5'

img_feature_df = torch.tensor(pd.read_hdf(feature_file,key='image').values)
img_norm = img_feature_df/img_feature_df.norm(dim=-1,keepdim=True)
txt_feature_df = pd.read_hdf(feature_file,key='text').values
b = []
for i in txt_feature_df[:100]:
    i = torch.tensor(i)
    i_norm = i/i.norm(dim=-1,keepdim=True)
    cos_simi = 100*i_norm@img_norm.t()
    a = torch.sum(cos_simi>28,dim=-1)
    print(a)
    b.append(a.item())
print(sum(b)/len(b))



