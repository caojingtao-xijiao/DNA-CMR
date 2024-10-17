import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from encoder import Encoder
from utils.Dataset_set import Encoder_dataset
from simulator import simulator
global seq_pairs
def encoder_test(feature_data_path,param_path):
    #实例化模型并加载参数
    encoder = Encoder(512,80)
    # encoder_text = Encoder_text(512,80)
    encoder_model_params = torch.load(param_path)['encoder_model_params']
    # # # encoder_text_model_params = torch.load(param_path)['encoder_text__model_params']
    encoder.load_state_dict(encoder_model_params)
    # encoder_text.load_state_dict(encoder_text_model_params)
    #读取数据
    dataset = Encoder_dataset(feature_data_path,isTrain=True)
    datloader = DataLoader(dataset,batch_size=50,num_workers=4)

    ids,features,labels = next(iter(datloader))
    seq_0 = encoder.feature_to_seq(features[:, 0])
    seq_1 = encoder.feature_to_seq(features[:, 1])

    seq_pairs = pd.DataFrame({
        "target_features": seq_0,
        "query_features": seq_1,
    })

    yields = np.array(simulator(seq_pairs))
    # lt = []
    # i = 0
    # for ids,features,labels in datloader:
    #     labels = np.array(labels)
    #     lt.append(labels.sum())
    #     i+=1
    #     if i>100:
    #         break
    # lt = np.array(lt)
    # lt_m = lt.mean()
    # image_ids = ids[:,0].reshape(-1)
    # text_ids = ids[:,1].reshape(-1)
    # for i in range(10):
    #     image = next(dataset.open_image(image_ids))
    #     text = next(dataset.open_text(text_ids))
    #
    #     print(text)
    #     print(labels[i])
    #     print(yields[i])
    #     if i > 10:
    #         break

    ids = np.array(ids)
    labels = np.array(labels)
    print(np.mean(np.abs(labels - yields)))
    print(yields)

    # yields = np.around(yields)
    print((yields==labels).sum())
    res = np.insert(ids,3,labels,axis=1)
    res = np.insert(res,4,yields,axis=1)
    res = np.array(res,dtype=np.float32)

    return res,yields,seq_pairs
a = encoder_test('/home/cao_ubuntu/桌面/similarity_search/new_similarity_search/simi/dataset/val_data/feature.h5',
                 '../model_save/encoder_train/model_params.pth')
#'../model_save/encoder_train/model_params.pth'
#'/home/cao_ubuntu/桌面/encoder_params/encoder_train_unnorm/model_params.pth'

print(a[0])




