import sys
sys.path.append('/home/cao/桌面/new_similarity_search/simi')
from argparse import ArgumentParser
import pandas as pd
import torch
from tqdm import tqdm
import tables
from utils.Dataset_set import Txt_feature_dataste,Img_feature_dataste
from torch.utils.data import DataLoader

#图像与文本配对
def image_txt_pair(feture_file,
                   train_data_path,
                   thre=26,
                   per_img_txt=500,
                   img_batch_size=100,
                   txt_batch_size=200000,
                   is_shuffle=True,
                   device='cuda'
                   ):
    #图像数据集
    img_dataset = Img_feature_dataste(feture_file)
    img_dataloader = DataLoader(img_dataset,batch_size=img_batch_size,num_workers=4)
    #文本数据集
    txt_feature_pd = pd.read_hdf(feture_file, key='text')
    #预计计算总的数据量
    data_count = len(img_dataset) * per_img_txt * 2
    #新建H5文件
    pair_train_data = tables.open_file(train_data_path,'w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_carray = pair_train_data.create_carray(pair_train_data.root,
                                                'data',
                                                tables.Atom.from_dtype(txt_feature_pd.values.dtype),
                                                shape=(data_count,2,512),
                                                filters=filters
                                                )
    label_carray = pair_train_data.create_carray(pair_train_data.root,
                                                'label',
                                                 tables.Atom.from_dtype(txt_feature_pd.values.dtype),
                                                 shape=(data_count,1),
                                                 filters=filters
                                                 )

    count = 0
    torch.manual_seed(42)
    for img_batch in tqdm(img_dataloader):
        img_batch = img_batch.to(device)
        img_batch_norm = img_batch / img_batch.norm(dim=-1, keepdim=True)
        while True:
            txt_sample = txt_feature_pd.sample(n=txt_batch_size)
            txt_sample_tensor = torch.tensor(txt_sample.values).to(device)
            txt_sample_tensor_norm = txt_sample_tensor / txt_sample_tensor.norm(dim=-1, keepdim=True)
            txt_sample_tensor_norm = txt_sample_tensor_norm

            cos_simi = 100 * img_batch_norm @ txt_sample_tensor_norm.t()

            cos_simi_indices = torch.argsort(cos_simi)

            non_cos_simi_bool_ = cos_simi < thre
            non_simi_count = torch.sum(non_cos_simi_bool_, dim=-1)

            cos_simi_bool_ = cos_simi >= thre
            simi_count = torch.sum(cos_simi_bool_, dim=-1)
            simi_sum_acc = (simi_count < 5).sum()
            print(simi_sum_acc)
            min_simi = torch.min(simi_count)
            if min_simi == 0 or simi_sum_acc>10:
                continue
            txt_lst = []
            for idx,img in enumerate(img_batch):
                non_simi_co = non_simi_count[idx]

                non_simi_indices = cos_simi_indices[idx, :non_simi_co]
                non_simi_ = non_simi_indices.shape[0]
                simi_indices = cos_simi_indices[idx, non_simi_co:]
                simi_ = simi_indices.shape[0]

                non_sample = torch.linspace(0, non_simi_ - 1, per_img_txt, dtype=torch.int)
                simi_sample = torch.linspace(0, simi_ - 1, per_img_txt, dtype=torch.int)
                non_sample_indices = non_simi_indices[non_sample]
                simi_sample_indices = simi_indices[simi_sample]
                all_sample = torch.cat([non_sample_indices, simi_sample_indices], dim=-1)

                txt_ = txt_sample_tensor[all_sample]
                txt_lst.append(txt_)

            txt_pair = torch.stack(txt_lst, dim=0)
            img_batch_repeat = img_batch.repeat(1, per_img_txt * 2).reshape(-1, per_img_txt * 2, 512)
            img_txt_pair = torch.cat([img_batch_repeat, txt_pair], dim=-1).reshape(-1, 2, 512).cpu().numpy()

            img_batch_size = img_batch.shape[0]
            non_label = torch.zeros(img_batch_size, per_img_txt, 1, dtype=torch.float32)
            simi_label = torch.ones(img_batch_size, per_img_txt, 1, dtype=torch.float32)
            label = torch.cat([non_label, simi_label], dim=1).reshape(-1, 1).numpy()


            #随机打乱顺序
            # indices = torch.randperm(label.size(0))
            # img_txt_pair = img_txt_pair[indices].numpy()
            # label = label[indices].numpy()
            data_num = img_batch_size * per_img_txt * 2
            data_carray[count:count + data_num] = img_txt_pair
            label_carray[count:count + data_num] = label
            count += data_num
            break
    pair_train_data.close()
#文本与图像配对
def txt_img_pair(
               feture_file,
               save_path,
               thre=26,
               img_per_count=20,
               txt_batch_size=1000,
               device='cuda'
                ):
    # 读取图像特征数据并提交到device
    img_pd = pd.read_hdf(feture_file,key='image')
    img_tensor = torch.tensor(img_pd.values).to(device)
    img_tensor_norm = img_tensor / img_tensor.norm(dim=-1, keepdim=True)
    #总的图像数量
    total_img_count = img_tensor.shape[0]
    # 文本数据
    txt_dataset = Txt_feature_dataste(feture_file)
    txt_dataloader = DataLoader(txt_dataset,batch_size=txt_batch_size,num_workers=4)
    data_counts = len(txt_dataset) * img_per_count * 2

    pair_train_data = tables.open_file(save_path, 'w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_carray = pair_train_data.create_carray(pair_train_data.root,
                                                'data',
                                                tables.Atom.from_dtype(img_pd.values.dtype),
                                                shape=(data_counts, 2, 512),
                                                filters=filters
                                                )
    label_carray = pair_train_data.create_carray(pair_train_data.root,
                                                 'label',
                                                 tables.Atom.from_dtype(img_pd.values.dtype),
                                                 shape=(data_counts, 1),
                                                 filters=filters
                                                 )
    count = 0
    torch.manual_seed(42)
    for txt_batch in tqdm(txt_dataloader):
        #文本提交到device并归一化
        txt_batch = txt_batch.to(device)
        txt_batch_norm = txt_batch / txt_batch.norm(dim=-1, keepdim=True)
        #计算相似度
        cos_simi = 100 * txt_batch_norm @ img_tensor_norm.t()
        #相似与不相似的分界点计算
        dem_point_set = torch.sum(cos_simi<thre,dim=-1)
        # 相似度排序
        cos_simi_indices = torch.argsort(cos_simi)
        img_lst = []
        img_id = []
        for idx,txt in enumerate(txt_batch):
            #获取每个分界点
            dem_point = dem_point_set[idx]
            #判断是否存在相似的，如果不存在则完全使用不相似的则只记录不相似的
            if total_img_count-dem_point==0:
                no_simi_sample_point = torch.linspace(0, dem_point - 1, img_per_count*2, dtype=torch.int)
                no_simi_sample = cos_simi_indices[idx][no_simi_sample_point]
                img_sample = img_tensor[no_simi_sample]
                img_lst.append(img_sample)
                img_id.append(idx)
            else:
                no_simi_sample_point = torch.linspace(0, dem_point-1, img_per_count, dtype=torch.int)
                simi_sample_point = torch.linspace(dem_point, cos_simi.shape[1] - 1, img_per_count, dtype=torch.int)
                #采样
                no_simi_sample = cos_simi_indices[idx][no_simi_sample_point]
                simi_sample = cos_simi_indices[idx][simi_sample_point]
                all_sample = torch.cat([no_simi_sample, simi_sample], dim=-1)
                img_sample = img_tensor[all_sample]
                img_lst.append(img_sample)
        #堆叠
        img_stack = torch.stack(img_lst, dim=0)
        #复制
        txt_batch_repeat = txt_batch.repeat(1, img_per_count * 2).reshape(-1, img_per_count * 2, 512)
        img_txt_pair = torch.cat([img_stack, txt_batch_repeat], dim=-1).reshape(-1, 2, 512).cpu()

        img_batch_size = txt_batch.shape[0]
        non_label = torch.zeros(img_batch_size, img_per_count, 1, dtype=torch.float32)
        simi_label = torch.ones(img_batch_size, img_per_count, 1, dtype=torch.float32)
        simi_label[img_id] = 0
        label = torch.cat([non_label, simi_label], dim=1).reshape(-1, 1)

        indices = torch.randperm(label.size(0))
        img_txt_pair = img_txt_pair[indices].numpy()
        label = label[indices].numpy()
        data_num = img_batch_size * img_per_count * 2
        data_carray[count:count + data_num] = img_txt_pair
        label_carray[count:count + data_num] = label
        count += data_num
    pair_train_data.close()
#图像与文本配对然后文本与图像配对
def pair_data(
        feature_file,
        save_path,
        thre=26,
        img_batch_size=400,
        txt_batch_size=1000,
        per_img_count_txt=10,
        per_txt_count_img=5,
        sample_txt_batch_size=100000,
        device='cuda'
):
    #读取图像数据
    img_dataset = Img_feature_dataste(feature_file)
    img_dataloader = DataLoader(img_dataset,batch_size=img_batch_size,num_workers=4)
    #读取文本数据
    txt_dataset = Txt_feature_dataste(feature_file)
    txt_dataloader = DataLoader(txt_dataset, batch_size=txt_batch_size, num_workers=4)
    txt_feature_pd = pd.read_hdf(feature_file,key='text')
    #预计数据总量
    data_counts = len(img_dataset)*per_img_count_txt*2+len(txt_dataset)*per_txt_count_img*2
    #新建H5文件
    pair_train_data = tables.open_file(save_path, 'w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_carray = pair_train_data.create_carray(pair_train_data.root,
                                                'data',
                                                tables.Atom.from_dtype(txt_feature_pd.values.dtype),
                                                shape=(data_counts, 2, 512),
                                                filters=filters
                                                )
    label_carray = pair_train_data.create_carray(pair_train_data.root,
                                                 'label',
                                                 tables.Atom.from_dtype(txt_feature_pd.values.dtype),
                                                 shape=(data_counts, 1),
                                                 filters=filters
                                                 )
    countor = 0
    for batch_img in tqdm(img_dataloader):
        batch_img = batch_img.to(device)
        batch_img_norm = batch_img / batch_img.norm(dim=-1, keepdim=True)
        #采样文本
        random_text = torch.tensor(txt_feature_pd.sample(n=sample_txt_batch_size).values).to(device)
        random_text_norm = random_text / random_text.norm(dim=-1, keepdim=True)
        #计算相似度
        img_txt_cos_score = 100*batch_img_norm@random_text_norm.t()
        del random_text_norm
        del batch_img_norm
        #相似度排序
        cos_simi_sort_indices = torch.argsort(img_txt_cos_score)
        #分界点计算
        dem_point_set = torch.sum(img_txt_cos_score < thre, dim=-1)
        txt_lst = []
        txt_id = []
        for idx,img in enumerate(batch_img):
            #获取具体的排序indices
            img_indices = cos_simi_sort_indices[idx]
            #获取分界点
            dem_point = dem_point_set[idx]
            #判断是否全无相似的
            if dem_point == sample_txt_batch_size:
                #若全无相似的则只采样不相似的
                no_sample_linespace = torch.linspace(0,dem_point-1,per_img_count_txt*2,dtype=torch.int)
                all_sample_indices = img_indices[no_sample_linespace]
                all_sample = random_text[all_sample_indices]
                txt_lst.append(all_sample)
                txt_id.append(idx)
            else:
                no_sample_linespace = torch.linspace(0, dem_point - 1, per_img_count_txt, dtype=torch.int)
                simi_sample_linespace = torch.linspace(dem_point, sample_txt_batch_size - 1, per_img_count_txt, dtype=torch.int)
                # 采样
                no_simi_sample = img_indices[no_sample_linespace]
                simi_sample = img_indices[simi_sample_linespace]
                all_sample_indices = torch.cat([no_simi_sample, simi_sample], dim=-1)
                all_sample = random_text[all_sample_indices]
                txt_lst.append(all_sample)
        del random_text
        # 堆叠
        txt_stack = torch.stack(txt_lst, dim=0)
        # 复制
        img_batch_repeat = batch_img.repeat(1, per_img_count_txt * 2).reshape(-1, per_img_count_txt * 2, 512)
        img_txt_pair = torch.cat([img_batch_repeat,txt_stack], dim=-1).reshape(-1, 2, 512).cpu().detach().numpy()

        img_batch_size = batch_img.shape[0]

        non_label = torch.zeros(img_batch_size, per_img_count_txt, 1, dtype=torch.float32)
        simi_label = torch.ones(img_batch_size, per_img_count_txt, 1, dtype=torch.float32)
        simi_label[txt_id] = 0
        label = torch.cat([non_label, simi_label], dim=1).reshape(-1, 1).numpy()

        data_num = img_batch_size * per_img_count_txt * 2

        data_carray[countor:countor + data_num] = img_txt_pair
        label_carray[countor:countor + data_num] = label
        countor += data_num
    del txt_feature_pd
    del img_dataset
    del img_dataloader
    img_tensor = torch.tensor(pd.read_hdf(feature_file, key='image').values).to(device)
    img_tensor_norm = img_tensor / img_tensor.norm(dim=-1, keepdim=True)
    total_img_count = img_tensor.shape[0]
    for batch_txt in tqdm(txt_dataloader):
        batch_txt = batch_txt.to(device)
        batch_txt_norm = batch_txt / batch_txt.norm(dim=-1, keepdim=True)
        #计算相似度
        txt_img_cos_scores = 100*batch_txt_norm@img_tensor_norm.t()
        # 相似度排序
        cos_simi_indices = torch.argsort(txt_img_cos_scores)
        # 相似与不相似的分界点计算
        dem_point_set = torch.sum(txt_img_cos_scores < thre, dim=-1)
        img_lst = []
        img_id = []
        for idx, txt in enumerate(batch_txt):
            # 获取每个分界点
            dem_point = dem_point_set[idx]
            #获取每个indices
            txt_indices = cos_simi_indices[idx]
            # 判断是否存在相似的，如果不存在则完全使用不相似的则只记录不相似的
            if total_img_count == dem_point:
                no_simi_sample_point = torch.linspace(0, dem_point - 1, per_txt_count_img * 2, dtype=torch.int)
                no_simi_sample = cos_simi_indices[idx][no_simi_sample_point]
                img_sample = img_tensor[no_simi_sample]
                img_lst.append(img_sample)
                img_id.append(idx)
            else:
                no_simi_sample_point = torch.linspace(0, dem_point - 1, per_txt_count_img, dtype=torch.int)
                simi_sample_point = torch.linspace(dem_point, total_img_count - 1, per_txt_count_img, dtype=torch.int)
                # 采样
                no_simi_sample = cos_simi_indices[idx][no_simi_sample_point]
                simi_sample = cos_simi_indices[idx][simi_sample_point]
                all_sample = torch.cat([no_simi_sample, simi_sample], dim=-1)
                img_sample = img_tensor[all_sample]
                img_lst.append(img_sample)
        # 堆叠
        img_stack = torch.stack(img_lst, dim=0)
        # 复制
        txt_batch_repeat = batch_txt.repeat(1, per_txt_count_img * 2).reshape(-1, per_txt_count_img * 2, 512)
        img_txt_pair = torch.cat([img_stack, txt_batch_repeat], dim=-1).reshape(-1, 2, 512).cpu().detach().numpy()

        img_batch_size = batch_txt.shape[0]
        non_label = torch.zeros(img_batch_size, per_txt_count_img, 1, dtype=torch.float32)
        simi_label = torch.ones(img_batch_size, per_txt_count_img, 1, dtype=torch.float32)
        simi_label[img_id] = 0
        label = torch.cat([non_label, simi_label], dim=1).reshape(-1, 1).numpy()

        data_num = img_batch_size * per_txt_count_img * 2
        data_carray[countor:countor + data_num] = img_txt_pair
        label_carray[countor:countor + data_num] = label
        countor += data_num
    pair_train_data.close()

#相似配对文件
def simi_pair(
        feature_file,
        save_path,
        thre=26,
        txt_batch_size=1000,
        per_txt_count_img=5,
        device='cuda'):
    #读取图像数据，并转tensor再归一化
    img_fea_pd = pd.read_hdf(feature_file, key='image')
    img_tensor = torch.tensor(img_fea_pd.values).to(device)
    img_tensor_norm = img_tensor / img_tensor.norm(dim=-1, keepdim=True)
    #读取文本数据，
    txt_dataset = Txt_feature_dataste(feature_file)
    txt_dataloader = DataLoader(txt_dataset, batch_size=txt_batch_size, num_workers=4)
    # 预计数据总量
    data_counts = len(txt_dataset) * per_txt_count_img
    # 新建H5文件
    pair_train_data = tables.open_file(save_path, 'w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_carray = pair_train_data.create_carray(pair_train_data.root,
                                                'data',
                                                tables.Atom.from_dtype(img_fea_pd.values.dtype),
                                                shape=(data_counts, 2, 512),
                                                filters=filters
                                                )
    label_carray = pair_train_data.create_carray(pair_train_data.root,
                                                 'label',
                                                 tables.Atom.from_dtype(img_fea_pd.values.dtype),
                                                 shape=(data_counts, 1),
                                                 filters=filters
                                                 )
    countor = 0
    total_img_count = img_tensor.shape[0]
    for batch_txt in tqdm(txt_dataloader):
        batch_txt = batch_txt.to(device)
        batch_txt_norm = batch_txt / batch_txt.norm(dim=-1, keepdim=True)
        # 计算相似度
        txt_img_cos_scores = 100 * batch_txt_norm @ img_tensor_norm.t()
        # 相似度排序
        cos_simi_indices = torch.argsort(txt_img_cos_scores)
        # 相似与不相似的分界点计算
        dem_point_set = torch.sum(txt_img_cos_scores < thre, dim=-1)
        img_lst = []
        img_id = []
        for idx, txt in enumerate(batch_txt):
            # 获取每个分界点
            dem_point = dem_point_set[idx]
            # 获取每个indices
            txt_indices = cos_simi_indices[idx]
            # 判断是否存在相似的，如果不存在则完全使用不相似的则只记录不相似的
            if total_img_count == dem_point:
                no_simi_sample_point = torch.linspace(0, dem_point - 1, per_txt_count_img, dtype=torch.int)
                no_simi_sample = cos_simi_indices[idx][no_simi_sample_point]
                img_sample = img_tensor[no_simi_sample]
                img_lst.append(img_sample)
                img_id.append(idx)
            # 如果存在相似的则只记录相似的
            else:
                simi_sample_point = torch.linspace(dem_point, total_img_count - 1, per_txt_count_img, dtype=torch.int)
                # 采样
                simi_sample = cos_simi_indices[idx][simi_sample_point]
                img_sample = img_tensor[simi_sample]
                img_lst.append(img_sample)
        # 堆叠
        img_stack = torch.stack(img_lst, dim=0)
        # 复制
        txt_batch_repeat = batch_txt.repeat(1, per_txt_count_img).reshape(-1, per_txt_count_img, 512)
        img_txt_pair = torch.cat([img_stack, txt_batch_repeat], dim=-1).reshape(-1, 2, 512).cpu().detach().numpy()

        img_batch_size = batch_txt.shape[0]
        simi_label = torch.ones(img_batch_size, per_txt_count_img, 1, dtype=torch.float32)
        simi_label[img_id] = 0
        label = simi_label.reshape(-1, 1).numpy()

        data_num = img_batch_size * per_txt_count_img
        data_carray[countor:countor + data_num] = img_txt_pair
        label_carray[countor:countor + data_num] = label
        countor += data_num
    pair_train_data.close()



def simi_data_pair(
        feature_file,
        save_path,
        thre=26,
        img_batch_size=400,
        txt_batch_size=1000,
        per_img_count_txt=15,
        per_txt_count_img=3,
        sample_txt_batch_size=100000,
        device='cuda'
):
    # 读取图像数据
    img_dataset = Img_feature_dataste(feature_file)
    img_dataloader = DataLoader(img_dataset, batch_size=img_batch_size, num_workers=4)
    # 读取文本数据
    txt_dataset = Txt_feature_dataste(feature_file)
    txt_dataloader = DataLoader(txt_dataset, batch_size=txt_batch_size, num_workers=4)
    txt_feature_pd = pd.read_hdf(feature_file, key='text')
    if len(txt_feature_pd.values) < sample_txt_batch_size:
        sample_txt_batch_size = len(txt_feature_pd.values)
    # 预计数据总量
    data_counts = len(img_dataset) * per_img_count_txt + len(txt_dataset) * per_txt_count_img
    # 新建H5文件
    pair_train_data = tables.open_file(save_path, 'w')
    filters = tables.Filters(complevel=5, complib='blosc')
    data_carray = pair_train_data.create_carray(pair_train_data.root,
                                                'data',
                                                tables.Atom.from_dtype(txt_feature_pd.values.dtype),
                                                shape=(data_counts, 2, 512),
                                                filters=filters
                                                )
    label_carray = pair_train_data.create_carray(pair_train_data.root,
                                                 'label',
                                                 tables.Atom.from_dtype(txt_feature_pd.values.dtype),
                                                 shape=(data_counts, 1),
                                                 filters=filters
                                                 )
    countor = 0
    #遍历所有图片
    for batch_img in tqdm(img_dataloader):
        batch_img = batch_img.to(device)
        batch_img_norm = batch_img / batch_img.norm(dim=-1, keepdim=True)
        # 采样文本
        random_text = torch.tensor(txt_feature_pd.sample(n=sample_txt_batch_size).values).to(device)
        random_text_norm = random_text / random_text.norm(dim=-1, keepdim=True)
        # 计算相似度
        img_txt_cos_score = 100 * batch_img_norm @ random_text_norm.t()
        del random_text_norm
        del batch_img_norm
        # 相似度排序
        cos_simi_sort_indices = torch.argsort(img_txt_cos_score)
        # 分界点计算
        dem_point_set = torch.sum(img_txt_cos_score < thre, dim=-1)
        txt_lst = []
        txt_id = []
        for idx, img in enumerate(batch_img):
            # 获取具体的排序indices
            img_indices = cos_simi_sort_indices[idx]
            # 获取分界点
            dem_point = dem_point_set[idx]
            # 判断是否完全没有相似的
            if dem_point == sample_txt_batch_size:
                # 若全无相似的则只采样不相似的
                no_sample_linespace = torch.linspace(0, dem_point - 1, per_img_count_txt, dtype=torch.int)
                all_sample_indices = img_indices[no_sample_linespace]
                all_sample = random_text[all_sample_indices]
                txt_lst.append(all_sample)
                txt_id.append(idx)
            else:
                #若有相似的则完全采样相似的
                simi_sample_linespace = torch.linspace(dem_point, sample_txt_batch_size - 1, per_img_count_txt,
                                                       dtype=torch.int)
                # 采样
                simi_sample = img_indices[simi_sample_linespace]
                all_sample = random_text[simi_sample]
                txt_lst.append(all_sample)
        del random_text
        # 堆叠
        txt_stack = torch.stack(txt_lst, dim=0)
        # 复制
        img_batch_repeat = batch_img.repeat(1, per_img_count_txt).reshape(-1, per_img_count_txt, 512)
        # 拼接
        img_txt_pair = torch.cat([img_batch_repeat, txt_stack], dim=-1).reshape(-1, 2, 512).cpu().detach().numpy()

        img_batch_size = batch_img.shape[0]
        simi_label = torch.ones(img_batch_size, per_img_count_txt, 1, dtype=torch.float32)
        simi_label[txt_id] = 0
        label = simi_label.reshape(-1, 1).numpy()

        data_num = img_batch_size * per_img_count_txt

        data_carray[countor:countor + data_num] = img_txt_pair
        label_carray[countor:countor + data_num] = label
        countor += data_num
    del txt_feature_pd
    del img_dataset
    del img_dataloader

    #每条文本匹配图片
    img_tensor = torch.tensor(pd.read_hdf(feature_file, key='image').values).to(device)
    img_tensor_norm = img_tensor / img_tensor.norm(dim=-1, keepdim=True)
    total_img_count = img_tensor.shape[0]
    #分批遍历文本
    for batch_txt in tqdm(txt_dataloader):
        batch_txt = batch_txt.to(device)
        batch_txt_norm = batch_txt / batch_txt.norm(dim=-1, keepdim=True)
        # 计算相似度
        txt_img_cos_scores = 100 * batch_txt_norm @ img_tensor_norm.t()
        # 相似度排序
        cos_simi_indices = torch.argsort(txt_img_cos_scores)
        # 相似与不相似的分界点计算
        dem_point_set = torch.sum(txt_img_cos_scores < thre, dim=-1)
        img_lst = []
        img_id = []
        for idx, txt in enumerate(batch_txt):
            # 获取每个分界点
            dem_point = dem_point_set[idx]
            # 获取每个indices
            txt_indices = cos_simi_indices[idx]
            # 判断是否存在相似的，如果不存在则完全使用不相似的则只记录不相似的
            if total_img_count == dem_point:
                no_simi_sample_point = torch.linspace(0, dem_point - 1, per_txt_count_img, dtype=torch.int)
                no_simi_sample = txt_indices[no_simi_sample_point]
                img_sample = img_tensor[no_simi_sample]
                img_lst.append(img_sample)
                img_id.append(idx)
            else:
                simi_sample_point = torch.linspace(dem_point, total_img_count - 1, per_txt_count_img, dtype=torch.int)
                # 采样
                simi_sample = txt_indices[simi_sample_point]

                img_sample = img_tensor[simi_sample]
                img_lst.append(img_sample)
        # 堆叠
        img_stack = torch.stack(img_lst, dim=0)
        # 复制
        txt_batch_repeat = batch_txt.repeat(1, per_txt_count_img).reshape(-1, per_txt_count_img, 512)
        img_txt_pair = torch.cat([img_stack, txt_batch_repeat], dim=-1).reshape(-1, 2, 512).cpu().detach().numpy()

        img_batch_size = batch_txt.shape[0]
        simi_label = torch.ones(img_batch_size, per_txt_count_img, 1, dtype=torch.float32)
        simi_label[img_id] = 0
        label = simi_label.reshape(-1, 1).numpy()

        data_num = img_batch_size * per_txt_count_img
        data_carray[countor:countor + data_num] = img_txt_pair
        label_carray[countor:countor + data_num] = label
        countor += data_num
    pair_train_data.close()

def main():
    parser = ArgumentParser(description="Pair data")

    parser.add_argument('-f', '--feature_file', type=str)
    parser.add_argument('-s', '--save_path', type=str)
    parser.add_argument('-t', '--thre', type=int)
    parser.add_argument('--img_batch_size', type=int)
    parser.add_argument('--txt_batch_size', type=int)
    parser.add_argument('--per_img_count_txt', type=int)
    parser.add_argument('--per_txt_count_img', type=int)
    parser.add_argument('--sample_txt_batch_size', type=int)

    args = parser.parse_args()
    simi_data_pair(
        args.feature_file,
        args.save_path,
        args.thre,
    )
if __name__ == '__main__':
    main()




# def image_image_pair(feture_file,train_data_path,thre=70,per_img_img=160,device='cuda',img_batch_size=200,txt_batch_size=100000):
#     #计算总的数据规模
#     img_feature_pd = pd.read_hdf(feture_file,key='image')
#     data_count = len(img_feature_pd)*per_img_img*2
#
#     img_dataset = Img_feature_dataste(feture_file)
#     img_dataloader = DataLoader(img_dataset,batch_size=img_batch_size,num_workers=4)
#
#
#     pair_train_data = tables.open_file(train_data_path,'w')
#     filters = tables.Filters(complevel=5, complib='blosc')
#     data_carray = pair_train_data.create_carray(pair_train_data.root,
#                                                 'data',
#                                                 tables.Atom.from_dtype(img_feature_pd.values.dtype),
#                                                 shape=(data_count,2,512),
#                                                 filters=filters
#                                                 )
#     label_carray = pair_train_data.create_carray(pair_train_data.root,
#                                                 'label',
#                                                  tables.Atom.from_dtype(img_feature_pd.values.dtype),
#                                                 shape=(data_count,1),
#                                                 filters=filters
#                                                 )
#     count = 0
#
#     for img_batch in img_dataloader:
#         img_batch = img_batch.to(device)
#         img_batch_norm = img_batch / img_batch.norm(dim=-1, keepdim=True)
#         img_batch_norm = img_batch_norm
#         while True:
#             img_sample = img_feature_pd.sample(n=txt_batch_size)
#             img_sample_tensor = torch.tensor(img_sample.values).to(device)
#             img_sample_tensor_norm = img_sample_tensor / img_sample_tensor.norm(dim=-1, keepdim=True)
#             img_sample_tensor_norm = img_sample_tensor_norm
#             cos_simi = 100 * img_batch_norm @ img_sample_tensor_norm.t()
#             cos_simi_indices = torch.argsort(cos_simi)
#             non_cos_simi_bool_ = cos_simi < thre
#             non_simi_count = torch.sum(non_cos_simi_bool_, dim=-1)
#             cos_simi_bool_ = cos_simi >= thre
#             simi_count = torch.sum(cos_simi_bool_, dim=-1)
#             simi_sum_acc = (simi_count < 5).sum()
#             print(simi_sum_acc)
#             min_simi = torch.min(simi_count)
#             if min_simi == 0 or simi_sum_acc>10:
#                 continue
#
#             txt_lst = []
#             for idx,img in tqdm(enumerate(img_batch),total=img_batch.shape[0],ncols=150):
#                 non_simi_co = non_simi_count[idx]
#                 non_simi_indices = cos_simi_indices[idx, :non_simi_co]
#                 non_simi_ = non_simi_indices.shape[0]
#                 simi_indices = cos_simi_indices[idx, non_simi_co:]
#                 simi_ = simi_indices.shape[0]
#
#                 non_sample = torch.linspace(0, non_simi_ - 1, per_img_img, dtype=torch.int)
#                 simi_sample = torch.linspace(0, simi_ - 1, per_img_img, dtype=torch.int)
#                 non_sample_indices = non_simi_indices[non_sample]
#                 simi_sample_indices = simi_indices[simi_sample]
#                 all_sample = torch.cat([non_sample_indices, simi_sample_indices], dim=-1)
#
#                 txt_ = img_sample_tensor[all_sample]
#                 txt_lst.append(txt_)
#
#             txt_pair = torch.stack(txt_lst, dim=0)
#             img_batch_repeat = img_batch.repeat(1, per_img_img * 2).reshape(-1, per_img_img * 2, 512)
#             img_txt_pair = torch.cat([img_batch_repeat, txt_pair], dim=-1).reshape(-1, 2, 512).cpu().numpy()
#
#             img_batch_size = img_batch.shape[0]
#             non_label = torch.zeros(img_batch_size, per_img_img, 1, dtype=torch.float32)
#             simi_label = torch.ones(img_batch_size, per_img_img, 1, dtype=torch.float32)
#             label = torch.cat([non_label, simi_label], dim=1).reshape(-1, 1).numpy()
#
#
#             #随机打乱顺序
#             # indices = torch.randperm(label.size(0))
#             # img_txt_pair = img_txt_pair[indices].numpy()
#             # label = label[indices].numpy()
#             data_num = img_batch_size * per_img_img * 2
#             data_carray[count:count + data_num] = img_txt_pair
#             label_carray[count:count + data_num] = label
#             count += data_num
#             break
#     pair_train_data.close()
#
# def text_text_pair(feture_file,train_data_path,thre=70,per_img_img=160,device='cuda',img_batch_size=200,txt_batch_size=100000):
#     # 计算总的数据规模
#     img_feature_pd = pd.read_hdf(feture_file, key='text')
#     data_count = len(img_feature_pd) * per_img_img * 2
#
#     img_dataset = Txt_feature_dataste(feture_file)
#     img_dataloader = DataLoader(img_dataset, batch_size=img_batch_size, num_workers=4)
#
#     pair_train_data = tables.open_file(train_data_path, 'w')
#     filters = tables.Filters(complevel=5, complib='blosc')
#     data_carray = pair_train_data.create_carray(pair_train_data.root,
#                                                 'data',
#                                                 tables.Atom.from_dtype(img_feature_pd.values.dtype),
#                                                 shape=(data_count, 2, 512),
#                                                 filters=filters
#                                                 )
#     label_carray = pair_train_data.create_carray(pair_train_data.root,
#                                                  'label',
#                                                  tables.Atom.from_dtype(img_feature_pd.values.dtype),
#                                                  shape=(data_count, 1),
#                                                  filters=filters
#                                                  )
#     count = 0
#
#     for img_batch in img_dataloader:
#         img_batch = img_batch.to(device)
#         img_batch_norm = img_batch / img_batch.norm(dim=-1, keepdim=True)
#         img_batch_norm = img_batch_norm
#         while True:
#             img_sample = img_feature_pd.sample(n=txt_batch_size)
#             img_sample_tensor = torch.tensor(img_sample.values).to(device)
#             img_sample_tensor_norm = img_sample_tensor / img_sample_tensor.norm(dim=-1, keepdim=True)
#             img_sample_tensor_norm = img_sample_tensor_norm
#             cos_simi = 100 * img_batch_norm @ img_sample_tensor_norm.t()
#             cos_simi_indices = torch.argsort(cos_simi)
#             non_cos_simi_bool_ = cos_simi < thre
#             non_simi_count = torch.sum(non_cos_simi_bool_, dim=-1)
#             cos_simi_bool_ = cos_simi >= thre
#             simi_count = torch.sum(cos_simi_bool_, dim=-1)
#             simi_sum_acc = (simi_count < 5).sum()
#             print(simi_sum_acc)
#             min_simi = torch.min(simi_count)
#             if min_simi == 0 or simi_sum_acc > 10:
#                 continue
#
#             txt_lst = []
#             for idx, img in tqdm(enumerate(img_batch), total=img_batch.shape[0], ncols=150):
#                 non_simi_co = non_simi_count[idx]
#                 non_simi_indices = cos_simi_indices[idx, :non_simi_co]
#                 non_simi_ = non_simi_indices.shape[0]
#                 simi_indices = cos_simi_indices[idx, non_simi_co:]
#                 simi_ = simi_indices.shape[0]
#
#                 non_sample = torch.linspace(0, non_simi_ - 1, per_img_img, dtype=torch.int)
#                 simi_sample = torch.linspace(0, simi_ - 1, per_img_img, dtype=torch.int)
#                 non_sample_indices = non_simi_indices[non_sample]
#                 simi_sample_indices = simi_indices[simi_sample]
#                 all_sample = torch.cat([non_sample_indices, simi_sample_indices], dim=-1)
#
#                 txt_ = img_sample_tensor[all_sample]
#                 txt_lst.append(txt_)
#
#             txt_pair = torch.stack(txt_lst, dim=0)
#             img_batch_repeat = img_batch.repeat(1, per_img_img * 2).reshape(-1, per_img_img * 2, 512)
#             img_txt_pair = torch.cat([img_batch_repeat, txt_pair], dim=-1).reshape(-1, 2, 512).cpu().numpy()
#
#             img_batch_size = img_batch.shape[0]
#             non_label = torch.zeros(img_batch_size, per_img_img, 1, dtype=torch.float32)
#             simi_label = torch.ones(img_batch_size, per_img_img, 1, dtype=torch.float32)
#             label = torch.cat([non_label, simi_label], dim=1).reshape(-1, 1).numpy()
#
#             # 随机打乱顺序
#             # indices = torch.randperm(label.size(0))
#             # img_txt_pair = img_txt_pair[indices].numpy()
#             # label = label[indices].numpy()
#             data_num = img_batch_size * per_img_img * 2
#             data_carray[count:count + data_num] = img_txt_pair
#             label_carray[count:count + data_num] = label
#             count += data_num
#             break
#     pair_train_data.close()
#
# def pair_img_to_txt_data(feture_file,train_data_path,img_to_txt_thre,per_img_sample_txt=160,img_batch_size=10000,txt_batch_size=10000,device='cuda',shuffle=False):
#     #读取特征数据
#     txt_dataset = Txt_feature_dataste(feture_file)
#     img_dataset = Img_feature_dataste(feture_file)
#     txt_dataloader = DataLoader(txt_dataset,batch_size=txt_batch_size,shuffle=False,num_workers=4)
#     img_dataloader = DataLoader(img_dataset, batch_size=img_batch_size, shuffle=False, num_workers=4)
#     #设置预计的数据量
#     data_count = len(img_dataset)*per_img_sample_txt*3
#     #新建H5文件
#     pair_train_data = tables.open_file(train_data_path, 'w')
#     a_ = np.array([0.1])
#     filters = tables.Filters(complevel=5, complib='blosc')
#     data_carray = pair_train_data.create_carray(pair_train_data.root,
#                                                 'data',
#                                                 tables.Atom.from_dtype(a_.dtype),
#                                                 shape=(data_count, 2, 512),
#                                                 filters=filters
#                                                 )
#     label_carray = pair_train_data.create_carray(pair_train_data.root,
#                                                  'label',
#                                                  tables.Atom.from_dtype(a_.dtype),
#                                                  shape=(data_count, 1),
#                                                  filters=filters
#                                                  )
#     #设置计数器与随机种子
#     count = 0
#     torch.manual_seed(42)
#     #成批的遍历图像特征
#     for img_feature_batch in img_dataloader:
#         #图像特征归一化并提交到GPU
#         img_feature_batch_norm = img_feature_batch / img_feature_batch.norm(dim=-1, keepdim=True)
#         img_feature_batch_norm = img_feature_batch_norm.to(device)
#         #
#         txt_tensor_lst = []
#         sample_indices_lst = []
#         sample_scores_lst = []
#
#         #与文本配对
#         for txt_feature_batch in tqdm(txt_dataloader):
#             #文本特征归一化并提交到GPU
#             txt_feature_batch_norm = txt_feature_batch / txt_feature_batch.norm(dim=-1, keepdim=True)
#             txt_feature_batch_norm = txt_feature_batch_norm.to(device)
#             #计算相似度img_batch_size*txt_batch_size
#             batch_cos_simi_score = 100*img_feature_batch_norm@txt_feature_batch_norm.t()
#             batch_cos_simi_score = batch_cos_simi_score.cpu()
#             del txt_feature_batch_norm
#             #每行按照相似度大小排序得到排序后的相似分数及其索引
#             batch_cos_simi_score_sort,batch_cos_simi_score_indices = torch.sort(batch_cos_simi_score)
#             del batch_cos_simi_score
#             #计算不想似的文本数量最小值以及相似的最大值
#             non_cos_simi_min_count = torch.min(torch.sum(batch_cos_simi_score_sort<img_to_txt_thre,dim=-1))
#             cos_simi_min_count = txt_feature_batch.shape[0]-non_cos_simi_min_count
#             #采样
#             number_of_sample = int(cos_simi_min_count)
#             non_cos_simi_sample = torch.linspace(0,non_cos_simi_min_count,number_of_sample, dtype=torch.int)
#             #拼接
#             batch_sample_indices = torch.cat([batch_cos_simi_score_indices[:,non_cos_simi_sample],batch_cos_simi_score_indices[:,non_cos_simi_min_count+1:]],dim=-1)
#             batch_sample_scores = torch.cat([batch_cos_simi_score_sort[:,non_cos_simi_sample],batch_cos_simi_score_sort[:,non_cos_simi_min_count+1:]],dim=-1)
#             sample_indices_lst.append(batch_sample_indices)
#             sample_scores_lst.append(batch_sample_scores)
#             txt_tensor_lst.append(txt_feature_batch)
#             del batch_sample_indices
#             del batch_sample_scores
#             del txt_feature_batch
#         sample_txt_lst = []
#         for idx, img in tqdm(enumerate(img_feature_batch),total=img_feature_batch.shape[0],ncols=150):
#             #平衡采样的tensor
#             txt_tensor_sample = torch.cat([txt_tensor[sample_indices_lst[idex][idx]] for idex,txt_tensor in enumerate(txt_tensor_lst)],dim=0)
#
#             cos_simi_scores = torch.cat([sample_scores[idx] for sample_scores in sample_scores_lst],dim=-1)
#
#             #排序
#             cos_simi_indices = torch.argsort(cos_simi_scores)
#             non_simi_co = torch.sum(cos_simi_scores<img_to_txt_thre)
#
#             non_sample = torch.linspace(0, non_simi_co, per_img_sample_txt, dtype=torch.int)
#             simi_sample = torch.linspace(non_simi_co+1,len(cos_simi_scores)-1, per_img_sample_txt, dtype=torch.int)
#             all_sample = torch.cat([non_sample, simi_sample], dim=-1)
#             all_sample_indices = cos_simi_indices[all_sample]
#             all_sample_txt = txt_tensor_sample[all_sample_indices]
#             sample_txt_lst.append(all_sample_txt)
#
#         txt_pair = torch.stack(sample_txt_lst, dim=0)
#         img_batch_repeat = img_feature_batch.repeat(1, per_img_sample_txt * 2).reshape(-1, per_img_sample_txt * 2, 512)
#         img_txt_pair = torch.cat([img_batch_repeat, txt_pair], dim=-1).reshape(-1, 2, 512).numpy()
#         #制作标签
#         img_batch_size_tmp = img_feature_batch.shape[0]
#         non_label = torch.zeros(img_batch_size_tmp, per_img_sample_txt, 1, dtype=torch.float32)
#         simi_label = torch.ones(img_batch_size_tmp, per_img_sample_txt, 1, dtype=torch.float32)
#         label = torch.cat([non_label, simi_label], dim=1).reshape(-1, 1).numpy()
#         data_num = img_batch_size * per_img_sample_txt * 2
#         #随机打乱顺序
#         if shuffle:
#             indices = torch.randperm(label.size(0))
#             img_txt_pair = img_txt_pair[indices]
#             label = label[indices]
#         data_carray[count:count + data_num] = img_txt_pair
#         label_carray[count:count + data_num] = label
#         count += data_num
#     pair_train_data.close()


# def pair_data(feture_file,
#               save_path,
#               pair_count,
#               first_batch_size,
#               second_batch_size,
#               img_img_thre=70,
#               img_txt_thre=25,
#               device='cuda',
#               ):
#     #读取特征
#     img_feature_pd = pd.read_hdf(feture_file, key='image')
#     txt_feature_pd = pd.read_hdf(feture_file, key='text')
#     data_count = len(img_feature_pd) * pair_count * 4
#     #batch弹出数据
#     img_dataset = Img_feature_dataste(feture_file)
#     img_dataloader = DataLoader(img_dataset, batch_size=first_batch_size, num_workers=4)
#     txt_dataset = Txt_feature_dataste(feture_file)
#     txt_dataloader = DataLoader(txt_dataset, batch_size=first_batch_size, num_workers=4)
#     #新建H5文件
#     img_img_pair = tables.open_file(save_path, 'w')
#     filters = tables.Filters(complevel=5, complib='blosc')
#     data_carray = img_img_pair.create_carray(img_img_pair.root,
#                                                 'data',
#                                                 tables.Atom.from_dtype(img_feature_pd.values.dtype),
#                                                 shape=(data_count, 2, 512),
#                                                 filters=filters
#                                                 )
#     label_carray = img_img_pair.create_carray(img_img_pair.root,
#                                                  'label',
#                                                  tables.Atom.from_dtype(img_feature_pd.values.dtype),
#                                                  shape=(data_count, 1),
#                                                  filters=filters
#                                                  )
#     count = 0
#     for img_batch in tqdm(img_dataloader,ncols=150):
#         img_batch = img_batch.to(device)
#         img_batch_norm = img_batch / img_batch.norm(dim=-1, keepdim=True)
#         while True:
#             #抽取文本
#             txt_sample = txt_feature_pd.sample(n=second_batch_size)
#             txt_sample_tensor = torch.tensor(txt_sample.values)
#             txt_sample_tensor = txt_sample_tensor.to(device)
#             del txt_sample
#             txt_sample_tensor_norm = txt_sample_tensor / txt_sample_tensor.norm(dim=-1, keepdim=True)
#             #计算相似度
#             img_txt_cos_simi = 100 * img_batch_norm @ txt_sample_tensor_norm.t()
#
#             del txt_sample_tensor_norm
#             #排序
#             img_txt_cos_simi_indices = torch.argsort(img_txt_cos_simi)
#             #计算相似的数量最小值
#             img_txt_simi_count = torch.min(torch.sum(img_txt_cos_simi > img_txt_thre, dim=-1))
#             #相似与非相似的分界点
#             img_txt_dem_point_set = torch.sum(img_txt_cos_simi < img_txt_thre, dim=-1)
#
#             #抽取图像
#             img_sample = img_feature_pd.sample(n=second_batch_size)
#             img_sample_tensor = torch.tensor(img_sample.values)
#             img_sample_tensor = img_sample_tensor.to(device)
#             del img_sample
#             img_sample_tensor_norm = img_sample_tensor / img_sample_tensor.norm(dim=-1, keepdim=True)
#             #计算相似度
#             img_img_cos_simi = 100 * img_batch_norm @ img_sample_tensor_norm.t()
#             del img_sample_tensor_norm
#             #排序
#             img_img_cos_simi_indices = torch.argsort(img_img_cos_simi)
#             #计算相似数量的最小值
#             img_img_simi_count = torch.min(torch.sum(img_img_cos_simi > img_img_thre, dim=-1))
#             #相似与非相似的分界点
#             img_img_dem_point_set = torch.sum(img_img_cos_simi < img_img_thre, dim=-1)
#             if img_txt_simi_count == 0 or img_img_simi_count == 0:
#                 continue
#             pair_lst = []
#             for idx,img in enumerate(img_batch):
#                 #先处理图像与文本
#                 dem_point = img_txt_dem_point_set[idx]
#                 non_simi_indices = img_txt_cos_simi_indices[idx, :dem_point]
#                 non_simi_ = non_simi_indices.shape[0]
#                 simi_indices = img_txt_cos_simi_indices[idx, dem_point:]
#                 simi_ = simi_indices.shape[0]
#                 non_sample = torch.linspace(0, non_simi_ - 1, pair_count, dtype=torch.int)
#                 simi_sample = torch.linspace(0, simi_ - 1, pair_count, dtype=torch.int)
#                 non_sample_indices = non_simi_indices[non_sample]
#                 simi_sample_indices = simi_indices[simi_sample]
#                 all_sample = torch.cat([non_sample_indices, simi_sample_indices], dim=-1)
#                 txt_ = txt_sample_tensor[all_sample]
#                 #再处理图像与图像
#                 img_img_dem_point = img_img_dem_point_set[idx]
#                 non_simi_indices_img = img_img_cos_simi_indices[idx, :img_img_dem_point]
#                 non_simi_img = non_simi_indices_img.shape[0]
#                 simi_indices_img = img_img_cos_simi_indices[idx, img_img_dem_point:]
#                 simi_img = simi_indices_img.shape[0]
#
#                 non_sample_img = torch.linspace(0, non_simi_img - 1, pair_count, dtype=torch.int)
#                 simi_sample_img = torch.linspace(0, simi_img - 1, pair_count, dtype=torch.int)
#
#                 non_sample_indices_img = non_simi_indices_img[non_sample_img]
#                 simi_sample_indices_img = simi_indices_img[simi_sample_img]
#
#                 all_sample_img = torch.cat([non_sample_indices_img, simi_sample_indices_img], dim=-1)
#                 img_ = img_sample_tensor[all_sample_img]
#                 per_img_pair = torch.cat([txt_,img_],dim=0)
#                 pair_lst.append(per_img_pair)
#
#             pair_tensor = torch.stack(pair_lst, dim=0)
#             img_batch_repeat = img_batch.repeat(1, pair_count * 4).reshape(-1, pair_count * 4, 512)
#             img_txt_pair = torch.cat([img_batch_repeat, pair_tensor], dim=-1).reshape(-1, 2, 512).cpu().numpy()
#
#             img_batch_size = img_batch.shape[0]
#
#             non_label = torch.zeros(img_batch_size, pair_count, 1, dtype=torch.float32)
#             simi_label = torch.ones(img_batch_size, pair_count, 1, dtype=torch.float32)
#             label = torch.cat([non_label, simi_label,non_label,simi_label], dim=1).reshape(-1, 1).numpy()
#
#             data_num = img_batch.shape[0] * pair_count * 4
#             data_carray[count:count + data_num] = img_txt_pair
#             label_carray[count:count + data_num] = label
#             count += data_num
#             break
#     img_img_pair.close()


















