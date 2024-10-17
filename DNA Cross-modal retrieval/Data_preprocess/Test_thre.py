import os
from utils.Tool_set import copy_image
import pandas as pd
import clip
import torch
import json


def test_thre(feature_file,mapping_file):
    img_feature = pd.read_hdf(feature_file,key='image')
    random_img = img_feature.sample(n=1)
    sample_img_id = random_img.index[0]
    print(sample_img_id)
    sample_img_feature = torch.tensor(random_img.values,dtype=torch.float32)
    mapping_file = pd.read_hdf(mapping_file,key='mapping')
    mapping_file.set_index('image_id',inplace=True)
    sample_cate_name = list(set(list(mapping_file.loc[int(sample_img_id)]['cate_name'])))

    device = "cuda"
    model, preprocess = clip.load("ViT-B/32", device=device)
    feature_lst = []
    text_op = 'person'
    print(text_op)
    print(sample_cate_name)
    feature_op = model.encode_text(clip.tokenize(['train']).to(device)).cpu()
    feature_lst.append(feature_op)

    for name in sample_cate_name:
        name_token = clip.tokenize([name]).to(device)
        text_features = model.encode_text(name_token).cpu()
        feature_lst.append(text_features)
    print(' '.join(sample_cate_name))
    feature_all = model.encode_text(clip.tokenize(' '.join(sample_cate_name)).to(device)).cpu()
    feature_lst.append(feature_all)

    new_feature_lst = feature_lst[1:-1]
    cumulative_sums = []
    current_sum = torch.zeros(1,512)
    for num in new_feature_lst:
        current_sum += num
        cumulative_sums.append(current_sum)


    feature_tensor = torch.cat(feature_lst,dim=0)
    cumulative_sums_tensor = torch.cat(cumulative_sums,dim=0)


    sample_img_feature_norm = sample_img_feature / sample_img_feature.norm(dim=-1, keepdim=True)
    feature_tensor_norm = feature_tensor / feature_tensor.norm(dim=-1, keepdim=True)
    feature_tensor_norm = torch.tensor(feature_tensor_norm,dtype=torch.float32)
    cumulative_sums_tensor_norm = cumulative_sums_tensor/cumulative_sums_tensor.norm(dim=-1, keepdim=True)
    cumulative_sums_tensor_norm = torch.tensor(cumulative_sums_tensor_norm, dtype=torch.float32)

    cos_simi = 100*sample_img_feature_norm@feature_tensor_norm.t()
    cos_simi_2 = 100*sample_img_feature_norm@cumulative_sums_tensor_norm.t()

    print(cos_simi)
    print(cos_simi_2)
def text_to_image_thr(feature_h5,txt_h5,txt_json,k1=1,k=8000):
    img_feature = pd.read_hdf(feature_h5,key='image')
    txt_feature = pd.read_hdf(txt_h5,key='text')
    img_feature_1 = img_feature.sample(n=k1)
    img_id_np = img_feature_1.index
    txt_feature_1 = txt_feature.sample(n=k)
    txt_id_np = txt_feature_1.index
    random_feature = torch.tensor(img_feature_1.values,dtype=torch.float32)
    random_feature_lst = torch.tensor(txt_feature_1.values,dtype=torch.float32)

    print(img_id_np)

    random_feature_norm = random_feature / random_feature.norm(dim=-1, keepdim=True)
    random_feature_lst_norm = random_feature_lst / random_feature_lst.norm(dim=-1, keepdim=True)
    cos_ = 100*random_feature_norm@random_feature_lst_norm.t()
    cos_ = cos_.reshape(-1)
    simi_1 = txt_id_np[cos_ < 20]
    simi_2 = txt_id_np[(cos_ >= 20) & (cos_ < 21)]
    simi_3 = txt_id_np[(cos_ >= 21) & (cos_ < 22)]
    simi_4 = txt_id_np[(cos_ >= 22) & (cos_ < 23)]
    simi_5 = txt_id_np[(cos_ >= 23) & (cos_ < 24)]
    simi_6 = txt_id_np[(cos_ >= 24) & (cos_ < 25)]
    simi_7 = txt_id_np[(cos_ >= 25) & (cos_ < 26)]
    simi_8 = txt_id_np[(cos_ >= 26)]

    simi_lst = [simi_1,simi_2,simi_3,simi_4,simi_5,simi_6,simi_7,simi_8]
    simi_count = [len(i) for i in simi_lst]




    with open(txt_json, 'r') as f:
        content = f.read()
        text_dict = json.loads(content)
    text_lst = text_dict['annotations']
    lst_lst = [[] for i in range(len(simi_lst))]
    for i in text_lst:
        if str(i['id']).zfill(12) in txt_id_np:
            for idx,j in enumerate(simi_lst):
                if str(i['id']).zfill(12) in j:
                    lst_lst[idx].append(i)
    sample_lst = []
    for i in lst_lst:
        if len(i) >= 10:
            sample_lst.append(i[:10])
        else:
            sample_lst.append(i)
    print(simi_count)
    print(sum(simi_count))
    for i in sample_lst:
        print(i)
def image_to_image_thr(feature_h5,txt_json,img_thre_low,img_gap,txt_thre_low,txt_gap,k1=1,k=30000):
    #读取图像与文本的特征
    img_feature = pd.read_hdf(feature_h5,key='image')
    txt_feature = pd.read_hdf(feature_h5,key='text')
    #随机抽取一张图片与一段文本
    img_feature_sample_low = img_feature.sample(n=k1)
    txt_feature_sample_low = txt_feature.sample(n=k1)
    #随机抽取大量的图片与文本
    img_feature_sample_high = img_feature.sample(n=k)
    txt_feature_sample_high = txt_feature.sample(n=k)
    #单张图片与文本的id
    img_id_low = img_feature_sample_low.index[0]
    txt_id_low = txt_feature_sample_low.index[0]
    #多张图片与文本的id
    img_id_high_np = img_feature_sample_high.index
    txt_id_high_np = txt_feature_sample_high.index
    # #根据图像之间的阈值新建文件夹
    # img_id_name_lst = ['sample_img',
    #                   f'<{img_thre_low}',
    #                   f'{img_thre_low}-{img_thre_low + 2}',
    #                   f'{img_thre_low + 2}-{img_thre_low + 4}',
    #                   f'{img_thre_low + 4}-{img_thre_low + 6}',
    #                   f'{img_thre_low + 6}-{img_thre_low + 8}',
    #                   f'>{img_thre_low + 8}']
    # img_to_img_dir = os.path.join(test_tmp_dir,'image')
    # img_dir_abpath_lst = [os.path.join(img_to_img_dir, i) for i in img_id_name_lst]
    # for dir in img_dir_abpath_lst:
    #     os.makedirs(dir)
    # #根据文本之间阈值新建文本
    # txt_to_txt_file = open()


    img_feature_sample_low_tensor = torch.tensor(img_feature_sample_low.values,dtype=torch.float32)
    txt_feature_sample_low_tensor = torch.tensor(txt_feature_sample_low.values,dtype=torch.float32)
    img_feature_sample_high_tensor = torch.tensor(img_feature_sample_high.values,dtype=torch.float32)
    txt_feature_sample_high_tensor = torch.tensor(txt_feature_sample_high.values,dtype=torch.float32)





    print(f'被选中[img]的id是：{img_id_low}')
    print(f'被选中[txt]的id是：{txt_id_low}')
    with open(txt_json, 'r') as f:
        content = f.read()
        text_dict = json.loads(content)
    text_lst = text_dict['annotations']

    for i in text_lst:
        if i['id'] == txt_id_low:
            tt = i['caption']
            print(f'被选中[txt]是：{tt}')

    img_feature_sample_low_tensor_norm = img_feature_sample_low_tensor / img_feature_sample_low_tensor.norm(dim=-1, keepdim=True)
    txt_feature_sample_low_tensor_norm = txt_feature_sample_low_tensor / txt_feature_sample_low_tensor.norm(dim=-1, keepdim=True)
    img_feature_sample_high_tensor_norm =img_feature_sample_high_tensor / img_feature_sample_high_tensor.norm(dim=-1, keepdim=True)
    txt_feature_sample_high_tensor_norm = txt_feature_sample_high_tensor / txt_feature_sample_high_tensor.norm(dim=-1, keepdim=True)

    img_img_cos_ = (100*img_feature_sample_low_tensor_norm@img_feature_sample_high_tensor_norm.t()).reshape(-1)
    txt_txt_cos_ = (100*txt_feature_sample_low_tensor_norm@txt_feature_sample_high_tensor_norm.t()).reshape(-1)


    img_img_simi_1 = img_id_high_np[img_img_cos_<img_thre_low]
    img_img_simi_2 = img_id_high_np[(img_img_cos_ >= img_thre_low) & (img_img_cos_ < img_thre_low+img_gap)]
    img_img_simi_3 = img_id_high_np[(img_img_cos_ >= img_thre_low+img_gap) & (img_img_cos_ < img_thre_low+2*img_gap)]
    img_img_simi_4 = img_id_high_np[(img_img_cos_ >= img_thre_low+2*img_gap) & (img_img_cos_ < img_thre_low+3*img_gap)]
    img_img_simi_5 = img_id_high_np[(img_img_cos_ >= img_thre_low+3*img_gap) & (img_img_cos_ < img_thre_low+4*img_gap)]
    img_img_simi_6 = img_id_high_np[img_img_cos_>=img_thre_low+4*img_gap]

    txt_txt_simi_1 = txt_id_high_np[txt_txt_cos_ < txt_thre_low]
    txt_txt_simi_2 = txt_id_high_np[(txt_txt_cos_ >=txt_thre_low ) & (txt_txt_cos_ < txt_thre_low+txt_gap)]
    txt_txt_simi_3 = txt_id_high_np[(txt_txt_cos_ >= txt_thre_low+txt_gap) & (txt_txt_cos_ < txt_thre_low+2*txt_gap)]
    txt_txt_simi_4 = txt_id_high_np[(txt_txt_cos_ >= txt_thre_low+2*txt_gap) & (txt_txt_cos_ < txt_thre_low+3*txt_gap)]
    txt_txt_simi_5 = txt_id_high_np[(txt_txt_cos_ >= txt_thre_low+3*txt_gap) & (txt_txt_cos_ < txt_thre_low+4*txt_gap)]
    txt_txt_simi_6 = txt_id_high_np[txt_txt_cos_ >= txt_thre_low+4*txt_gap]

    img_img_simi_lst = [img_img_simi_1,img_img_simi_2,img_img_simi_3,img_img_simi_4,img_img_simi_5,img_img_simi_6]
    txt_txt_lst = [txt_txt_simi_1,txt_txt_simi_2,txt_txt_simi_3,txt_txt_simi_4,txt_txt_simi_5,txt_txt_simi_6]




    img_id_lst = [list(i)[:5] for i in img_img_simi_lst]
    print('图像与图像')
    for i in img_id_lst:
        print(i)


    txt_lst = [[] for i in range(len(txt_txt_lst))]
    for i in text_lst:
        if i['id'] in txt_id_high_np:
            for idx,j in enumerate(txt_txt_lst):
                if i['id'] in j:
                    txt_lst[idx].append(i)
    sample_lst = []
    for i in txt_lst:
        if len(i) >= 10:
            sample_lst.append(i[:5])
        else:
            sample_lst.append(i)
    print('文本与文本')
    for i in sample_lst:
        print(i)









feature_file = '/home/cao/桌面/new_similarity_search/simi/Dataset/train_data/new_feature.h5'
txt_json = '/home/cao/桌面/new_similarity_search/simi/Dataset/train_data/text.json'
# mapping_file = '/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/mapping.h5'
#
# test_thre(feature_file,mapping_file)

image_to_image_thr(feature_file, txt_json,68,1,70,1)
