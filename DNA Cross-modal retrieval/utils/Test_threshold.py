import os
import json
import random
import pandas as pd
import torch
from PIL import Image
import clip

#寻找合适的阈值
def test_txt_to_image_thr(image_file_1,image_file_2,txt_file,query='a woman in swim suit'):
    img_id_1 = os.path.basename(image_file_1).replace('.jpg', '')
    img_id_2 = os.path.basename(image_file_2).replace('.jpg', '')
    device = 'cpu'
    image_1 = Image.open(image_file_1)
    image_2 = Image.open(image_file_2)
    clip_model,prepro = clip.load("ViT-B/32",device=device)
    img_1 = prepro(image_1).unsqueeze(0).to(device)
    img_2 = prepro(image_2).unsqueeze(0).to(device)

    with open(txt_file, 'r') as f:
        content = f.read()
        text_dict = json.loads(content)
    text_lst = text_dict['annotations']
    simi_txt_lst = []
    for i in text_lst:
        if str(i['image_id']).zfill(12) == img_id_1:
            simi_txt_lst.append(i)
    for i in text_lst:
        if str(i['image_id']).zfill(12) == img_id_2:
            simi_txt_lst.append(i)


    print(simi_txt_lst)
    img_feature_1 = clip_model.encode_image(img_1)
    img_feature_2 = clip_model.encode_image(img_2)
    txt_lst = []
    for txt_dict in simi_txt_lst:
        txt_str = txt_dict['caption']
        txt_lst.append(txt_str)
    txt_lst.append(query)
    text_tensor = clip.tokenize(txt_lst)
    txt_fea = clip_model.encode_text(text_tensor)

    image_norm_1 = img_feature_1 / img_feature_1.norm(dim=-1, keepdim=True)
    text_nor_1 = txt_fea / txt_fea.norm(dim=-1, keepdim=True)
    image_norm_2 = img_feature_2 / img_feature_2.norm(dim=-1, keepdim=True)
    dis_lst = []
    for i in text_nor_1:
        txt_to_img__distance = 100 * torch.dot(image_norm_1[0], i)
        dis_lst.append(txt_to_img__distance)
    print(100 * torch.dot(image_norm_1[0], image_norm_2[0]))
    print(dis_lst[:5])
    print(dis_lst[5:])

def test_image_to_image_thr(feature_h5,thr,k1=8000,k=8000):
    img_feature = pd.read_hdf(feature_h5,key='image')
    # img_id_lst = list(img_feature.index)
    # random_id = np.array(img_id_lst[2])
    # random_id_lst = np.array(random.choices(img_id_lst,k=k))
    random_feature = torch.tensor(img_feature.sample(n=k1).values)
    random_feature_lst = torch.tensor(img_feature.sample(n=k).values)
    random_feature_norm = random_feature / random_feature.norm(dim=-1, keepdim=True)
    random_feature_lst_norm = random_feature_lst / random_feature_lst.norm(dim=-1, keepdim=True)
    cos_ = 100*random_feature_norm@random_feature_lst_norm.t()
    simi = cos_>=thr
    simi = torch.tensor(simi,dtype=torch.float32)
    print(simi)
    return simi.sum(dim=1).mean()

def test_text_to_image_thr(feature_h5,txt_h5,txt_json,k1=1,k=8000):
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

def test_image_to_image_thr(feature_h5,,k1=1,k=8000):
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

















def test_text_iamge_2(feature_h5,img_simi_thr=70,k1=1,k=8000):
    img_feature = pd.read_hdf(feature_h5, key='image')
    txt_feature = pd.read_hdf(feature_h5, key='text')
    ran_img_fea = torch.tensor(img_feature.sample(n=k1).values)
    img_2_pd = img_feature.sample(n=k)
    img_2_id = img_2_pd.index
    ran_img_fea_2 = torch.tensor(img_2_pd.values)

    ran_img_fea_norm = ran_img_fea / ran_img_fea.norm(dim=-1, keepdim=True)
    ran_img_fea_norm_2 = ran_img_fea_2 / ran_img_fea_2.norm(dim=-1, keepdim=True)
    img_cos = 100*ran_img_fea_norm@ran_img_fea_norm_2.t()
    simi_img = img_cos>img_simi_thr
    simi_img = simi_img.reshape(-1)
    print(simi_img.shape)
    print(img_2_id.shape)
    img_2_id_simi = img_2_id[simi_img.numpy()]
    print(img_2_id_simi.shape)

    txt_simi_fea = txt_feature.loc[img_2_id_simi]
    txt_tensor = torch.tensor(txt_simi_fea.values)
    txt_tensor_norm = txt_tensor / txt_tensor.norm(dim=-1, keepdim=True)
    txt_img_cos = 100*ran_img_fea_norm@txt_tensor_norm.t()

    print(txt_img_cos.mean())



# feature_h5 = '/home/cao/桌面/new_similarity_search/simi/Dataset/train_data/feature.h5'
# txt_feature = '/home/cao/桌面/new_similarity_search/simi/Dataset/train_data/feature_txt.h5'
# text_json = '/home/cao/桌面/new_similarity_search/simi/Dataset/train_data/text.json'
# # mea = test_image_to_image_thr(feature_h5,70)
# # print(mea)
# # test_text_iamge_2(feature_h5)
# test_text_to_image_thr(feature_h5,txt_feature,text_jso

def randomly_extract_img(img_dir,img_count=5):
    img_file_lst = os.listdir(img_dir)
    random_img_lst = random.choices(img_file_lst,k=img_count)
    random_abpath_lst = [os.path.join(img_dir,i) for i in random_img_lst]
    random_img_id_lst = [os.path.basename(j).replace('.jpg', '') for j in random_img_lst]
    return random_abpath_lst,random_img_id_lst

def find_corresponding_text(text_json,random_abpath_lst,random_img_id_lst):
    with open(text_json, 'r') as f:
        content = f.read()
        text_dict = json.loads(content)
    text_lst = text_dict['annotations']
    simi_random_text_lst = []
    for img_id in random_img_id_lst:
        for txt_ in text_lst:
            if txt_['image_id'] == int(img_id):
                simi_random_text_lst.append(txt_)

test_txt_to_image_thr('/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/image/000000000285.jpg',
'/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/image/000000000285.jpg',
                      '/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/text.json',
                      query='bear')
