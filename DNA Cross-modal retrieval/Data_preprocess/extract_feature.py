import json
import torch
import clip
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import concurrent.futures

#提取训练数据特征
device = "cuda"
model, preprocess = clip.load("ViT-B/32",device=device)

def image_preprocess(fp):
    index,img = fp
    img = Image.open(img)
    return index,preprocess(img).unsqueeze(0).to(device)
def text_preprocess(fp):
    index,txt = fp
    return index,txt['caption']
def multi_pre_process(data_lst,fun,max_workers=20):
    process_res = dict()  # 创建空字典，存储结果
    # 创建 ThreadPoolExecutor 对象，指定线程数为 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交多个任务到线程池
        futures = [executor.submit(fun, (index,data)) for index,data in enumerate(data_lst)]
    # wait方法等待一组Future对象完成，可以传入一个或多个Future对象，并指定等待的超时时间
    completed, not_completed = concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)
    # 获取任务执行结果
    for future in completed:
        img_index, img_arr = future.result()
        process_res[img_index] = img_arr
    # 按键（索引）排序并获取值放入列表
    sorted_imgs = [img for index, img in sorted(process_res.items())]
    return sorted_imgs

def extract_features(data_dir,save_name,batch_size=1000):
    #获取每张图片的绝对路径
    img_dir = os.path.join(data_dir, 'image')
    image_paths = os.listdir(img_dir)
    image_ab_paths = [os.path.join(img_dir,i) for i in image_paths]
    #特征存储路径
    save_path = os.path.join(data_dir,save_name)
    feature_store = pd.HDFStore(save_path, complevel=9, mode='w')
    if len(image_ab_paths) > batch_size:
        splits = np.array_split(np.array(image_ab_paths), len(image_ab_paths) / batch_size)
    else:
        splits = np.expand_dims(np.array(image_ab_paths), axis=0)
    print(len(image_ab_paths))
    for split in tqdm(splits,desc='Extract image features'):
        image_id = [os.path.basename(path).replace('.jpg', '') for path in split]
        image_lst = multi_pre_process(split,image_preprocess)

        # 转成tensor
        image_array = torch.cat(image_lst).to(device)
        # 使用模型进行预测
        features = model.encode_image(image_array)
        features = features.cpu().detach().numpy()
        # 将结果保存
        frame = pd.DataFrame(features, index=image_id)
        # 使用df作为关键词
        feature_store.append('image', frame)
    #读取文本特征
    with open(os.path.join(data_dir ,'text.json'), 'r') as f:
        content = f.read()
        text_dict = json.loads(content)
    text_lst = text_dict['annotations']
    print(len(text_lst))

    #文本分组
    if len(text_lst) > batch_size:
        text_splits = np.array_split(np.array(text_lst), len(text_lst) / batch_size *2)
    else:
        text_splits = np.expand_dims(np.array(text_lst), axis=0)
    #分批处理
    for text_split in tqdm(text_splits, desc='Extract text feature'):
        text_id = [text['id'] for text in text_split]
        text_lst = multi_pre_process(text_split, text_preprocess)
        text_tensor = clip.tokenize(text_lst).to(device)
        text_features = model.encode_text(text_tensor)
        text_features = text_features.cpu().detach().numpy()
        text_frame = pd.DataFrame(text_features, index=text_id)
        feature_store.append('text', text_frame)
    #图像分组

    feature_store.close()

def extract_txt_features(data_dir,batch_size=1000):
    save_path = data_dir + '/feature_txt.h5'
    feature_store = pd.HDFStore(save_path, complevel=9, mode='w')
    with open(data_dir + '/text.json', 'r') as f:
        content = f.read()
        text_dict = json.loads(content)
    text_lst = text_dict['annotations']
    print(len(text_lst))
    if len(text_lst) > batch_size:
        text_splits = np.array_split(np.array(text_lst), len(text_lst) / batch_size * 2)
    else:
        text_splits = np.expand_dims(np.array(text_lst), axis=0)
    for text_split in tqdm(text_splits, desc='Extract text feature'):
        text_id = [str(text['id']).zfill(12) for text in text_split]
        text_lst = multi_pre_process(text_split, text_preprocess)
        text_tensor = clip.tokenize(text_lst).to(device)
        text_features = model.encode_text(text_tensor)
        del text_tensor
        text_features_pd = text_features.detach().cpu().numpy()
        text_frame = pd.DataFrame(text_features_pd, index=text_id)
        del text_features, text_features_pd
        feature_store.append('text', text_frame)
    feature_store.close()

def mapping_relationship(instance_json,save_path):
    with open(instance_json,'r') as f:
        content = f.read()
        text_dict = json.loads(content)
    ann_lst = text_dict['annotations']
    cate_lst = text_dict['categories']

    img_id_lst = []
    cate_id_lst = []
    cate_name_lst = []
    for ann in tqdm(ann_lst):
        img_id_lst.append(ann['image_id'])
        cate_id_lst.append(ann['category_id'])
        for j in cate_lst:
            if ann['category_id'] == j['id']:
                cate_name_lst.append(j['name'])
    mapping_df = pd.DataFrame(columns=['image_id','cate_id','cate_name'])
    mapping_df['image_id'] = img_id_lst
    mapping_df['cate_id'] = cate_id_lst
    mapping_df['cate_name'] = cate_name_lst

    mapping_df.to_hdf(save_path,key='mapping')

def feature_file_adjustment(feature_file_1,feature_file_2,new_feature_file):
    img_feature = pd.read_hdf(feature_file_1,key='image')
    txt_feature = pd.read_hdf(feature_file_2,key='text')
    txt_id = txt_feature.index
    txt_id = np.array(txt_id,dtype=np.int)
    txt_feature.index = txt_id
    img_feature.to_hdf(new_feature_file,key='image')
    txt_feature.to_hdf(new_feature_file,key='text')









if __name__ == '__main__':
    train_data_dir = '/home/cao/桌面/new_similarity_search/simi/Dataset/train_data'
    train_save_name = 'train_feature.h5'
    val_data_dir = '/home/cao/桌面/new_similarity_search/simi/Dataset/val_data'
    val_save_name = 'val_feature.h5'
    extract_features(val_data_dir,val_save_name)
    extract_features(train_data_dir,train_save_name)
    # feature_file_1 = '/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/feature.h5'
    # feature_file_2 = '/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/feature_txt.h5'
    # new_feature_file = '/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/new_feature.h5'
    # feature_file_adjustment(feature_file_1,feature_file_2,new_feature_file)
    # # extract_txt_features('../train_data')
    # # extract_txt_features('../val_data')
    # # instance_json = '/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/instance_val.json'
    # # save_path = '/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/mapping.h5'
    # # mapping_relationship(instance_json,save_path)
    # # a = pd.read_hdf(save_path,key='mapping')
    # # b = a['image_id'][0]
    # # print(b)
    # # print(a.iloc[0])
    # # for idx,i in enumerate(a['image_id'][1:]):
    # #     if i==b:
    # #         print(a.iloc[idx+1])
    #
    # fea_img = pd.read_hdf(new_feature_file,key='image')
    # fea_txt = pd.read_hdf(new_feature_file,key='text')
    # print(fea_img)
    # print(fea_txt)






