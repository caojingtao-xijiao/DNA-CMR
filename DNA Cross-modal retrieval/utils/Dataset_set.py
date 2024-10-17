import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import tables
import json
import clip
from PIL import Image

class Encoder_dataset(Dataset):
    def __init__(self, h5_path):
        self.enco_data = tables.open_file(h5_path,'r')
        self.data = self.enco_data.root.data
        self.label = self.enco_data.root.label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, item):
        img_txt_pair = torch.tensor(self.data[item],dtype=torch.float32)
        label = torch.tensor(self.label[item][0],dtype=torch.float32)
        return img_txt_pair,label
#加载相似数据
class Simi_dataset(Dataset):
    def __init__(self,simi_h5):
        self.simi_pair = tables.open_file(simi_h5, 'r')
    def __len__(self):
        return len(self.simi_pair.root.label)
    def __getitem__(self, item):
        simi_data = torch.tensor(self.simi_pair.root.data[item])
        simi_label = torch.tensor(self.simi_pair.root.label[item])
        return simi_data,simi_label
#加载不相似数据
class Non_simi_dataset():
    def __init__(self,feature_file,thre,batch_size):
        self.img_feature = pd.read_hdf(feature_file,key='image')
        self.txt_feature = pd.read_hdf(feature_file,key='text')
        self.thre = thre
        self.batch_size = batch_size
    def __getitem__(self, item):
        #随机抽取样本
        sample_img_feature = torch.tensor(self.img_feature.sample(n=self.batch_size).values).to('cuda')
        sample_txt_feature = torch.tensor(self.txt_feature.sample(n=self.batch_size).values).to('cuda')
        #计算相似度
        img_norm = sample_img_feature / sample_img_feature.norm(dim=-1, keepdim=True)
        txt_norm = sample_txt_feature / sample_txt_feature.norm(dim=-1, keepdim=True)
        cos_score = torch.diag(100 * img_norm @ txt_norm.t())
        #数据配对
        no_simi_pair_data = torch.stack([sample_img_feature,sample_txt_feature],dim=1)
        #制作标签
        label = torch.zeros(self.batch_size,1,dtype=torch.float32)
        label[cos_score>self.thre] = 1
        return no_simi_pair_data,label



class Clip_img_dataset(Dataset):
    def __init__(self,img_dir):
        self.img_abpath_lst = [os.path.join(img_dir,i) for i in os.listdir(img_dir)]
        clip_model,self.preprocess = clip.load("ViT-B/32", device='cpu')
    def __len__(self):
        return len(self.img_abpath_lst)
    def __getitem__(self, item):
        img_abpath = self.img_abpath_lst[item]

        img_tensor = self.preprocess(Image.open(img_abpath))

        return img_tensor

class Clip_txt_dataset(Dataset):
    def __init__(self,json_file):
        with open(json_file, 'r') as f:
            content = f.read()
            text_dict = json.loads(content)
        self.text_lst = text_dict['annotations']
    def __len__(self):
        return len(self.text_lst)
    def __getitem__(self, item):
        txt_dict = self.text_lst[item]
        txt_tensor = clip.tokenize(txt_dict['caption'])
        return txt_tensor
class Img_feature_dataste(Dataset):
    def __init__(self,img_h5):
        self.img_pd:pd.DataFrame = pd.read_hdf(img_h5,key='image')
        self.img_id = self.img_pd.index
    def __len__(self):
        return len(self.img_pd)
    def __getitem__(self, item):
        img_tensor = torch.tensor(self.img_pd.iloc[item])

        return img_tensor
class Txt_feature_dataste(Dataset):
    def __init__(self,img_h5):
        self.img_pd:pd.DataFrame = pd.read_hdf(img_h5,key='text')
        self.img_id = self.img_pd.index
    def __len__(self):
        return len(self.img_pd)
    def __getitem__(self, item):
        img_tensor = torch.tensor(self.img_pd.iloc[item])
        return img_tensor









# class Encoder_dataset(Dataset):
#     def __init__(self, root_path,img_to_img_thr=75,txt_to_txt_thr=75,txt_to_img_thr=26,isTrain=False,batch_size=1):
#         self.root_path = root_path
#         self.image = pd.read_hdf(self.root_path, key="image")
#         self.text = pd.read_hdf(self.root_path, key="text")
#         self.batch_size = batch_size
#         self.img_to_img_thr = img_to_img_thr
#         self.txt_to_txt_thr = txt_to_txt_thr
#         self.txt_to_img_thr = txt_to_img_thr
#         self.isTrain = isTrain
#     def open_image(self,image_ids):
#         image_lst = image_ids.numpy().tolist()
#         image_dir = os.path.join(os.path.abspath(os.path.join(self.root_path,os.path.pardir)),'image')
#         for image_id in image_lst:
#             image_id_zi = str(image_id).zfill(12)
#             print(type(image_id_zi))
#             img_path = os.path.join(
#                 image_dir, '%s.jpg' % (image_id_zi)
#             )
#             image = Image.open(img_path)
#             image.load()
#             yield image
#     def open_text(self,text_ids):
#         txt_id_lst = text_ids.numpy().tolist()
#         text_dir = os.path.join(os.path.abspath(os.path.join(self.root_path,os.path.pardir)),'text.json')
#         with open(text_dir, 'r') as f:
#             content = f.read()
#             text_dict = json.loads(content)
#         text_lst = text_dict['annotations']
#         id_text_dict = dict()
#         for item in text_lst:
#             id_text_dict[str(item['image_id'])] = item['caption']
#         for text_id in txt_id_lst:
#             yield id_text_dict[str(text_id)]
#
#     def __len__(self):
#         return len(self.image)
#     def __getitem__(self,index):
#         # id1 = np.random.randint(2,3)
#         id2 = np.random.randint(0,2)
#         if self.isTrain:
            # if id1==0:

            #     while True:
            #         image1 = self.image.sample(n=self.batch_size, replace=True)
            #         image2 = self.image.sample(n=self.batch_size, replace=True)
            #         image1_values = torch.tensor(image1.values[0])
            #         image2_values = torch.tensor(image2.values[0])
            #         ids = []
            #         image1_id = int(image1.index[0])
            #         image2_id = int(image2.index[0])
            #         ids.append(image1_id)
            #         ids.append(image2_id)
            #         ids.append(0)
            #         image1_norm = image1_values / image1_values.norm(dim=-1, keepdim=True)
            #         image2_norm = image2_values / image2_values.norm(dim=-1, keepdim=True)
            #         img2_distance = 100 * torch.dot(image1_norm,image2_norm)
            #         isSimi = img2_distance > self.img_to_img_thr
            #         isSimi = int(isSimi)
            #         if id2 == isSimi:
            #             return torch.tensor(ids),torch.stack((image1_values, image2_values)), torch.tensor(isSimi, dtype=torch.float)
            #         else:
            #             continue
            # elif id1==1:
            #
            #     while True:
            #         text1 = self.text.sample(n=self.batch_size, replace=True)
            #         text2 = self.text.sample(n=self.batch_size, replace=True)
            #         text1_values = torch.tensor(text1.values[0])
            #         text2_values = torch.tensor(text2.values[0])
            #         ids = []
            #         text1_id = int(text1.index[0])
            #         text2_id = int(text2.index[0])
            #         ids.append(text1_id)
            #         ids.append(text2_id)
            #         ids.append(1)
            #         text1_norm = text1_values / text1_values.norm(dim=-1, keepdim=True)
            #         text2_norm = text2_values / text2_values.norm(dim=-1, keepdim=True)
            #         img2_distance = 100 * torch.dot(text1_norm, text2_norm)
            #         isSimi = img2_distance > self.txt_to_txt_thr
            #         isSimi = int(isSimi)
            #         if id2 == isSimi:
            #             return torch.tensor(ids), torch.stack((text1_values, text2_values)), torch.tensor(isSimi, dtype=torch.float)
            #         else:
            #             continue


        #     while True:
        #         image = self.image.sample(n=self.batch_size, replace=True)
        #         text = self.text.sample(n=self.batch_size, replace=True)
        #         image_value = torch.tensor(image.values[0])
        #         text_value = torch.tensor(text.values[0])
        #         ids = []
        #         image_id = int(image.index[0])
        #         text_id = int(text.index[0])
        #         ids.append(image_id)
        #         ids.append(text_id)
        #         ids.append(2)
        #         image_norm = image_value / image_value.norm(dim=-1, keepdim=True)
        #         text_norm = text_value / text_value.norm(dim=-1, keepdim=True)
        #         txt_to_img__distance = 100 * torch.dot(image_norm,text_norm)
        #         isSimi = txt_to_img__distance > self.txt_to_img_thr
        #         isSimi = int(isSimi)
        #         if id2 == isSimi:
        #             return torch.tensor(ids),torch.stack((image_value, text_value)), torch.tensor(isSimi, dtype=torch.float)
        #         else:
        #             continue
        # else:
        #     if id1 == 0:
        #         image1 = self.image.sample(n=self.batch_size, replace=True)
        #         image2 = self.image.sample(n=self.batch_size, replace=True)
        #         image1_values = torch.tensor(image1.values[0])
        #         image2_values = torch.tensor(image2.values[0])
        #         ids = []
        #         image1_id = int(image1.index[0])
        #         image2_id = int(image2.index[0])
        #         ids.append(image1_id)
        #         ids.append(image2_id)
        #         ids.append(0)
        #         image1_norm = image1_values / image1_values.norm(dim=-1, keepdim=True)
        #         image2_norm = image2_values / image2_values.norm(dim=-1, keepdim=True)
        #         img2_distance = 100 * torch.dot(image1_norm, image2_norm)
        #         isSimi = img2_distance > self.img_to_img_thr
        #         isSimi = int(isSimi)
        #         return torch.tensor(ids), torch.stack((image1_values, image2_values)), torch.tensor(isSimi,dtype=torch.float)
        #     elif id1 == 1:
        #         text1 = self.text.sample(n=self.batch_size, replace=True)
        #         text2 = self.text.sample(n=self.batch_size, replace=True)
        #         text1_values = torch.tensor(text1.values[0])
        #         text2_values = torch.tensor(text2.values[0])
        #         ids = []
        #         text1_id = int(text1.index[0])
        #         text2_id = int(text2.index[0])
        #         ids.append(text1_id)
        #         ids.append(text2_id)
        #         ids.append(1)
        #         text1_norm = text1_values / text1_values.norm(dim=-1, keepdim=True)
        #         text2_norm = text2_values / text2_values.norm(dim=-1, keepdim=True)
        #         img2_distance = 100 * torch.dot(text1_norm, text2_norm)
        #         isSimi = img2_distance > self.txt_to_txt_thr
        #         isSimi = int(isSimi)
        #         return torch.tensor(ids), torch.stack((text1_values, text2_values)), torch.tensor(isSimi, dtype=torch.float)
        #     elif id1 == 2:
        #         image = self.image.sample(n=self.batch_size, replace=True)
        #         text = self.text.sample(n=self.batch_size, replace=True)
        #         image_value = torch.tensor(image.values[0])
        #         text_value = torch.tensor(text.values[0])
        #         ids = []
        #         image_id = int(image.index[0])
        #         text_id = int(text.index[0])
        #         ids.append(image_id)
        #         ids.append(text_id)
        #         ids.append(2)
        #         image_norm = image_value / image_value.norm(dim=-1, keepdim=True)
        #         text_norm = text_value / text_value.norm(dim=-1, keepdim=True)
        #         txt_to_img__distance = 100 * torch.dot(image_norm, text_norm)
        #         isSimi = txt_to_img__distance > self.txt_to_img_thr
        #         isSimi = int(isSimi)
        #         return torch.tensor(ids),torch.stack((image_value, text_value)), torch.tensor(isSimi, dtype=torch.float)











