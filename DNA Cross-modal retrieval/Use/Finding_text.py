import pandas as pd
import json

def fing_img_text(cate_name,mapping_file,text_json):
    mapping_pd = pd.read_hdf(mapping_file,key='mapping')
    cate_name_np = mapping_pd['cate_name']
    cate_name_bool_ = cate_name_np == cate_name
    cate_name_pd = mapping_pd[cate_name_bool_]
    img_id_np = cate_name_pd['image_id']
    with open(text_json,'r') as f:
        content = f.read()
        text_dict = json.loads(content)
    text_lst = text_dict['annotations']
    text_lst_set = []
    for img_id in img_id_np:
        for text in text_lst:
            if text['image_id'] == img_id:
                text_lst_set.append(text['caption'])
                break
    print(cate_name)
    print(text_lst_set)

def img_id_txt(img_id,text_json):
    '''
    根据图片id找到对应的文本
    '''
    with open(text_json,'r') as f:
        content = f.read()
        text_dict = json.loads(content)
    text_lst = text_dict['annotations']
    caption_lst = []
    for text in text_lst:
        if text['image_id'] == img_id:
            print(text['caption'])
            caption_lst.append(text['caption'])




# mapping_file = '/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/mapping.h5'
text_json = '/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/text.json'
# fing_img_text('dog',mapping_file,text_json)
# fing_img_text('bus',mapping_file,text_json)
# fing_img_text('bench',mapping_file,text_json)

img_id_txt(110359,text_json)

