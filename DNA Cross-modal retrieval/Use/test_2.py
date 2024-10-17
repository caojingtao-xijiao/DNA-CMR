import os

import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.Dataset_set import Encoder_dataset
from torch import nn


from PIL import Image

# 打开图像文件
# img_lst = ['000000343937.jpg','000000470952.jpg','000000080671.jpg','000000407002.jpg','000000273715.jpg']
# input_image = '/home/cao/桌面/new_similarity_search/simi/Use/Dna_Lib/awomanpo'
# new_size = (224, 224)
# for img_file in img_lst:
#     img_abpath = os.path.join(input_image, img_file)
#     img = Image.open(img_abpath)
#     resized_img = img.resize(new_size)
#     output_image = os.path.join(input_image,img_file[:-4]+'resize.jpg')
#     resized_img.save(output_image)
#     resized_img.close()
#
# # 调整图像大小（resize）
#  # 新的宽度和高度
# for i in range(10):
#     simi_id = np.random.randint(0,2)
#     print(simi_id)

# 保存调整大小后的图像


# 关闭图像对象


print('Resized and saved successfully.')























