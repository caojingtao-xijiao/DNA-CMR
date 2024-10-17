import torch



sample_img_feature = torch.rand(200,512)
sample_txt_feature = torch.rand(200,512)
no_simi_pair_data = torch.stack([sample_img_feature,sample_txt_feature],dim=1)
print(no_simi_pair_data.shape)


print(sample_img_feature[0])
print(sample_txt_feature[0])
print(no_simi_pair_data[0])