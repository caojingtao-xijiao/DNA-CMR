import os.path
import sys
sys.path.append('/home/cao/桌面/new_similarity_search/simi')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import torch
import clip

from models.simulator import simulator
from models.encoder import Encoder
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

def Image_feature_to_DNA_library(encoder_model,encoder_model_param_path,H5_path,DNA_library_save_path,batch_size=1000,device='cuda'):
    encoder_model = encoder_model.to(device)
    check_c = torch.load(encoder_model_param_path)
    encoder_model.load_state_dict(check_c['model_params'])
    feature = pd.read_hdf(H5_path,key='image')
    DNA_library = pd.HDFStore(DNA_library_save_path, complevel=9, mode='a')
    print('正在将数据转为DNA序列库...')
    count = 0
    while len(feature)-count>0:
        feature_id = feature.index[count:count + batch_size]
        batch_feature = torch.tensor(feature.iloc[count:count + batch_size].values).to(device)
        encoder_model.eval()
        with torch.no_grad():
            batch_DNA_seq = encoder_model.feature_to_seq(batch_feature)
        frame = pd.DataFrame(batch_DNA_seq, index=feature_id)
        DNA_library.append('DNA_library',frame)
        count += batch_size
    DNA_library.close()
    print('完成了！')

def quary_to_dna(quary,quary_id,encoder,encoder_model_param_path,quary_save_path,H5_path,device='cuda'):
    print('正在将关键词转为DNA序列...')
    encoder = encoder.to(device)
    feature_lib = pd.read_hdf(H5_path, key='image')

    check_c = torch.load(encoder_model_param_path)
    encoder.load_state_dict(check_c['model_params'])
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    quary_h5 = pd.HDFStore(quary_save_path, complevel=9, mode='a')
    #query-->feature
    if quary[-3:] == 'jpg':
        info = preprocess(Image.open(quary)).unsqueeze(0).to(device)
        query_feature = clip_model.encode_image(info)

    else:
        info = clip.tokenize([quary]).to(device)
        query_feature = clip_model.encode_text(info)
    #feature-->dna
    encoder.eval()
    with torch.no_grad():
        query_feature = torch.tensor(query_feature,dtype=torch.float32)
        query_seq = encoder.feature_to_seq(query_feature)
    frame = pd.DataFrame(query_seq, index=[quary_id])
    try:
        his = quary_h5['Quary_DNA']
        if quary_id not in his.index:
            his.loc[quary_id] = query_seq
            quary_h5.put('Quary_DNA', his)
    except:
        quary_h5.put('Quary_DNA',frame)


    cos_lst = []
    for idx in feature_lib.index:
        target_feature = torch.tensor(feature_lib.loc[idx].values).to(device)
        query_feature = query_feature.squeeze(dim=0)
        target_norm = target_feature / target_feature.norm(dim=-1, keepdim=True)
        query_norm = query_feature / query_feature.norm(dim=-1, keepdim=True)
        cos_simi = 100 * torch.dot(target_norm, query_norm)
        cos_lst.append(cos_simi.cpu().numpy())
    cos_frame = pd.DataFrame(np.array(cos_lst),index=feature_lib.index,columns=['cos_simi'])
    quary_h5.put(quary_id,cos_frame)
    quary_h5.close()
    print('完成了！')

def search_simu(dna_lib,quary_id,quary_save_path):
    print('关键词碱基序列与DNA库中的序列杂交模拟中...')
    dna_library = pd.read_hdf(dna_lib,key="DNA_library")
    # infor_id = dna_library.index
    quary_lib = pd.read_hdf(quary_save_path,key="Quary_DNA")

    if quary_id in quary_lib.index:
        quary_seq = quary_lib.loc[quary_id].values
    else:
        exit('quary id not exist')

    quary = np.repeat([quary_seq],len(dna_library))

    quary_dna_lib_pairs = pd.DataFrame({
        "dna_library":dna_library.values[:,0],
        "quary":quary
    })
    simu_yields = simulator(quary_dna_lib_pairs)
    save_pd = pd.HDFStore(quary_save_path,complevel=9, mode='a')
    query_pd = save_pd[quary_id]
    query_pd['yields'] = simu_yields

    save_pd.put(quary_id,query_pd)
    save_pd.close()
    print('完成了！')


def top_search_image_ids(quary_id,quary_save_path,img_dir,search_result_dir,top=20,new_size = (224, 224)):
    print('检索到的杂交分数最高的图像正在输出...')
    quary_lib = pd.read_hdf(quary_save_path, key=quary_id)
    sort_lib = quary_lib.sort_values(by='yields', ascending=False)
    most_similar_id = sort_lib.index[:top]
    if not os.path.exists(search_result_dir):
        os.makedirs(search_result_dir)
        img_abpath = [os.path.join(img_dir,f'{i}.jpg') for i in most_similar_id]
        search_abpath = [os.path.join(search_result_dir,f'{i}.jpg') for i in most_similar_id]
        for cou in range(len(search_abpath)):
            img_file = img_abpath[cou]
            sea = search_abpath[cou]
            # os.system(f'cp {img} {sea}')
            img = Image.open(img_file)
            resized_img = img.resize(new_size)
            # output_image = os.path.join(input_image, img_file[:-4] + 'resize.jpg')
            resized_img.save(sea)
            resized_img.close()
    print('完成了！')
    return most_similar_id

def plt_search_result(quary,query_id,query_lib,search_result_dir,thre,interval=5):
    print('绘制余弦相似度与Nupack杂交模拟分数的散点图与小提琴图...')
    query_result = pd.read_hdf(query_lib,key=query_id)
    #绘制散点图
    save_path = os.path.join(search_result_dir,f'{query_id}_Scatter.jpg')
    x = query_result['cos_simi']
    y = query_result['yields']
    plt.xlabel('Cosine similarity')
    plt.ylabel('Nupack Simulated yield')
    plt.title(quary)
    plt.scatter(x, y)
    plt.savefig(save_path)
    plt.close()
    #绘制小提琴图
    violi_save_path = os.path.join(search_result_dir,f'{query_id}_Violi.jpg')
    cos_simi = query_result['cos_simi'].values
    x_ticks = [
               f'<={thre - 2*interval}',
               f'({thre - 2*interval},{thre-interval}]',
               f'({thre-interval},{thre}]',
               f'({thre},{thre+interval}]',
               f'>{thre+interval}'
    ]
    positions = [i for i in range(1, len(x_ticks) + 1)]
    cos_simi_bool_ = [cos_simi<=thre-2*interval,
                      (thre-2*interval < cos_simi) & (cos_simi <= thre-interval),
                      (thre-interval < cos_simi) & (cos_simi <= thre),
                      (thre < cos_simi) & (cos_simi <= thre+interval),
                      thre+interval < cos_simi,
                      ]
    simi_level_lst = [query_result[cos_simi]['yields'].values for cos_simi in cos_simi_bool_]
    # data = {
    #     'x_label':['<20','[20,22)','[22,24)','[24,26)','>=26'],
    #     'data':[simi_1,simi_2,simi_3,simi_4,simi_5]
    # }


    plt.violinplot(simi_level_lst, positions=positions)
    plt.xticks(positions, x_ticks)
    plt.xlabel('Cosine similarity')
    plt.ylabel('Nupack Simulated yield')
    plt.title(quary)
    plt.savefig(violi_save_path)
    plt.close()
    print('完成了！')


#主函数
def main():
    parser = ArgumentParser(description="Search Simulate")

    parser.add_argument('-q', '--query', type=str)
    parser.add_argument('-n', '--encoder_name',type=str)
    parser.add_argument('-t', '--thre',type=int)
    parser.add_argument('--save_dir', type=str, default="/home/cao/桌面/new_similarity_search/simi/Use/Dna_Lib")
    parser.add_argument('--feature_lib', type=str, default="/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/feature.h5")
    parser.add_argument('--img_dir', type=str, default='/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/image')
    parser.add_argument('--encoder_model_dir', type=str, default="/home/cao/桌面/new_similarity_search/simi/model_save/encoder")

    args = parser.parse_args()
    # if len(sys.argv[1:]) == 0:
    #     parser.print_help()
    #     sys.exit(1)
    encoder = Encoder(512, 80)
    feature_lib = args.feature_lib
    img_dir = args.img_dir
    encoder_model_dir = args.encoder_model_dir
    thre = args.thre
    save_dir = args.save_dir
    encoder_name = args.encoder_name
    query = args.query
    query_id = "".join(query.split())[:8]
    save_path = os.path.join(save_dir,encoder_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    encoder_model_param = f"{encoder_model_dir}/{encoder_name}/encoder/encoder_params.pth"
    dna_lib = os.path.join(save_path,'DNA_lib.h5')
    query_lib = os.path.join(save_path,'Query_lib.h5')
    search_result_dir = os.path.join(save_path,query_id)

    if not os.path.exists(dna_lib):
        Image_feature_to_DNA_library(encoder, encoder_model_param, feature_lib, dna_lib)

    quary_to_dna(query, query_id, encoder, encoder_model_param, query_lib, feature_lib)
    search_simu(dna_lib, query_id, query_lib)
    most_similar_id = top_search_image_ids(query_id, query_lib, img_dir, search_result_dir)
    plt_search_result(query, query_id, query_lib, search_result_dir,thre)
    print(most_similar_id)







# def query_search_()
if __name__ == "__main__":
    main()

    quary = "a photo of bear"
    quary_id = "".join(quary.split())[:8]
    encoder = Encoder(512, 80)
    encoder_model_param_path = "/home/cao/桌面/new_similarity_search/simi/model_save/encoder/train_2/encoder/encoder_params.pth"

    feature_lib = "/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/feature.h5"
    dna_lib =  "/home/cao/桌面/new_similarity_search/simi/Use/Dna_Lib/val_data_dna_lib_train_2.h5"
    quary_save_path = "/home/cao/桌面/new_similarity_search/simi/Use/Dna_Lib/query_library_train_2.h5"
    img_dir = '/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/image'
    search_result_dir = f"/home/cao/桌面/new_similarity_search/simi/Use/Dna_Lib/{quary_id}"

    H5_path = "/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/feature.h5"

    DNA_library_save_path = "/home/cao/桌面/new_similarity_search/simi/Use/Dna_Lib/val_data_dna_lib_train_2.h5"

    Image_feature_to_DNA_library(encoder, encoder_model_param_path, H5_path, DNA_library_save_path)


    quary_to_dna(quary, quary_id, encoder, encoder_model_param_path,quary_save_path,feature_lib)
    search_simu(dna_lib,quary_id,quary_save_path)
    most_similar_id = top_search_image_ids(quary_id,quary_save_path,img_dir,search_result_dir)
    plt_search_result(quary, quary_id, quary_save_path, search_result_dir)
    print(most_similar_id)