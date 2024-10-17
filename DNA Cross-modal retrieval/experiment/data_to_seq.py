import clip
import pandas as pd
import torch
from PIL import Image
from models.simulator import simulator
from models.encoder import Encoder,Encoder_copy

def data_trans_seq(data_path):

    device = 'cpu'
    encoder = Encoder(512, 80)
    check_c = torch.load('/home/cao/桌面/new_similarity_search/simi/model_save/encoder_params/encoder_train_unnorm/model_params.pth')
    # check_c = torch.load('/home/cao/桌面/new_similarity_search/simi/model_save/test_params/test_params.pth')
    encoder.load_state_dict(check_c['encoder_model_params'])

    clip_model, preprocess = clip.load("ViT-B/32", device=device)


    if data_path[-3:] == 'jpg':
        img = preprocess(Image.open(data_path)).unsqueeze(0).to(device)
        img_feature = clip_model.encode_image(img)
        encoder.eval()
        with torch.no_grad():
            img_matrix = encoder(img_feature)
            img_seq = encoder.prob_to_seq(img_matrix)
        return img_seq,img_matrix
    else:
        txt =  clip.tokenize([data_path]).to(device)
        txt_feature = clip_model.encode_text(txt)
        # txt_feature = torch.tensor(txt_feature, dtype=torch.float32)
        encoder.eval()
        with torch.no_grad():
            txt_matrix = encoder(txt_feature)
            txt_seq = encoder.prob_to_seq(txt_matrix)
        return txt_seq,txt_matrix

for i in range(4):

    a,b = data_trans_seq('A woman')
    c,d = data_trans_seq('/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/image/000000000785.jpg')
    print(i)
    print(b.dtype)
    print(d.dtype)
    print(b[0][0])
    print(d[0][0])
    seq_pairs = pd.DataFrame({
        'dds':a,
        'dadd':c
    })
    yields = simulator(seq_pairs)
    print(yields)
