import clip
import torch
from PIL import Image
from models.encoder import Encoder
import pandas as pd

def quary_to_dna(quary,quary_id,encoder,encoder_model_param_path,quary_save_path,device='cpu'):
    encoder = encoder.to(device)
    check_c = torch.load(encoder_model_param_path)
    encoder.load_state_dict(check_c['encoder_model_params'])
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    DNA_quary = pd.HDFStore(quary_save_path, complevel=9, mode='a')
    if quary[-3:] == 'jpg':
        img = preprocess(Image.open(quary)).unsqueeze(0).to(device)
        img_feature = clip_model.encode_image(img)
        encoder.eval()
        with torch.no_grad():
            img_seq = encoder.feature_to_seq(img_feature)
        frame = pd.DataFrame(img_seq, index=[quary_id])
        DNA_quary.append('Quary',frame)
        DNA_quary.close()

    else:
        txt = clip.tokenize([quary]).to(device)
        txt_feature = clip_model.encode_text(txt)
        encoder.eval()
        with torch.no_grad():
            txt_seq = encoder.feature_to_seq(txt_feature)
        frame = pd.DataFrame(txt_seq, index=[quary_id])
        DNA_quary.append('Quary', frame)
        DNA_quary.close()


if __name__ == "__main__":
    #文本描述要检索的东西
    quary = "person"
    #给描述赋值id
    quary_id = "".join(quary.split())[:8]

    encoder = Encoder(512,80)
    encoder_model_param_path = "/home/cao/桌面/new_similarity_search/simi/model_save/encoder_params/encoder_train_unnorm/model_params.pth"

    quary_save_path = "/home/cao/桌面/new_similarity_search/simi/Use/Dna_Lib/dna_library.h5"
    quary_to_dna(quary,quary_id,encoder,encoder_model_param_path,quary_save_path)
