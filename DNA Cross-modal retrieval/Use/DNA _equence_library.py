import torch
import pandas as pd
from models.encoder import Encoder
def Image_feature_to_DNA_library(encoder_model,encoder_model_param_path,H5_path,DNA_library_save_path,batch_size=1000,device='cuda'):
    encoder_model = encoder_model.to(device)
    check_c = torch.load(encoder_model_param_path)
    encoder_model.load_state_dict(check_c['model_params'])
    feature = pd.read_hdf(H5_path,key='image')
    DNA_library = pd.HDFStore(DNA_library_save_path, complevel=9, mode='a')

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

if __name__ == "__main__":
    encoder = Encoder(512,80)
    encoder_model_param_path = "/home/cao/桌面/new_similarity_search/simi/model_save/encoder/train_2/encoder/encoder_params.pth"

    H5_path = "/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/feature.h5"

    DNA_library_save_path = "/home/cao/桌面/new_similarity_search/simi/Use/Dna_Lib/val_data_dna_lib_train_2.h5"
    Image_feature_to_DNA_library(encoder, encoder_model_param_path, H5_path, DNA_library_save_path)




    #




