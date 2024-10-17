import os.path
from argparse import ArgumentParser
import sys

import numpy as np

sys.path.append('/home/cao/桌面/new_similarity_search/simi')
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.encoder import Encoder
from models.predictor import Local_layer_predictor
from models.encoder_trainer import EncoderTrainer
from utils.Dataset_set import Simi_dataset,Non_simi_dataset
from train.train_predictor import train_predictor

#对softmax的输出做出惩罚
def encoder_entropy(seq_probs,strength=0.01):
    #seq_probs:(80,4)
    # seq_probs.shape.assert_is_compatible_with([None, None, 4])
    # Adding a little epsilon (1e-10) so we never take the log of zero (good catch, callie!)
    ent_by_position = -torch.sum(
        seq_probs * torch.log(seq_probs + 1e-10),
        dim = 2
    )
    #
    mean_ent_by_sequence = torch.mean(
        ent_by_position,
        dim = 1
    )
    mean_ent_by_batch = torch.mean(
        mean_ent_by_sequence,
        dim = 0
    )
    return strength * mean_ent_by_batch


def train_encoder(
    encoder_model,
    predictor_model,
    save_dir,
    train_data_name,
    val_data_name,
        thre,
    encoder_data_dir='../Dataset',
    encoder_train_batch_size=200,
    encoder_val_batch_size=10000,
    epochs=100,
    is_use_gpu=True,
    predictor_pairs_num=500,
    predict_params_path='../model_save/pre_train_predictor_local/predictor_trained.pth',
    refit_every=1,
    refit_epochs=10,
):

        train_data_path = os.path.join(encoder_data_dir,f'train_data/{train_data_name}')
        train_data_feature = os.path.join(encoder_data_dir,'train_data/train_feature.h5')
        val_data_path = os.path.join(encoder_data_dir,f'val_data/{val_data_name}')
        val_data_feature = os.path.join(encoder_data_dir, 'val_data/val_feature.h5')
        # save_dir
        predcitor_save_dir = os.path.join(save_dir,'predictor')
        encoder_save_dir = os.path.join(save_dir,'encoder')

        if not os.path.exists(predcitor_save_dir):
            os.makedirs(predcitor_save_dir)
        if not os.path.exists(encoder_save_dir):
            os.makedirs(encoder_save_dir)
        #encoder
        encoder_loss_jpg_save_path = os.path.join(encoder_save_dir, 'encoder_loss.jpg')
        pd_encoder_loss_save_path = os.path.join(encoder_save_dir,'encoder_loss.xlsx')
        encoder_param_path = os.path.join(encoder_save_dir, 'encoder_params.pth')


        enocder_train_simi_dataset = Simi_dataset(val_data_path)
        enocder_train_non_simi_dataset = Non_simi_dataset(val_data_feature,len(enocder_train_simi_dataset),thre)

        enocder_train_simi_dataloader = DataLoader(enocder_train_simi_dataset,batch_size=int(encoder_train_batch_size/2))
        enocder_train_non_simi_dataloader = DataLoader(enocder_train_non_simi_dataset,batch_size=int(encoder_train_batch_size/2))

        enocder_val_simi_dataset = Simi_dataset(train_data_path)
        enocder_val_non_simi_dataset = Non_simi_dataset(train_data_feature, len(enocder_val_simi_dataset), thre)

        enocder_val_simi_dataloader = DataLoader(enocder_val_simi_dataset,
                                                   batch_size=int(encoder_val_batch_size / 2))
        enocder_val_non_simi_dataloader = DataLoader(enocder_val_non_simi_dataset,
                                                       batch_size=int(encoder_val_batch_size / 2))


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if is_use_gpu:
            if torch.cuda.is_available():
                print('cuda')
            else:
                exit('cuda is None')
        else:
            device = torch.device('cpu')
        encoder_model = encoder_model.to(device)
        predictor_model = predictor_model.to(device)
        encoder_trainer_model = EncoderTrainer(encoder_model, predictor_model)
        encoder_trainer_model = encoder_trainer_model.to(device)

        check_c = torch.load(predict_params_path)
        encoder_trainer_model.predictor.load_state_dict(check_c['model_params'])

        #定义优化器与损失函数
        optimizer = torch.optim.Adagrad(encoder_trainer_model.encoder.parameters(),1e-3)
        loss_fun = nn.BCELoss()
        pd_encoder_loss = pd.DataFrame(columns=["encoder_train_loss", "encoder_val_loss"])

        for epoch in range(1,epochs+1):
            encoder_trainer_model.train()
            encoder_trainer_model.predictor_trainable(encoder_trainer_model.predictor,False)
            encoder_train_loss_sum = 0
            encoder_train_step_sum = 0
            train_encoder_loop = tqdm(enocder_train_simi_dataloader,desc=f'Encoder Train Epoch [{epoch}/{epochs}]',ncols=150)
            for train_simi_data,train_simi_label in train_encoder_loop:
                train_non_simi_data,train_non_simi_label = next(iter(enocder_train_non_simi_dataloader))
                train_batch_data = torch.cat([train_simi_data,train_non_simi_data],dim=0)
                train_label = torch.cat([train_simi_label,train_non_simi_label],dim=0)
                train_batch_data = train_batch_data.to(device)
                train_label = train_label.to(device)
                optimizer.zero_grad()
                pre,s1,s2 = encoder_trainer_model(train_batch_data)
                train_loss = loss_fun(pre, train_label)
                s1_reg = encoder_entropy(s1)
                s2_reg = encoder_entropy(s2)
                train_loss += s1_reg
                train_loss += s2_reg
                train_loss.backward()
                optimizer.step()
                encoder_train_step_sum += 1
                encoder_train_loss_sum += train_loss.item()
                train_encoder_loop.set_postfix({'encoder_train_loss':train_loss.item(),'average_loss':encoder_train_loss_sum/encoder_train_step_sum})
            train_encoder_loop.close()

            encoder_trainer_model.eval()
            encoder_val_loss_sum = 0
            encoder_val_step_sum = 0
            val_encoder_loop = tqdm(enocder_val_simi_dataloader, desc=f'Encoder Val Epoch [{epoch}/{epochs}]', ncols=150)
            for val_simi_data,val_simi_label in val_encoder_loop:
                with torch.no_grad():
                    val_non_simi_data,val_non_simi_label = next(iter(enocder_val_non_simi_dataloader))
                    val_batch_data = torch.cat([val_simi_data,val_non_simi_data])
                    val_label = torch.cat([val_simi_label,val_non_simi_label])
                    val_batch_data = val_batch_data.to(device)
                    val_label = val_label.to(device)
                    pre,S1,S2 = encoder_trainer_model(val_batch_data)
                    val_loss = loss_fun(pre, val_label)
                    s1_reg = encoder_entropy(S1)
                    s2_reg = encoder_entropy(S2)
                    val_loss += s1_reg
                    val_loss += s2_reg
                encoder_val_step_sum += 1
                encoder_val_loss_sum += val_loss.item()


                val_encoder_loop.set_postfix(
                    {'encoder_val_loss': val_loss.item(), 'average_loss': encoder_val_loss_sum / encoder_val_step_sum})
            val_encoder_loop.close()

            en_tr_loss_per_epo = encoder_train_loss_sum/encoder_train_step_sum
            en_va_loss_per_epo = encoder_val_loss_sum / encoder_val_step_sum

            info = (en_tr_loss_per_epo, en_va_loss_per_epo)
            pd_encoder_loss.loc[epoch-1] = info


            if epoch % refit_every == 0:
                simi_data,simi_label = next(iter(enocder_train_simi_dataloader))
                non_simi_data,non_simi_label = next(iter(enocder_train_non_simi_dataloader))

                simi_data = simi_data[:int(predictor_pairs_num/2)]
                simi_label = simi_label[:int(predictor_pairs_num/2)]
                non_simi_data = non_simi_data[:int(predictor_pairs_num/2)]
                non_simi_label = non_simi_label[:int(predictor_pairs_num/2)]

                encoder_trainer_model.predictor_trainable(encoder_trainer_model.predictor, True)

                pre_train_data_epoch = torch.cat([simi_data,non_simi_data],dim=0).to(device)


                seq_0 = encoder_trainer_model.encoder.feature_to_seq(pre_train_data_epoch[:, 0, :].clone().detach())
                seq_1 = encoder_trainer_model.encoder.feature_to_seq(pre_train_data_epoch[:, 1, :].clone().detach())
                seq_pairs = pd.DataFrame({
                    "target_features": seq_0,
                    "query_features": seq_1,
                })
                train_predictor(encoder_trainer_model.predictor,
                                predcitor_save_dir,
                                seq_pairs=seq_pairs,
                                epochs=refit_epochs
                                )
                encoder_trainer_model.predictor_trainable(encoder_trainer_model.predictor, False)

            encoder_state_dict = {'model_params': encoder_trainer_model.encoder.state_dict()}
            torch.save(encoder_state_dict, encoder_param_path)

        pd_encoder_loss.to_csv(pd_encoder_loss_save_path)

        x_en = range(len(pd_encoder_loss))
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title('Encoder_loss')
        plt.plot(x_en, pd_encoder_loss["encoder_train_loss"], color='blue',label="train_loss")
        plt.plot(x_en, pd_encoder_loss["encoder_val_loss"], color='red',label="val_loss")
        plt.legend()
        plt.savefig(encoder_loss_jpg_save_path)
        plt.close()


def main():
    parser = ArgumentParser(description="Train Encoder")
    parser.add_argument('-m', '--model_name', type=str)
    parser.add_argument('-t', '--train_data_name', type=str)
    parser.add_argument('-v', '--val_data_name', type=str)
    parser.add_argument('--thre', type=int)
    args = parser.parse_args()
    model_name = args.model_name
    train_data_name = args.train_data_name
    val_data_name = args.val_data_name
    thre = args.thre
    save_dir = os.path.join('../model_save/encoder',model_name)
    encoder_model = Encoder()
    predictor_model = Local_layer_predictor()
    train_encoder(encoder_model,predictor_model,save_dir,train_data_name,val_data_name,thre)

if __name__ == "__main__":
    main()