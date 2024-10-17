import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import dataset,DataLoader
import torch
from torch import nn
from tqdm import tqdm
from models.simulator import simulator
from models.predictor import Local_layer_predictor



def train_predictor(
        predictor_model,
        model_result_save_psth,
        seq_pairs=None,
        len_of_seq=80,
        num_of_seq=5000,
        is_use_gpu=True,
        epochs=15,
        train_percentage=0.8,
        pre_train_batch_size=32,
        pre_val_batch_size=1000,
        pre_data_set='../Dataset/predictor_pretrain_data'
                    ):
    #实例化模型

    if not os.path.exists(model_result_save_psth):
        os.makedirs(model_result_save_psth)

    loss_pd_save_path = os.path.join(model_result_save_psth,'loss.xlsx')
    model_params_save_path = os.path.join(model_result_save_psth,'predictor_trained.pth')
    predictor_loss_jpg_save_path = os.path.join(model_result_save_psth,'predictor_loss.jpg')
    simu_jpg_save_path = os.path.join(model_result_save_psth,'simu.jpg')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if is_use_gpu:
        if torch.cuda.is_available():
            print('cuda')
        else:
            exit('cuda is None')
    else:
        device = torch.device('cpu')
    predictor_model = predictor_model.to(device)



    if seq_pairs is None:
        print('Pretraining')
        if os.path.exists(pre_data_set):
            print('Training with old simulated data')
            seq_pairs = pd.read_csv(os.path.join(pre_data_set, "seq.txt"))
            yield_pd = pd.read_csv(os.path.join(pre_data_set, 'yields.txt'))
            yields = yield_pd['yields'].values
            yield_value = torch.tensor(yields, dtype=torch.float32)
        else:
            print('Training with new simulated data')
            print('Generating simulated data ...')
            random_pairs, mut_rates = predictor_model.random_mutant_pairs(num_of_seq, len_of_seq)
            seq_pairs = pd.DataFrame({
                "target_features": random_pairs[:, 0],
                "query_features": random_pairs[:, 1],
                "mut_rates": mut_rates
            })
            yields = simulator(seq_pairs)
            yield_pd = pd.DataFrame({'yields': yields})
            yield_pd.to_csv(os.path.join(pre_data_set, 'yields.txt'))
            seq_pairs.to_csv(os.path.join(pre_data_set, "seq.txt"))
            yield_value = torch.tensor(yields, dtype=torch.float32)
    else:
        print('Retraining')
        yields = simulator(seq_pairs)
        yield_value = torch.tensor(yields, dtype=torch.float32)
        num_of_seq = len(seq_pairs)


    onehot_pairs = torch.transpose(torch.tensor(predictor_model.seq_pairs_to_onehots(seq_pairs), dtype=torch.float), 1, 3)

    num_of_train = int(num_of_seq * train_percentage)

    pre_train_dataset = dataset.TensorDataset(onehot_pairs[:num_of_train], yield_value[:num_of_train])
    pre_train_dataloader = DataLoader(pre_train_dataset, batch_size=pre_train_batch_size, shuffle=False)
    pre_val_dataset = dataset.TensorDataset(onehot_pairs[num_of_train:], yield_value[num_of_train:])
    pre_val_dataloader = DataLoader(pre_val_dataset, batch_size=pre_val_batch_size, shuffle=False)

    loss_pd = pd.DataFrame(columns=['train_loss', 'val_loss'])
    optimizer = torch.optim.RMSprop(predictor_model.parameters(), 1e-3)
    loss_fun = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        predictor_model.train()
        train_loss_sum = 0
        train_step = 0
        train_loop = tqdm(pre_train_dataloader, desc=f'Train epoch [{epoch}/{epochs}]', total=len(pre_train_dataloader), ncols=150)
        for batch_train_data, batch_label in train_loop:
            batch_train_data = batch_train_data.to(device)
            batch_label = batch_label.to(device)
            optimizer.zero_grad()
            pre = predictor_model(batch_train_data)
            loss = loss_fun(pre, batch_label)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_step += 1
            train_loop.set_postfix({'real_time_loss': loss.item(), 'average_loss': float(train_loss_sum / train_step)})
        train_loop.close()

        predictor_model.eval()
        val_loss_sum = 0
        val_step = 0
        val_loop = tqdm(pre_val_dataloader, desc=f'Val epoch [{epoch}/{epochs}]', total=len(pre_val_dataloader), ncols=150)
        for batch_val_data, val_label in  val_loop:
            val_step += 1
            with torch.no_grad():
                batch_val_data = batch_val_data.to(device)
                val_label = val_label.to(device)
                val_pre = predictor_model(batch_val_data)
                val_loss = loss_fun(val_pre, val_label)
                val_loss_sum += val_loss.item()
                val_loop.set_postfix(
                    {'real_time_loss': val_loss.item(), 'average_loss': float(val_loss_sum / val_step)})
        val_loop.close()

        train_loss_per_epoch = train_loss_sum / train_step
        val_loss_per_epoch = val_loss_sum / val_step

        info = (train_loss_per_epoch, val_loss_per_epoch)
        loss_pd.loc[epoch-1] = info

    predict_state = predictor_model.state_dict()

    save_dict = {'model_params': predict_state}

    torch.save(save_dict, model_params_save_path)
    loss_pd.to_excel(loss_pd_save_path)

    with torch.no_grad():
        onehot_seq_pairs = onehot_pairs.to(device)
        pred_yield = predictor_model(onehot_seq_pairs).cpu().numpy()
    try:
        mut_rates = seq_pairs['mut_rates'].values
        plt.scatter(mut_rates, yields, c=np.abs(pred_yield - yields))
        plt.xlabel("Hamming distance")
        plt.ylabel("Simulated yield")
        plt.colorbar(label="|Simulated - Predicted|")
        plt.savefig(simu_jpg_save_path)
        plt.close()
    except:
        print('Retraining does not generate simulation graphs')
    x = range(epochs)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title('Predictor_loss')
    plt.plot(x, loss_pd["train_loss"], color='blue', label='train_loss')
    plt.plot(x, loss_pd["val_loss"], color='yellow', label='val_loss')
    plt.legend()
    plt.savefig(predictor_loss_jpg_save_path)
    plt.close()

def main():
    untrained_predictor = Local_layer_predictor()
    train_predictor(
        untrained_predictor,
        '/home/cao/桌面/new_similarity_search/simi/model_save/pre_train_predictor_local',
    )

if __name__ == '__main__':
    main()



