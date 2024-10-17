import numpy as np
import torch.nn as nn
import torch
from params import share_param


class LocalInteractionsLayer(nn.Module):
    def __init__(self,window_size=1):
        super(LocalInteractionsLayer, self).__init__()
        self.window_size = window_size
    def forward(self, seq_pairs):
        batch_size, channels, seq_len, features = seq_pairs.size()
        by_position = []
        for pos in range(self.window_size, seq_len - self.window_size):
            by_channel = []
            for channel in range(channels):
                top = seq_pairs[:, channel:channel + 1, :, 0][:, :, pos - self.window_size:pos + self.window_size + 1]
                bot = seq_pairs[:, channel:channel + 1, :, 1][:, :, pos - self.window_size:pos + self.window_size + 1]
                top = top.permute(0, 2, 1)
                mat = torch.matmul(top, bot)
                by_channel.append(mat.view(-1, ((self.window_size * 2) + 1) ** 2))
            by_channel = torch.cat(by_channel, dim=1)
            # print(by_channel.shape)
            by_position.append(by_channel)
        by_position = torch.stack(by_position, dim=1)
        return by_position

class Local_layer_predictor(nn.Module):
    def __init__(self,window_size=share_param.window_size,DNA_len=share_param.DNA_len):
        super(Local_layer_predictor,self).__init__()
        self.local_layer = LocalInteractionsLayer(window_size)
        self.infe= DNA_len-2*window_size
        self.predictor_model = nn.Sequential(
            self.local_layer,
            nn.AvgPool1d(kernel_size=3),
            nn.Conv1d(in_channels=self.infe,out_channels=36,kernel_size=3),
            nn.Tanh(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(36,1),
            nn.Sigmoid(),
            nn.Flatten(0)
        )
        self.bases = np.array(list("ATCG"))
        self.randseq = lambda n: "".join(np.random.choice(self.bases, n))
    #(b,4,80,2) --> (b,)
    def forward(self,input):
        return self.predictor_model(input)

    # (b,DNA_seq,4) --> (b,2,DNA_seq,4)
    def seq_pairs_to_onehots(self, seq_pairs):
        onehot_pairs = np.stack([
            self.seqs_to_onehots(seq_pairs.target_features.values),
            self.seqs_to_onehots(seq_pairs.query_features.values)
        ], axis = 1)
        return onehot_pairs
    # DNA序列 --> 独热码
    # (b,1) --> (b,DNA_seq,4)
    def seqs_to_onehots(self,seqs):
        seq_array = np.array(list(map(list, seqs)))
        return np.array([(seq_array == b).T for b in self.bases]).T.astype(float)

    def random_mutant_pairs(self,n, d):
        targets = [self.randseq(d) for _ in range(n)]
        mut_rates = np.random.uniform(0, 1, size=n)

        pairs = np.array(
            [[target, self.mutate(target, rate)]
             for target, rate in zip(targets, mut_rates)
             ]
        )
        seq_hdists = np.array(
            [self.seq_hdist(s1, s2) for s1, s2 in pairs]
        )

        return pairs, seq_hdists

    def mutate(self,seq, mut_rate=0.5):
        seq_list = list(seq)
        for i, b in enumerate(seq_list):
            if np.random.random() < mut_rate:
                seq_list[i] = np.random.choice([base for base in self.bases if base != b])
        return "".join(seq_list)

    def seq_hdist(self,s1, s2):
        return np.mean(np.array(list(s1)) != np.array(list(s2)))

#
# deivce = torch.device('cuda')
# A = Local_layer_predictor().to(deivce)
# b = torch.randn(2,4,80,2).to(deivce)
# print(A(b))














# class Hybridization_prediction_CNN(nn.Module):
#     def __init__(self,DNA_len=80):
#         super(Hybridization_prediction_CNN,self).__init__()
#         self.predictor_model = nn.Sequential(
#             nn.Conv2d(2, 4, (1, 4), 1),
#             nn.Tanh(),
#             nn.AvgPool2d((2,1)),
#             nn.Conv2d(4,8,(3,1),1),
#             nn.Tanh(),
#             nn.AvgPool2d((2,1)),
#             nn.Flatten(),
#             nn.Linear(152,1),
#             nn.Sigmoid(),
#             nn.Flatten(0)
#         )
#         self.initialize()
#         self.bases = np.array(list("ATCG"))
#     def forward(self,x):
#         return self.predictor_model(x)
#     def initialize(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#             elif isinstance(m, nn.Conv2d):
#                 nn.init.xavier_normal_(m.weight)
#
#     def seq_pairs_to_onehots(self, seq_pairs):
#         onehot_pairs = np.stack([
#             self.seqs_to_onehots(seq_pairs.target_features.values),
#
#             self.seqs_to_onehots(seq_pairs.query_features.values)
#         ], axis = 1)
#         return onehot_pairs
#
#     def seqs_to_onehots(self,seqs):
#         seq_array = np.array(list(map(list, seqs)))
#         return np.array([(seq_array == b).T for b in self.bases]).T.astype(float)







# device = 'cuda'
# predictor = Local_layer_predictor().to(device)
# batch_train_data = torch.randn(2,4,80,2).to(device)
# pre = predictor(batch_train_data)

