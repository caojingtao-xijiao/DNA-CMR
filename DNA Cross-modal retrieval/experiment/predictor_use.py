import numpy as np
import torch
from models.simulator import simulator
from models.predictor import Hybridization_prediction_CNN
import tools.sequences as seqtools
import pandas as pd
predictor = Hybridization_prediction_CNN()
check = torch.load('../model_save/pre_train_predictor/predictor_trained.pth')
predictor.load_state_dict(check['model_params'])

random_pairs, mut_rates = seqtools.random_mutant_pairs(5000, 80)
seq_pairs = pd.DataFrame({
    "target_features": random_pairs[:, 0],
    "query_features": random_pairs[:, 1],
    "mut_rates":mut_rates
})
yields = torch.tensor(simulator(seq_pairs))



onehot_seq_pairs = predictor.seq_pairs_to_onehots(seq_pairs)
onehot_seq_pairs = torch.tensor(onehot_seq_pairs, dtype=torch.float32)

with torch.no_grad():
    pre = predictor(onehot_seq_pairs)
    pre = np.array(pre).reshape(-1)
difference = np.abs(yields-pre)
yields = pd.DataFrame({'yields': yields})
yields['pre'] = pre
yields['difference'] = difference

print((yields['difference']>0.1).sum())
print((yields['difference']>0.2).sum())
print((yields['difference']>0.3).sum())
print((yields['difference']>0.4).sum())
yields.to_csv('./result.txt')
seq_pairs.to_csv('./seq_p.txt')