#!/bin/bash
#SBATCH -J 25_70
#SBATCH -N 1
#SBATCH --ntasks-per-node=3
#SBATCH -p shuoxing
#SBATCH --gres gpu:1
#SBATCH -o logs/%j_25_70.out
#SBATCH -e logs/%j_25_70.err

#环境
export PATH="/opt/software/anaconda3/envs/sim/bin:$PATH"
source activate sim

thre=26
train_feature_file="/home/cao/桌面/new_similarity_search/simi/Dataset/train_data/train_feature.h5"
train_data_save_path="/home/cao/桌面/new_similarity_search/simi/Dataset/train_data/pair_${thre}_train_simi_data"

val_feature_file="/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/val_feature.h5"
val_data_save_path="/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/pair_${thre}_val_simi_data"


#python ../Data_preprocess/pair_train_data.py \
#-f ${val_feature_file} \
#-s ${val_data_save_path}

python ../Data_preprocess/pair_train_data.py \
-f ${val_feature_file} \
-s ${val_data_save_path} \
-t ${thre}
python ../Data_preprocess/pair_train_data.py \
-f ${train_feature_file} \
-s ${train_data_save_path} \
-t ${thre}


