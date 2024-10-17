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

#train_feature_file="/home/cao/桌面/new_similarity_search/simi/Dataset/train_data/feature.h5"
#val_feature_file="/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/feature.h5"
#train_data_save_path="/home/cao/桌面/new_similarity_search/simi/Dataset/train_data/${train_data_name}"
#val_data_save_path="/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/${val_data_name}"
#data_name='thre_25'
thre=25
model_name="pair_${thre}"
train_data_name="${model_name}_train_data"
val_data_name="${model_name}_val_data"

train_feature_file="/home/cao/桌面/new_similarity_search/simi/Dataset/train_data/train_feature.h5"
train_data_save_path="/home/cao/桌面/new_similarity_search/simi/Dataset/train_data/${train_data_name}"

val_feature_file="/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/val_feature.h5"
val_data_save_path="/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/${val_data_name}"


python ../Data_preprocess/pair_train_data.py \
-f ${val_feature_file} \
-s ${val_data_save_path} \
-t ${thre} \
--img_batch_size 500 \
--txt_batch_size 500 \
--per_img_count_txt 10 \
--per_txt_count_img 3 \
--sample_txt_batch_size 25000


python ../Data_preprocess/pair_train_data.py \
-f ${train_feature_file} \
-s ${train_data_save_path} \
-t ${thre} \
--img_batch_size 500 \
--txt_batch_size 500 \
--per_img_count_txt 10 \
--per_txt_count_img 3 \
--sample_txt_batch_size 200000

#step_2 train_encoder
python ../train/train_encoder.py -m ${model_name} -t ${train_data_name} -v ${val_data_name}
#step_3 test_encoder
python ../Use/Search_simulate.py -q "A woman posing for the camera standing on skis" -n ${model_name} -t ${thre}
python ../Use/Search_simulate.py -q "A large white bowl of many green apples." -n ${model_name} -t ${thre}
python ../Use/Search_simulate.py -q "A city bus drives through a city area" -n ${model_name} -t ${thre}
python ../Use/Search_simulate.py -q "A big burly grizzly bear is show with grass in the background." -n ${model_name} -t ${thre}
python ../Use/Search_simulate.py -q "A person on skis makes her way through the snow" -n ${model_name} -t ${thre}
python ../Use/Search_simulate.py -q "A train traveling through rural countryside lined with trees." -n ${model_name} -t ${thre}
python ../Use/Search_simulate.py -q "A group of giraffes stand together feeding on leaves in a large park." -n ${model_name} -t ${thre}
python ../Use/Search_simulate.py -q "A pizza with Canadian bacon and pineapple in a fluted pan." -n ${model_name} -t ${thre}
python ../Use/Search_simulate.py -q "A red stop sign sitting next to a forest." -n ${model_name} -t ${thre}
python ../Use/Search_simulate.py -q "A couple of zebra eating a small pile of hay." -n ${model_name} -t ${thre}
python ../Use/Search_simulate.py -q "A table topped with a white plate covered in three donuts." -n ${model_name} -t ${thre}
python ../Use/Search_simulate.py -q "A large aircraft in the blue sky by itself" -n ${model_name} -t ${thre}
