#!/bin/bash
#SBATCH -J model_10
#SBATCH -N 1
#SBATCH --ntasks-per-node=3
#SBATCH -p shuoxing
#SBATCH --gres gpu:1
#SBATCH -o logs/%j_model_10_train_1.out
#SBATCH -e logs/%j_model_10_train_1.err

#环境
export PATH="/opt/software/anaconda3/envs/sim/bin:$PATH"
source activate sim

thre=26
model_name="pair_${thre}"
#python ../Use/Search_simulate.py -q "A woman posing for the camera standing on skis" -n ${model_name} -t ${thre}
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

