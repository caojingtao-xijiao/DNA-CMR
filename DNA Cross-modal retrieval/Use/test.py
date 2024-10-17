import os

import pandas as pd
import torch
import numpy as np

import torch

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from Use.Search_simulate import plt_search_result

thre=26
model_name=f"pair_{thre}"
quary = "A woman posing for the camera standing on skis"
query_id = "".join(quary.split())[:8]
save_dir = "/home/cao/桌面/new_similarity_search/simi/Use/Dna_Lib"
save_path = os.path.join(save_dir,model_name)
query_lib = os.path.join(save_path,'Query_lib.h5')
search_result_dir = os.path.join(save_path,query_id)
plt_search_result(quary,
                  query_id,
                  query_lib,
                  search_result_dir,
                  thre)



