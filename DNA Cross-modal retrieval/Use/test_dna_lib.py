import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def plt_search_result(query_id,query_lib):
    query_result = pd.read_hdf(query_lib,key=query_id)
    x = query_result['cos_simi']
    y = query_result['yields']
    plt.scatter(x, y)
    plt.show()
quary = "person"
quary_id = "".join(quary.split())[:8]
lib = pd.read_hdf("/home/cao/桌面/new_similarity_search/simi/Use/Dna_Lib/dna_library.h5",key="DNA_library")
quary = pd.read_hdf("/home/cao/桌面/new_similarity_search/simi/Use/Dna_Lib/query_library.h5",key="Quary_DNA")
quary_cos = pd.read_hdf("/home/cao/桌面/new_similarity_search/simi/Use/Dna_Lib/query_library.h5",key="person")
# fea = pd.read_hdf("/home/cao/桌面/new_similarity_search/simi/Dataset/val_data/feature.h5",key="image")
# result = pd.read_hdf("/home/cao/桌面/new_similarity_search/simi/Use/Dna_Lib/rearch_result_1.h5",key="search")
# result_2 = pd.read_hdf("/home/cao/桌面/new_similarity_search/simi/Use/Dna_Lib/rearch_result_2.h5",key="search")
print(lib)
print(quary)
print(quary_cos)
print(quary_cos['cos_simi'].mean())
print(quary_cos['cos_simi'].max())
print(quary_cos['cos_simi'].min())
plt_search_result(quary_id,"/home/cao/桌面/new_similarity_search/simi/Use/Dna_Lib/query_library.h5")
# print(fea)
# print(result)
# print(result_2)
# di = dict()
# res = result_2[result_2[0]>0.85]
# for i in res.index:
#     di[i] = res.loc[i].values
# print(di)
# result_2 = result.sort_values(by=0,axis=0,ascending=False)
# save_pd = pd.HDFStore("/home/cao/桌面/new_similarity_search/simi/Use/Dna_Lib/rearch_result_2.h5",complevel=9, mode='w')
# save_pd.append('search',result_2)
# save_pd.close()
# quary_seq=np.array(["asd"])
# quary = np.repeat([quary_seq],10,axis=0)
# print(quary)
# quary = "people"
# quary_id = "".join(quary.split())[:8]
# car_se = pd.read_hdf(f"/home/cao/桌面/new_similarity_search/simi/Use/Dna_Lib/{quary_id}_rearch_result.h5",key='search')
# di = dict()
# high = car_se[car_se[0] > 0.97]
# for i in high.index:
#     di[i] = high.loc[i].values
# print(di)