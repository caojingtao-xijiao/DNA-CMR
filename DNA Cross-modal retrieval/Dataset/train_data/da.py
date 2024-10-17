
import pandas
import pandas as pd

df = pd.read_hdf('./feature.h5',key='text')
print(df)