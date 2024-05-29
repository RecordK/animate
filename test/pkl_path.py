import pandas as pd

pkl_file_path = "J.video.pkl"

data = pd.read_pickle(pkl_file_path)

print(data.iloc[0])