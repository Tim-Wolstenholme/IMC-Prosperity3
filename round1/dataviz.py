import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
df = pd.read_csv("data//prices_round_1_day_0.csv",sep=";")
df = pd.DataFrame(df)
df_kelp = df.groupby("product").get_group("KELP")[["timestamp","mid_price"]]
df_resin = df.groupby("product").get_group("RAINFOREST_RESIN")[["timestamp","mid_price"]]
df_ink = df.groupby("product").get_group("SQUID_INK")[["timestamp","mid_price"]]
df2 = pd.read_csv("data//prices_round_1_day_-1.csv",sep=";")
df2 = pd.DataFrame(df2)
df_ink_2 = df2.groupby("product").get_group("SQUID_INK")[["timestamp","mid_price"]]
df3 = pd.read_csv("data//prices_round_1_day_-2.csv",sep=";")
df3 = pd.DataFrame(df3)
df_ink_3 = df3.groupby("product").get_group("SQUID_INK")[["timestamp","mid_price"]]
#Calculating realised volatility
ink_data1 = df_ink["mid_price"].to_numpy()
ink_data2 = df_ink_2["mid_price"].to_numpy()
ink_data3 = df_ink_3["mid_price"].to_numpy()
print(min(ink_data2))
print(max(np.diff(ink_data1)))
print(max(np.diff(ink_data3)))
ink_data = np.concatenate((ink_data3,ink_data2,ink_data1),axis=None)
max_diff = max(np.diff(ink_data))
for i,e in enumerate(np.diff(ink_data)):
    if e == max_diff:
        print(i)
plt.plot(ink_data)
plt.show()
