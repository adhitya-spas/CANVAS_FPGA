import pandas as pd

hex_str = "0000000000A79FC0"
print("The given hex string is ")
print(hex_str)
res = int(hex_str,16)
print("The resultant integer is ")
print(res)
pd.read_table('./Python_Results/high-high_5deg_03khz_channel_i_avg_hex.txt', header=None, delimiter=None)