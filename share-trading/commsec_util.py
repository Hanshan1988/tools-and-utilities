import pandas as pd
import numpy as np

top_n = 10

def convert_buy_sell_prices(df):
	# assumes df in the shape of one row per snapshot in time with 10 buy prices and 10 sell prices
	df_sub = df[df.asx_state == 1].reset_index(drop=True)
	df_sub_samp = df_sub.sample(n=1)

	buy_price_cols = ['buy_price_{0}'.format(i) for i in range(top_n)]
	sell_price_cols = ['sell_price_{0}'.format(i) for i in range(top_n)]
	buy_vol_cols = ['buy_volume_{0}'.format(i) for i in range(top_n)]
	sell_vol_cols = ['sell_volume_{0}'.format(i) for i in range(top_n)]

	buy_price_values = df_sub_samp[buy_price_cols].values[0]
	sell_price_values = df_sub_samp[sell_price_cols].values[0]
	buy_vol_values = df_sub_samp[buy_vol_cols].values[0]
	sell_vol_values = df_sub_samp[sell_vol_cols].values[0]

	return buy_price_values, buy_vol_values, sell_price_values, sell_vol_values
