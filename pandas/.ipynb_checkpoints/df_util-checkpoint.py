def get_sampled_sets(df_list, strat_cols=[], frac=0.1):
    sample_func = lambda x: x.sample(frac=frac)
    if len(strat_cols) > 0:
        df_list_out = (df.groupby(strat_cols, group_keys=False).apply(sample_func) for df in df_list)
    else:
        df_list_out = (df.sample(frac=frac) for df in df_list)
    return df_list_out

def get_column_types_dict(df):
	g = df.columns.to_series().groupby(df.dtypes).groups
	return {k.name: v for k, v in g.items()}

def upweigh_df(df, col, value, weight):
    """Upweight the dataframe where col == value
    """
    df_extra = df[df[key] == value]
    df_reweighed = df.append([df_extra] * (weight - 1), ignore_index=True)
    return df_reweighed