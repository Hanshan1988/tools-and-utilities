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


def reweigh_by_criteria(data, filter_key, filter_value, weight_name):        
    extra = data[data[filter_key] == filter_value]
    n_times = extra[weight_name].unique()
    if len(n_times) > 1:
        raise ValueError('This function assumes a unique weight in the filtered data.')
        return 
    reweighed = data.append([extra] * (n_times[0] - 1), ignore_index=True)
    reweighed = reweighed.drop([weight_name], axis=1).reset_index(drop=True)
    return reweighed


def reweigh_by_row_weight(data, weight_name):
    expanded = data.loc[data.index.repeat(data[weight_name])].assign()
    reweighed = expanded.drop([weight_name], axis=1).reset_index(drop=True)
    return reweighed