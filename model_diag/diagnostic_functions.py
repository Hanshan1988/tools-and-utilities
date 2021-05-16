import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    
# calculate gains
def CalculateGains(indata = [], target_name = "", pred_name = "", exposure_weights = "", 
                   frequency_variable = "", positive_target_only = False, plot = False):
    
    # error checking
    if len(indata) == 0 or target_name == "" or pred_name == "":
        print("ERROR: indata, name of target variable and name of prediction must be specified. No default value.")
        return
    
    # calculate weights
    if exposure_weights in indata.columns:
        exposure_w = indata[exposure_weights]
    else:
        exposure_w = np.tile(1, len(indata))
    
    if frequency_variable in indata.columns:
        frequency_w = indata[frequency_variable]
    else:
        frequency_w = np.tile(1, len(indata))
    
    weights = np.multiply(exposure_w, frequency_w)
    target = indata[target_name].values
    pred = indata[pred_name].values
    
    # remove records with missing targets
    target_orig = target
    pred_orig = pred
    weights_orig = weights
    
    target = target_orig[~np.isnan(target_orig)]
    pred = pred_orig[~np.isnan(target_orig)]
    weights = weights_orig[~np.isnan(target_orig)]
    
    if positive_target_only:
        pred = pred[target > 0]
        weights = weights[target > 0]
        target = target[target > 0]
    
    nobs = len(target)
    
    # sort target
    target_ord = pd.Series([x for _,x in sorted(zip(target, target), reverse = True)]).values
    target_pred_ord = pd.Series([x for _,x in sorted(zip(pred, target), reverse = True)]).values
    
    # sort weights
    weights_ord = pd.Series([x for _,x in sorted(zip(target, weights), reverse = True)]).values
    weights_pred_ord = pd.Series([x for _,x in sorted(zip(pred, weights), reverse = True)]).values
    
    perc = np.cumsum(weights_ord)/np.sum(weights)
    perc_pred = np.cumsum(weights_pred_ord)/np.sum(weights)
    
    # Next get cumulative proportion of total
    target_cumsum = np.cumsum(np.multiply(target_ord, weights_ord))/np.sum(np.multiply(target_ord, weights_ord))
    target_pred_cumsum = np.cumsum(np.multiply(target_pred_ord, weights_pred_ord))/np.sum(np.multiply(target_pred_ord, weights_pred_ord))
    
    ## Create lagged versions for differencing
    target_cumsum_lag = np.append(0, target_cumsum[0:(nobs-1)])
    target_pred_cumsum_lag = np.append(0, target_pred_cumsum[0:(nobs-1)])
    
    perc_lag = np.append(0, perc[0:(nobs-1)])
    perc_pred_lag = np.append(0, perc_pred[0:(nobs-1)])

    # calculate area under gains curve
    max_gains = np.sum((perc - perc_lag) * (target_cumsum + target_cumsum_lag)/2) - 0.5
    model_gains = np.sum((perc_pred - perc_pred_lag) * (target_pred_cumsum + target_pred_cumsum_lag)/2) - 0.5
    
    # percentage of theoretical max gains
    gains = model_gains/max_gains
    
    if plot:
        import matplotlib.pyplot as plt
        plt.subplots()
    
        t, = plt.plot(np.append(0, perc_pred), np.append(0, target_pred_cumsum), '-', color = 'blue', linewidth = 2, label = "t")
        p, = plt.plot(np.append(0, perc), np.append(0, target_cumsum), '-', color = 'green', linewidth = 2, label = "p")
        r, = plt.plot([0, 1], [0, 1], '-', color = 'grey', linewidth = 2, label = "r")
        
        plt.xlabel("Cumulative proportion of population", fontsize = 12)
        plt.ylabel("Gains", fontsize = 12)
        plt.title("Gains chart", fontsize = 14)
        
        plt.legend([t, p, r], ["Model Gains", "Theoretical Max Gains", "Random Gains"], fontsize = 12)
    
    return gains

# actual vs predicted by predicted band
def AvsEPred(target, pred, groups = 20, model_name = "", save_plot = False, save_path = "", auc = None, save_name = None):

    target_ord = pd.Series([x for _,x in sorted(zip(pred,target))])
    pred_ord = pd.Series([x for _,x in sorted(zip(pred,pred))])
    
    percentile_size = len(target_ord)/groups
    rank = np.linspace(1, len(target_ord), len(target_ord))
    perc_bands = np.ceil(rank/percentile_size)/groups
    
    target_avg = np.array(target_ord.groupby(perc_bands).mean())
    pred_avg = np.array(pred_ord.groupby(perc_bands).mean())
    x_axis = np.unique(perc_bands)
    
    plt.subplots()
    
    t, = plt.plot(x_axis, target_avg, '-', color = 'blue', linewidth = 2, label = "t")
    p, = plt.plot(x_axis, pred_avg, '-', color = 'green', linewidth = 2, label = "p")
    
    plt.xlabel("Predicted band", fontsize = 12)
    plt.ylabel("Target", fontsize = 12)
    
    if isinstance(auc, float) or isinstance(auc, int):
        plt.title(("Actual vs predicted by predicted band\nModel: " + str(model_name) 
                   + "\nAUC: {0:.2f}".format(auc)), fontsize = 14)
    else:
        plt.title(("Actual vs predicted by predicted band\nModel: " + str(model_name)), fontsize = 14)
    
    plt.legend([t, p], ["Actual", "Predicted"], fontsize = 12)
    
    if save_plot:
        if not save_name:
            save_name = model_name
        plt.savefig(save_path + "plot_AvsEPred_" + str(save_name) + ".png")


# actual vs expected by predictor
def AvsE(indata = [], target_name = "", pred_name = "", predictor_name = "", model_name = "", 
         exposure_weights = "", frequency_variable = "", graph_groups = 20, equal_size_or_distance = "distance",
         cup = 0, cap = 0, num_as_cat = False, fig_dpi = 80, save_plot = False, save_path = ""):
    
    # error checking
    if len(indata) == 0 or target_name == "" or pred_name == "" or predictor_name == "":
        print("ERROR: indata, name of target variable, name of prediction and name of predictor must be specified. No default value.")
        return
    
    if predictor_name not in indata.columns:
        print("ERROR: predictor not found in indata. Check the input.")
        return
        
    # define weights
    if exposure_weights in indata.columns:
        exposure_w = indata[exposure_weights]
    else:
        exposure_w = np.tile(1, len(indata))
    
    if frequency_variable in indata.columns:
        frequency_w = indata[frequency_variable]
    else:
        frequency_w = np.tile(1, len(indata))
    
    weights = pd.DataFrame(exposure_w*frequency_w)
    
    # prepare dataframe for plotting
    plot_data = pd.concat([indata[[target_name, pred_name, predictor_name]].reset_index(drop = True),
                               weights.reset_index(drop = True)], axis = 1)

    plot_data.columns = ["A", "E", "P", "W"]
    
    # take into account weights
    plot_data["A"] = plot_data["A"] * plot_data["W"]
    plot_data["E"] = plot_data["E"] * plot_data["W"]
    
    # remove missing values
    plot_data = plot_data.dropna()
    
    # set barplot width
    bar_width = 0.8

    # treatment with numeric Predictor
    if np.issubdtype(indata[predictor_name].dtype, np.number):
        # capping and cupping to exclude outliers
        cupping = plot_data["P"].quantile(cup)
        capping = plot_data["P"].quantile(1-cap)
        
        plot_data = plot_data[plot_data["P"].between(cupping, capping, inclusive = True)]
        
        if len(plot_data["P"].unique()) > graph_groups and not num_as_cat:
            if equal_size_or_distance == "distance":
                s, bins = pd.cut(plot_data["P"], graph_groups, retbins = True)
                bar_width = np.amin(bins[1:] - bins[:-1])*0.9
            else:
                s, bins = pd.qcut(plot_data["P"], graph_groups, retbins = True, duplicates = 'drop')
                bar_width = np.amin(bins[1:] - bins[:-1])
            
            mid = [(a + b)/2 for a, b in zip(bins[:-1], bins[1:])]
            plot_data["P"] = s.cat.rename_categories(mid)
            
    # summary by predictor
    plot_data = plot_data.groupby(["P"]).apply(lambda x: pd.Series([sum(x[v]) for v in ["A", "E", "W"]]))
    
    plot_data.columns = ["A", "E", "W"]
    plot_data["A"] = plot_data["A"] / plot_data["W"]
    plot_data["E"] = plot_data["E"] / plot_data["W"]
    
    # create plot
    fig = plt.figure(dpi = fig_dpi)
    
    # define x axis
    x_axis = plot_data.index.tolist()
 
    # set primary axis  
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(x_axis, plot_data["A"], color = 'orange', linewidth = 3)
    line2, = ax1.plot(x_axis, plot_data["E"], color = 'green',  linewidth = 3)
    plt.ylabel(target_name, fontsize = 12)
    ax1.set_yticks(np.linspace(ax1.get_yticks()[0], ax1.get_yticks()[-1], len(ax1.get_yticks())))
    
    # set secondary axis that shares the x-axis with the ax1
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    bar1 = ax2.bar(x_axis, plot_data["W"], color = 'grey', alpha = 0.3, width = bar_width)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel("Exposure", fontsize = 12)
    ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))

    # set legend 
    plt.legend((line1, line2, bar1), ("Actual", "Expected", "Exposure"))
    plt.xlabel(predictor_name, fontsize = 12)
    plt.title(("Actual vs Expected by Predictor\nModel: " + str(model_name)), fontsize = 14)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(save_path + "plot_AvsE_" + str(predictor_name) + ".png")
            


# create pseudo data for partial dependency plots
def PseudoData(indata = None, predictor_name = "", predictor_levels = None, keep = None,
               num_as_cat = False, sampling = None, cup = 0, cap = 0, npts = 20):
    
    if indata is None or predictor_name == "":
        print("ERROR: indata and name of predictor must be specified. No default value.")
        return
    
    if predictor_name not in indata.columns:
        print("ERROR: predictor not found in indata. Check the input.")
        return
    
    if keep is not None and predictor_name not in keep:
        print("ERROR: predictor is not being kept. Check the input.")
        return
    
    # define levels for partial dependency
    if predictor_levels is None:
        if np.issubdtype(indata[predictor_name].dtype, np.number) and len(indata[predictor_name].unique()) > npts and not num_as_cat:
            # capping and cupping to exclude outliers           
            bar = np.linspace(start=indata[predictor_name].quantile(cup), stop=indata[predictor_name].quantile(1-cap), num=npts)
        else:
            bar = indata[predictor_name].unique()
            if np.issubdtype(indata[predictor_name].dtype, np.number):
                bar = bar[~np.isnan(bar)]
    else:
        bar = predictor_levels
        
    # take a sample to plot, to speed up
    if sampling is not None:
        if sampling < 1:
            indata = indata.sample(frac = sampling)
        else:
            indata = indata.sample(n = sampling)
    
    # keep required columns only
    if keep is not None:
        indata = indata.loc[:, keep]
    
    # prepare pseudo data to score on
    pseudo_data = pd.DataFrame(np.tile(indata.values, (len(bar), 1)), columns = indata.columns)
    bar_ = np.repeat(bar, indata.shape[0])
    pseudo_data[predictor_name] = bar_

    return pseudo_data
    

def PartialPlot(indata = None, pred_name = "", predictor_name = "", exposure_weights = "", frequency_variable = "",
                model_name = "", table_out = False, fig_dpi = 80, save_plot = False, save_path = "", save_name = ""):
    
    if indata is None or pred_name == "" or predictor_name == "":
        print("ERROR: indata and name of predictor must be specified. No default value.")
        return
    
    if predictor_name not in indata.columns:
        print("ERROR: predictor not found in indata. Check the input.")
        return
    
    # define weights
    if exposure_weights in indata.columns:
        exposure_w = indata[exposure_weights]
    else:
        exposure_w = np.tile(1, len(indata))
    
    if frequency_variable in indata.columns:
        frequency_w = indata[frequency_variable]
    else:
        frequency_w = np.tile(1, len(indata))
    
    weights = pd.DataFrame(exposure_w*frequency_w)
    
    # prepare dataframe for plotting
    plot_data = pd.concat([indata[[pred_name, predictor_name]].reset_index(drop = True), 
                           weights.reset_index(drop = True)], axis = 1)

    plot_data.columns = ["E", "P", "W"]
    
    # take into account weights
    plot_data["E"] = plot_data["E"] * plot_data["W"]
    
    # remove missing values
    plot_data = plot_data.dropna()
    
    # summary by predictor
    plot_data = plot_data.groupby(["P"]).apply(lambda x: pd.Series([sum(x[v]) for v in ["E", "W"]]))
    
    plot_data.columns = ["E", "W"]
    plot_data["E"] = plot_data["E"] / plot_data["W"]
    
    # create plot
    fig = plt.figure(dpi = fig_dpi)
    
    # define x axis
    x_axis = plot_data.index.tolist()
 
    # set primary axis  
    ax1 = fig.add_subplot(111)
    line2, = ax1.plot(x_axis, plot_data["E"], color = 'green',  linewidth = 3)
    plt.ylabel("Predicted target", fontsize = 12)
    ax1.set_yticks(np.linspace(ax1.get_yticks()[0], ax1.get_yticks()[-1], len(ax1.get_yticks())))
    
    plt.xlabel(predictor_name, fontsize = 12)
    plt.title(("Partial dependency plot\nModel: " + str(model_name)), fontsize = 14)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(save_path + "plot_PD_" + str(predictor_name) + str(save_name) + ".png")

    if table_out:
        return plot_data


def calcLift(y_prob, y_actual, bins=10):
    
    cols = ['actual','prob_positive']
    data = [y_actual,y_prob]
    
    df = pd.DataFrame(dict(zip(cols,data)))
    df.sort_values(by='prob_positive', ascending=True, inplace=True)

    #Observations where y=1
    total_positive_n = df['prob_positive'].sum()
    #Total Observations
    total_n = df.index.size
    natural_positive_prob = total_positive_n/float(total_n)
    idx_positive_prob = 1

    #Create Bins where First Bin has Observations with the
    #Highest Predicted Probability that y = 1
    df['bin_positive'] = pd.qcut(df['prob_positive'],bins,labels=False)
    
    #Rearrange bins into decreasing order of probability
    df['bin_positive'] = max(df['bin_positive']) - df['bin_positive'] + 1
    
    pos_group_df = df.groupby('bin_positive')
    #Percentage of Observations in each Bin where y = 1 
    lift_positive = pos_group_df['actual'].sum()/pos_group_df['actual'].count()
    lift_index_positive = (lift_positive/natural_positive_prob)
    
    
    #Consolidate Results into Output Dataframe
    lift_df = pd.DataFrame({'baseline_percentage':natural_positive_prob
                            ,'lift_percentage':lift_positive
                            ,'baseline_index':idx_positive_prob
                            ,'lift_index':lift_index_positive
                           })
    return lift_df, pos_group_df


def plotLiftChart(lift: pd.DataFrame):
    
    plt.figure()
    plt.plot(lift['baseline_index'], 'r-', label='Random Selection')
    plt.plot(lift['lift_index'], 'g-', label='Model Selection')
    plt.legend()
    plt.ylabel('Lift')
    plt.xlabel('Decile')
    plt.title('Lift Chart for Online Pick UP Model')
    plt.savefig('PickUp.png', dpi=199)
    plt.show()
    
def find_min_max_rng(df, val_col, normalise=False):
    # find the min and max values
    min_val = df[val_col].min() if not normalise else 0
    max_val = df[val_col].max() if not normalise else 1
    rng = max_val - min_val
    return min_val, max_val, rng

def get_interval(df, val_col, min_val, max_val, bins):
    cut = pd.cut(np.array([min_val, max_val]), bins, retbins=True)
    cut_intervals = cut[1]
    bins_cut = pd.IntervalIndex.from_breaks(cut_intervals)
    s_interval = pd.cut(df[val_col], bins_cut)
    return s_interval
    
def prep_data_for_hist(df, l_groupby_col, plot_col, cutoff=0.95, x_normalize=True, n_bins=100):
    """
    Find the distribution of values for plot_col when grouped by l_groupby_col
    It's often hard to plot when we have more than 10k points
    if normalize: theoretical min is zero and max is one
    """
    # remove null values in plot_col
    df = df.dropna(subset=[plot_col]).reset_index(drop=True)

    # find the min and max values
    normalise = True if df[plot_col].max() <= 1 and df[plot_col].min() >= 0 else False
    min_val, max_val, rng = find_min_max_rng(df, plot_col, normalise)
    precision = rng / n_bins
    s_interval = get_interval(df, plot_col, min_val, max_val, n_bins) # initial bins for cut-off
    
    cut = pd.cut(np.array([min_val, max_val]), n_bins, retbins=True)
    cut_intervals = cut[1]
    bins = pd.IntervalIndex.from_breaks(cut_intervals)
    
    df['interval_right'] = [i.right for i in s_interval.values]
    l_new_groupby = l_groupby_col + ['interval_right']
    df_plot = df.groupby(l_new_groupby)[plot_col].count().reset_index()

    # rename columns
    df_plot.columns = l_groupby_col + ['value', 'count']

    df_plot_norm = (df_plot.groupby(l_groupby_col + ['value'])['count'].apply(lambda x: x.sum()) /  \
        df_plot.groupby(l_groupby_col)['count'].apply(lambda x: x.sum())).reset_index()
    df_plot_norm['cum_count_norm'] = df_plot_norm.groupby(l_groupby_col)['count'].cumsum()
    df_plot_norm_cut = df_plot_norm[df_plot_norm['cum_count_norm'] <= cutoff]

    min_val, max_val, rng = find_min_max_rng(df_plot_norm_cut, 'value', False)
    min_val = math.floor(min_val * 10)/10.0
    max_val = math.ceil(max_val * 100)/100.0 + .01

    df_cut = df[(df[plot_col] >= min_val) & (df[plot_col] <= max_val)].reset_index(drop=True)
    s_interval = get_interval(df_cut, plot_col, min_val, max_val, n_bins) # check if use n_bins or bins?
    df_cut['interval_right'] = [i.right for i in s_interval.values]
    df_cut_plot = df_cut.groupby(l_new_groupby)[plot_col].count().reset_index()
    df_cut_plot.columns = l_groupby_col + ['value', 'count']
    df_cut_plot_norm = (df_cut_plot.groupby(l_groupby_col + ['value'])['count'].apply(lambda x: x.sum()) /  \
        df_cut_plot.groupby(l_groupby_col)['count'].apply(lambda x: x.sum())).reset_index()
    return df_cut_plot_norm

def plot_hist_comp(df_plot, l_groupby_col, x_l=None, x_r=None, y_l=None, y_r=None,
                   title='Score Comparison',filename='hist_comp.html'):
    groupby_str = '-'.join(l_groupby_col)
    if len(l_groupby_col) > 1:
        df_plot['groupby'] = df_plot[l_groupby_col].apply(lambda row: '|'.join(row.values.astype(str)), axis=1)
    else:
        df_plot['groupby'] = df_plot[l_groupby_col]
    uniq_grp = list(set(df_plot['groupby'].values))
    data = [go.Bar(name=grp, 
                   x=df_plot[df_plot[l_groupby_col[0]] == grp]['value'],
                   y=df_plot[df_plot[l_groupby_col[0]] == grp]['count']) for grp in uniq_grp]

    if x_l is None:
        x_l = round(df_plot['value'].min(), 2)
    if x_r is None:
        x_r = round(df_plot['value'].max(), 2)
    if y_l is None:
        y_l = 0
    if y_r is None:
        y_r = 1

    fig = go.Figure(data=data, layout_xaxis_range=[x_l, x_r], layout_yaxis_range=[y_l, y_r])
    fig.update_layout(
        title=title,
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='Count',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1 # gap between bars of the same location coordinate.
    )

    ## NEED TO NORMALISE BY COUNT
    plot(fig, filename=filename)


def plot_audience_comp(data, l_groupby_col, l_score_col, inc_feat_cols=True, date_prefix='2021-04-08'):
    """
    1. total number of plots = number of score columns + number of feature columns
    2. number of groups per plot = number of unique combinations in groupby cols
    """
    groupby_str = '-'.join(l_groupby_col)
    
    df_summ = data.groupby(l_groupby_col)[l_score_col].agg(['min', 'mean', 'median', 'max'])
    
    if len(l_groupby_col) > 1:
        data['combined_groupby'] = data[l_groupby_col].apply(lambda row: '|'.join(row.values.astype(str)), axis=1)
    else:
        data['combined_groupby'] = data[l_groupby_col]
    
    # 2D Density contour
    for score_col in l_score_col:
        fig = px.histogram(data, x=score_col, color="combined_groupby", marginal="rug", # can be `box`, `violin`
                           histnorm='probability') # hover_data=df.columns, 
#         fig = px.histogram(df_samp, x=score_col, color="selection_group", 
#                    marginal="rug", # can be `box`, `violin`
#                    histnorm='probability density', facet_row="dataset_name", 
#                    title=f'{date_prefix} {score_col}') # hover_data=df.columns, 
        # Overlay both histograms
        fig.update_layout(barmode='overlay')
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.5)
        plot(fig, filename=f'{date_prefix}_hist_{groupby_str}_{score_col}.html')
    return None
