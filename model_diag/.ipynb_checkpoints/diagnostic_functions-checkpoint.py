

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


    
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