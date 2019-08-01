import matplotlib.pyplot as plt

def get_roc_fpr_tpr(y_true, y_prob):
    assert y_true.shape[1] == y_prob.shape[1], "y_true and y_prob has different shapes!"
    n_cols = y_true.shape[1]
    # Compute ROC curve and ROC area for each series
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_cols):
        fpr[i], tpr[i], _ = mt.roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc[i] = mt.auc(fpr[i], tpr[i])
    is_multiclass = False
    # Compute micro-average ROC curve and ROC area
    if n_cols > 2:
        is_multiclass = True
        fpr["micro"], tpr["micro"], _ = mt.roc_curve(y_true.ravel(), y_prob.ravel())
        roc_auc["micro"] = mt.auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc, n_cols, is_multiclass


def plot_roc_curves(d={}, fname=None):
    fig = plt.figure(figsize=(8, 8))
    fpr, tpr, roc_auc, n_cols, is_multiclass = dict(), dict(), dict(), dict(), dict()
    series = list(d.keys())
    for s in series:
        fpr[s], tpr[s], roc_auc[s], n_cols[s], is_multiclass[s] = get_roc_fpr_tpr(d[s]['target'], d[s]['pred'])
    # test if is_multiclass_all is same for each key
    lw = 3
    colors = cycle(['darkorange', 'aqua', 'cornflowerblue'])
    # Binary case
    for s, color in zip(series, colors):
        plt.plot(fpr[s][0], tpr[s][0], color=color, lw=lw,
                 label='ROC curve of {0} (area = {1:0.2f})'
                 ''.format(s, roc_auc[s][0]))
    # Multiclass case
    if is_multiclass[s]:
        # Plot multiple classes roc's
        # Plot micro and macro averages
        pass

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    if fname is not None:
        fig.savefig(fname + '.png', bbox_inches="tight")
		
def calc_lift(y_true, y_prob, bins=10):
    cols = ['actual', 'prob_positive']
    data = [y_true, y_prob]
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
    df['bin_positive'] = 1 + max(df['bin_positive']) - df['bin_positive']
    
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
    
    return lift_df
	
    
def plot_lift_chart(lift: pd.DataFrame, fname=None):
    fig = plt.figure(figsize=(8, 8))
    plt.plot(lift['baseline_index'], 'r-', label='Lift from random selection')
    plt.plot(lift['lift_index'], 'g*-', label='Lift from model selection')
    plt.legend(loc='best')
    plt.xticks(range(1, lift.index.values.max()+1))
    if fname is not None:
        fig.savefig(fname + '.png', bbox_inches="tight")
    plt.show()
