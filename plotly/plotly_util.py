# import plotly.plotly as py
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf


def plot_2d_scatter(df, x, y, xlab = '', ylab='', title='', filename='2d-scatter.html', inline=False):
    # Create a trace
    trace = go.Scattergl(
        x = df[x],
        y = df[y],
        mode = 'markers'
    )
    
    data = [trace]
    if inline:
        init_notebook_mode()
        iplot(data, filename=filename)
    else:
        plot(data, filename=filename)


def plot_2d_ts(df, to_excl=[], xlab='Dates', ylab='', title='', filename = '2d-ts.html', inline=False):
    
    all_cols = df.columns
    cols_plot = list(set(all_cols) - set(to_excl))
    df_plot = df[cols_plot]
    
    data = df.iplot(asFigure=True, kind='scatter', xTitle=xlab, yTitle=ylab, title=title)
    
    if inline:
        init_notebook_mode()
        iplot(data, filename=filename)
    else:
        plot(data, filename=filename)


def plot_2d_bar(df, x, y, xlab ='', ylab='', title='', pct=False, filename='2d-bar.html', inline=False):
    # Each bar stands for a value of x and is split up by y values
    df['dummy'] = 1
    x_uniq = df[x].unique().tolist()
    x_uniq = pd.to_datetime(x_uniq) if df.dtypes[x] == 'datetime64[ns]' else x_uniq
    y_uniq = df[y].unique().tolist()
    
    multi_index = pd.MultiIndex.from_product([x_uniq, y_uniq], names=[x, y])
    df_sub = df.groupby([x, y])['dummy'].sum()
    pct_func = lambda x: x / float(x.sum()) if pct else x
    df_sub_2 = df_sub.groupby(level=0).apply(pct_func).reindex(multi_index, fill_value=0)
    
    data = [go.Bar(
        x = [str(x) for x in x_uniq],
        y = df_sub_2[:, y_uniq_n],
        name = y_uniq_n
        ) for y_uniq_n in y_uniq]
    
    layout = go.Layout(
        barmode='stack',
        'xaxis':{'title':x})
    
    fig = go.Figure(data=data, layout=layout)
    
    if inline:
        init_notebook_mode()
        iplot(fig, filename=filename)
    else:
        plot(fig, filename=filename)


def plot_3d_scatter(df, x, y, z, filename = '3d-scatter.html', inline=False):
    pass
