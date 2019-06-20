# import plotly.plotly as py
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


def plot_3d_scatter(df, x, y, z, filename = '3d-scatter.html', inline=False):
    pass
