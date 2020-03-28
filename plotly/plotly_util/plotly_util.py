# import plotly.plotly as py
import pandas as pd
import numpy as np
import cufflinks as cf
import plotly.graph_objs as go
from datetime import datetime
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


def convert_buy_sell_prices(df, top_n=10):
	# assumes df in the shape of one row per snapshot in time with 10 buy prices and 10 sell prices
	buy_price_cols = ['buy_price_{0}'.format(i) for i in range(top_n)]
	sell_price_cols = ['sell_price_{0}'.format(i) for i in range(top_n)]
	buy_vol_cols = ['buy_volume_{0}'.format(i) for i in range(top_n)]
	sell_vol_cols = ['sell_volume_{0}'.format(i) for i in range(top_n)]

	buy_price_values = df[buy_price_cols].values[0]
	sell_price_values = df[sell_price_cols].values[0]
	buy_vol_values = df[buy_vol_cols].values[0]
	sell_vol_values = df[sell_vol_cols].values[0]

	return buy_price_values, buy_vol_values, sell_price_values, sell_vol_values


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
    
    data = df.iplot(asFigure=True, kind='scatter', xTitle=xlab, yTitle=ylab, title=title, showlegend=True)
    
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
        barmode='stack'
        )
    
    fig = go.Figure(data=data, layout=layout)
    
    if inline:
        init_notebook_mode()
        iplot(fig, filename=filename)
    else:
        plot(fig, filename=filename)


def plot_3d_scatter(df, x, y, z, filename='3d-scatter.html', inline=False):
    pass


def plot_candlestick_single(input, stock='', filename='candlestick.html', inline=False):
    # stock in the form 'AAPL'
    if len(stock) > 0:
        prefix = stock + '.'
    df = pd.read_csv(input)
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                        open=df['{}Open'.format(prefix)],
                        high=df['{}High'.format(prefix)],
                        low=df['{}Low'.format(prefix)],
                        close=df['{}Close'.format(prefix)])])
    if inline:
        init_notebook_mode()
        iplot(fig, filename=filename)
    else:
        plot(fig, filename=filename)


def plot_depth_chart(buy_price_list, buy_vol_list, sell_price_list, sell_vol_list, filename='depth-chart.html', inline=False):
    # # df has type, price and volume columns
    # df_buy = df[df['type'] == 'buy'].sort_values('price', ascending=False).reset_index(drop=True)
    # df_sell = df[df['type'] == 'sell'].sort_values('price', ascending=True).reset_index(drop=True)
    # # convert to cumulative volumes
    # buy_price_list = df_buy['price'].tolist()
    # sell_price_list = df_sell['price'].tolist()
    # buy_vol_list = df_buy['volume'].cumsum().tolist()
    # sell_vol_list = df_sell['volume'].cumsum().tolist()

    buy_indx = np.argsort(buy_price_list)[::-1]
    sell_indx = np.argsort(sell_price_list)

    buy_price_list = buy_price_list[buy_indx]
    buy_vol_list = np.cumsum(buy_vol_list[buy_indx])
    sell_price_list = sell_price_list[sell_indx]
    sell_vol_list = np.cumsum(sell_vol_list[sell_indx])

    # plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=buy_price_list, y=buy_vol_list,
                             line={"shape": 'hv'},
                             fill='tozeroy',
                             mode='lines+markers',
                             name='bid prices'))
    fig.add_trace(go.Scatter(x=sell_price_list, y=sell_vol_list,
                             line={"shape": 'hv'},
                             fill='tozeroy',
                             mode='lines+markers',
                             name='ask prices'))
    if inline:
        init_notebook_mode(connected=True)
        iplot(fig, filename=filename)
    else:
        plot(fig, filename=filename)
        
        
def plot_moving_depth_chart(df, filename='moving-depth-chart.html', inline=False):
    # Takes in a standard df from commsec website
    price_cols = [col for col in df.columns if 'price' in col]
    min_price = df[price_cols].values.min()
    max_price = df[price_cols].values.max()
    
    buy_vol_max = df[[col for col in df.columns if 'buy_volume' in col]].sum(axis=1).max()
    sell_vol_max = df[[col for col in df.columns if 'sell_volume' in col]].sum(axis=1).max()
    max_vol = buy_vol_max if buy_vol_max > sell_vol_max else sell_vol_max

    times = df.timestamp.tolist()

    # make figure
    figure = {'data': [], 'layout': {}, 'frames': []}

    # fill in most of layout
    figure['layout']['xaxis'] = {'range': [min_price, max_price], 'title': 'Price'}
    figure['layout']['yaxis'] = {'range': [0, max_vol], 'title': 'Volume'}
    figure['layout']['hovermode'] = 'closest'
    figure['layout']['sliders'] = {
        'args': ['transition', {'duration': 100, 'easing': 'cubic-in-out'}],
        'initialValue': times[0],
        'plotlycommand': 'animate',
        'values': times,
        'visible': True
    }
    figure['layout']['updatemenus'] = [
        {
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 500, 'redraw': False},
                             'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                    'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }
    ]

    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Time:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }

    # make data first frame
    time = times[0]
    df_sub = df[df.timestamp == time]
    buy_price_values, buy_vol_values, sell_price_values, sell_vol_values = convert_buy_sell_prices(df_sub)
    buy_indx = np.argsort(buy_price_values)[::-1]
    sell_indx = np.argsort(sell_price_values)
    buy_price_list = buy_price_values[buy_indx]
    buy_vol_list = np.cumsum(buy_vol_values[buy_indx])
    sell_price_list = sell_price_values[sell_indx]
    sell_vol_list = np.cumsum(sell_vol_values[sell_indx])

    for price, vol, name in zip([buy_price_list, sell_price_list], [buy_vol_list, sell_vol_list], ['buy', 'sell']):
        data_dict = {'x':list(price), 'y':list(vol),
                                 'line' :{"shape": 'hv'},
                                 'fill' :'tozeroy',
                                 'mode' :'lines+markers',
                                 'name' :name}
        figure['data'].append(data_dict)

    # make frames
    for time in times:
        frame = {'data': [], 'name': str(time)}

        df_sub = df[df.timestamp == time]
        buy_price_values, buy_vol_values, sell_price_values, sell_vol_values = convert_buy_sell_prices(df_sub)

        buy_indx = np.argsort(buy_price_values)[::-1]
        sell_indx = np.argsort(sell_price_values)
        buy_price_list = buy_price_values[buy_indx]
        buy_vol_list = np.cumsum(buy_vol_values[buy_indx])
        sell_price_list = sell_price_values[sell_indx]
        sell_vol_list = np.cumsum(sell_vol_values[sell_indx])

        # plot
        for price, vol, name in zip([buy_price_list, sell_price_list], [buy_vol_list, sell_vol_list], ['buy', 'sell']):
            data_dict = {'x':list(price), 'y':list(vol),
                                     'line' :{"shape": 'hv'},
                                     'fill' :'tozeroy',
                                     'mode' :'lines+markers',
                                     'name' :name}

            frame['data'].append(data_dict)

        figure['frames'].append(frame)
        slider_step = {'args': [
            [time],
            {'frame': {'duration': 300, 'redraw': False},
             'mode': 'immediate',
           'transition': {'duration': 300}}
         ],
         'label': time,
         'method': 'animate'}
        sliders_dict['steps'].append(slider_step)

    figure['layout']['sliders'] = [sliders_dict]

    if inline:
        init_notebook_mode(connected=True)
        iplot(figure, filename=filename)
    else:
        plot(figure, filename=filename)
