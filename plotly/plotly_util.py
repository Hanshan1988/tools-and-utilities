# import plotly.plotly as py
import pandas as pd
import numpy as np
import cufflinks as cf
import plotly.graph_objs as go
from datetime import datetime
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

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


def plot_candlestick_single(df, stock='', title='', ts_col='Date', filename='candle.html', peaks_troughs=False, rangeslider=True,
                            price_norm_col=None, rangeselector=False, bull_cols=[], bear_cols=[], text_cols=[], weighted_col=None,
                            highlight_times=[], highlight_period=pd.Timedelta(minutes=1), bb_cols=[], legend_only=False,
                            price_senses=[], hovers=[], roll_means=[], df_txn=pd.DataFrame(), inline=False, return_fig=False):
    
    prefix = stock + '.' if len(stock) > 0 else stock

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if legend_only:
        visible_param = 'legendonly'
    else:
        visible_param = True
        
    # Candlestick
    fig.add_trace(go.Candlestick(x=df[ts_col],
                                 open=df['{}Open'.format(prefix)],
                                 high=df['{}High'.format(prefix)],
                                 low=df['{}Low'.format(prefix)],
                                 close=df['{}Close'.format(prefix)],
                                 visible=visible_param), secondary_y=False)
    fig.layout.update(title=title)
    fig['layout']['yaxis1'].update(title='Price ($)')
    
    # range slide and range selector
    if not rangeslider:
        fig.update_layout(xaxis_rangeslider_visible=False)
#     if rangeselector: # for daily data
#         fig.update_xaxes(
#             rangeslider_visible=True,
#             rangeselector=dict(
#                 buttons=list([
#                     dict(count=1, label="1m", step="month", stepmode="backward"),
#                     dict(count=6, label="6m", step="month", stepmode="backward"),
#                     dict(count=1, label="YTD", step="year", stepmode="todate"),
#                     dict(count=1, label="1y", step="year", stepmode="backward"),
#                     dict(step="all")
#                     ])
#                 )
#         )
    
    # Add Bolinger Bands BB
    if len(bb_cols) == 2:
        fig.add_scatter(x=df[ts_col], y=df['BB Top'], mode='lines', marker=dict(size=1, color="darkturquoise"), name='BB Top')
        fig.add_scatter(x=df[ts_col], y=df['BB Bot'], mode='none', name='BB Bot', fill='tonexty')
    
    # Add closing price as line 
    fig.add_scatter(x=df[ts_col], y=df['{}Close'.format(prefix)], mode='lines+markers', hoverinfo='all', text=text_cols,
                    name='Closing Price', marker=dict(size=3, color="Cyan"), line = dict(width=1))
    # Add another trace for prime norm
    if price_norm_col is not None:
        # fig.add_trace(go.Scatter(x=df[ts_col], y=df[price_norm_col], name="Closing Price Normed", yaxis="y2"))
        fig.add_trace(go.Scatter(x=df[ts_col], y=df[price_norm_col], name="Closing Price Normed"), secondary_y=False)

    # Add weighted average price
    if weighted_col is not None:
        fig.add_trace(go.Scatter(x=df[ts_col], y=df[weighted_col], name="Weighted Price"), secondary_y=False)

        
    # Add volumes data
    # fig.add_bar(x=df[ts_col], y=df['{}Volume'.format(prefix)], name='Volume', secondary_y=True)
    if 'Volume' in df.columns.tolist():
        close_higher = (df['{}Open'.format(prefix)] < df['{}Close'.format(prefix)])
        fig.add_trace(go.Bar(x=df[ts_col][close_higher], y=df['{}Volume'.format(prefix)][close_higher], 
                             name='Volume (Close Higher)', marker=dict(color='Green')), secondary_y=True)
        fig.add_trace(go.Bar(x=df[ts_col][~close_higher], y=df['{}Volume'.format(prefix)][~close_higher], 
                             name='Volume (Close Lower)', marker=dict(color='Red')), secondary_y=True)
        fig['layout']['yaxis2'].update(title='Volume', range=[0, df['{}Volume'.format(prefix)].max() * 10], autorange=False, showgrid=False)
        
#     fig.update_layout(
#     yaxis=dict(
#         title="Price",
#     ),
#     yaxis2=dict(
#         title="Price Normed",
#         anchor="free",
#         overlaying="y",
#         side="left",
#         position=0.02,
#         range=[-0.8, 1.2],
#         autorange=False
#     ))
        
    # Add past transactions
    if df_txn.shape[0] > 0:
        df_txn_b = df_txn[df_txn.action == 'B'].reset_index(drop=True)
        df_txn_s = df_txn[df_txn.action == 'S'].reset_index(drop=True)
        if df_txn_b.shape[0] > 0:
            fig.add_scatter(x=df_txn_b['Date'], y=df_txn_b['price'], mode='markers', name='Action - Buy',
                            marker=dict(size=8, color="Blue"))
        if df_txn_s.shape[0] > 0:
            fig.add_scatter(x=df_txn_s['Date'], y=df_txn_s['price'], mode='markers', name='Action - Sell',
                            marker=dict(size=8, color="Red"))
    # Add rolling SMA 
    if len(roll_means) > 0:
        colour = ['red', 'green', 'blue']
        for period, colour in zip(roll_means, colour):
            df['sma_{}'.format(period)] = df['{}Close'.format(prefix)].rolling(period).mean()
            fig.add_scatter(x=df[ts_col], y=df['sma_{}'.format(period)], mode='lines',
                            name='SMA {}'.format(period), marker=dict(size=3, color=colour), line=dict(width=2))
            
    # Add peaks an troughs
    if peaks_troughs:
        prominence = .01 * df['{}Close'.format(prefix)].mean()
        peak_indices = find_peaks(df['{}Close'.format(prefix)], prominence=prominence, distance=20)[0]
        trough_indices = find_peaks(-df['{}Close'.format(prefix)], prominence=prominence, distance=20)[0]
        fig.add_scatter(x=df[ts_col][peak_indices], y=[df['{}Close'.format(prefix)][j] for j in peak_indices], mode='markers',
                        name='Peaks', marker=dict(size=5, color="Red", symbol='cross'))
        fig.add_scatter(x=df[ts_col][trough_indices], y=[df['{}Close'.format(prefix)][j] for j in trough_indices], mode='markers',
                        name='Troughs', marker=dict(size=5, color="Green", symbol='cross'))
        
    # Highlight and annotate annoucements
    if len(highlight_times) > 0:
        colours = ["Navy" if price_sense == 1 else "LightSalmon" for price_sense in price_senses]
        fig.layout.update(
            shapes=[
                dict(type="rect",
                    xref="x",
                    yref="paper",
                    x0=highlight_time,
                    y0=0,
                    x1=highlight_time,
                    y1=1,
                    fillcolor=colour,
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                    ) for highlight_time, colour in zip(highlight_times, colours)],
            annotations=[
                dict(x=highlight_time,
                     y=df['{}High'.format(prefix)].max(),
                     xref="x",
                     yref="y",
                     text=hover,
                     showarrow=False,
                     ax=0,
                     ay=20) for highlight_time, hover in zip(highlight_times, hovers)])
        
    # highlight bullish candlestick patterns
    if len(bull_cols) > 0:
        sum_indicators = df[bull_cols].sum(axis=1)
        highlight_ts = df[ts_col][sum_indicators > 0].tolist()
        highlight_y = df['Close'][sum_indicators > 0].tolist()
        highlight_size = (10 * sum_indicators[sum_indicators > 0]).tolist()
        fig.add_scatter(x=highlight_ts, y=highlight_y, mode='markers',
                        name='Bullish Indicators',
                        marker=dict(size=highlight_size, color="Green", symbol='square-dot'))
        
    if len(bear_cols) > 0:
        sum_indicators = df[bear_cols].sum(axis=1)
        highlight_ts = df[ts_col][sum_indicators > 0].tolist()
        highlight_y = df['Close'][sum_indicators > 0].tolist()
        highlight_size = (10 * sum_indicators[sum_indicators > 0]).tolist()
        fig.add_scatter(x=highlight_ts, y=highlight_y, mode='markers',
                        name='Bearish Indicators', 
                        marker=dict(size=highlight_size, color="Red", symbol='square-dot'))
        
    # show crosshair
    fig['layout']['xaxis'].update(showspikes=True, spikedash = 'solid', spikethickness=1)
    fig['layout']['yaxis'].update(showspikes=True, spikedash = 'solid', spikethickness=1)
    
    fig.update_layout(
#         autosize=False,
        margin=dict(
            l=10,
            r=10,
            b=10,
            t=40,
            pad=5
        ),
        hovermode='x'
    )


    if return_fig:
        return fig
    else:
        if inline:
            init_notebook_mode()
            iplot(fig, filename=filename)
        else:
            plot(fig, filename=filename)
    


def plot_depth_chart(buy_price_list, buy_vol_list, sell_price_list, sell_vol_list, 
                     filename='depth-chart.html', inline=False, return_fig=False):
    # # df has type, price and volume columns
    # df_buy = df[df['type'] == 'buy'].sort_values('price', ascending=False).reset_index(drop=True)
    # df_sell = df[df['type'] == 'sell'].sort_values('price', ascending=True).reset_index(drop=True)
    # # convert to cumulative volumes
    # buy_price_list = df_buy['price'].tolist()
    # sell_price_list = df_sell['price'].tolist()
    # buy_vol_list = df_buy['volume'].cumsum().tolist()
    # sell_vol_list = df_sell['volume'].cumsum().tolist()
    buy_price_list, buy_vol_list, sell_price_list, sell_vol_list = \
        np.array(buy_price_list), np.array(buy_vol_list), np.array(sell_price_list), np.array(sell_vol_list)

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
    
    fig.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        dtick = 0.01
        )
    )
    
    fig.update_layout(
#         autosize=False,
        margin=dict(
            l=10,
            r=10,
            b=10,
            t=10,
            pad=1
        )
    )
    
    if return_fig:
        return fig
    else:
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
