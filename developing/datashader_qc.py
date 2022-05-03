import os
from pathlib import Path
from holoviews.core.spaces import DynamicMap
from numba.core.errors import HighlightColorScheme
import panel as pn
import holoviews as hv
import numpy as np
import pandas as pd
import panel
from holoviews import opts, streams
from holoviews.streams import Selection1D
from holoviews.operation.datashader import datashade, shade, dynspread, spread, rasterize
import datashader
import inspect
hv.extension('bokeh')
pn.extension(comms='vscode')
opts.defaults(opts.Points(tools=['box_select', 'lasso_select']))
# conda install -c bokeh jupyter_bokeh
source_dir = Path('after_qc')


stations =  ['통영','거제도', '모슬포', '부산', '완도', '제주', '진도', '추자도']
params = ['sal', 'sst']
years = np.arange(2015,2022)

usecols = ['datetime','sst','sal']

def load_data(station,  **kwargs):
    file = source_dir / f'{station}.csv'
    df = pd.read_csv(
        file, parse_dates=['datetime'], infer_datetime_format=True, 
        usecols=usecols, **kwargs)
    df['station'] = file.stem
    return df

# df = pn.state.as_cached('sal', load_data)
df = pd.concat(load_data(station) for station in stations)

df['id'] = np.arange(df.shape[0])
df = df.reset_index(drop=True)
df['year'] = df['datetime'].dt.year

subsample_mask = (df.datetime.dt.minute == 0) & (df.datetime.dt.hour == 0) 
df = df[subsample_mask]


station = pn.widgets.Select(options=stations)
param = pn.widgets.Select(options=params)
year = pn.widgets.IntSlider(name='Year', value=2019, start=2015, end=2021)
widgetbox = pn.WidgetBox(station, param, year)


@pn.depends(station, param, year, watch=True)
def plot_page(station, param, year):
    points =  hv.Points(
        df.query('station==@station and year==@year'), 
        ['datetime',param]).options(alpha=1, height=400, width=500, framewise=True)
    return rasterize(
        points,  
        # dynamic=False,
        ).relabel("rasterized").options(
        # height=1080, width=1600, 
        height=400, width=500, 
        tools=['box_select'], 
        cmap='fire', cnorm='linear', framewise=True)


ls = hv.link_selections.instance(
    selected_color='#bf0000',
    unselected_color='#ff9f9f',
    unselected_alpha=1)

dmap = pn.Row(plot_page)
dmap = ls(dmap)

checkpoints = []
def remove(data): 
    # data = select_data_list[-1]
    if len(data) == 1:
        return
    if len(data) > df.shape[0] * 0.5:
        return

    checkpoints.append(data)
    qc_mask = df.id.isin(data.id)
    df.loc[qc_mask, param.value] = np.nan
    station.select(station.value)
    return
    # return plot_page(station.value, param.value)

    if data.shape[0] < 1000000:
        qc_mask = df.datetime.isin(data.datetime)
        df.loc[qc_mask, 'sal'] = np.nan

    nan_count = (df.sal == np.nan).sum()
    return pn.pane.Markdown(
        str(nan_count),
        align='center', width=500
    )

# def update(event, save=True): 
#     dest = Path('qc2') / file.name
#     df.to_csv(dest, index=False)
#     return 1
# button = pn.widgets.Button(name='Save')
# button.on_click(update)

dynamic_count = pn.bind(remove, ls.selection_param(df))
content = pn.Column(
    pn.Row(dynamic_count),
    widgetbox,
    dmap)


app = pn.panel(content, center=True, widget_location='right_bottom')

app.servable()
