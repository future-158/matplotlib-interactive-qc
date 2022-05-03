import inspect
import os
from pathlib import Path

import holoviews as hv
import numpy as np
import pandas as pd
import panel
import panel as pn
from holoviews import opts, streams
from holoviews.core.spaces import DynamicMap
from holoviews.streams import Selection1D

# panel serve my_script.py --show 
# conda install -c bokeh jupyter_bokeh
# pn.extension(comms="bokeh")
pn.extension(comms="vscode")
opts.defaults(opts.Points(tools=["box_select", "lasso_select"]))
# conda install -c bokeh jupyter_bokeh
import numpy as np
from matplotlib import cm, interactive
from matplotlib.backends.backend_agg import \
    FigureCanvas  # not needed for mpl >= 3.1
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import axes3d


def load_data(station):
    source = "test.csv"
    params = ["기온(℃)", "수온(℃)", "기압(hPa)", "습도(%)"]
    renamer = {"기온(℃)": "temp", "수온(℃)": "sst", "기압(hPa)": "qff", "습도(%)": "rh"}
    df = pd.read_csv(source, index_col=["시간"], parse_dates=["시간"])
    df = df.loc[df.항 == station, params]
    df = df.rename(columns=renamer)
    return df


station = "인천항"
param = "sst"
# df = pn.state.as_cached('sal', load_data)
df = load_data(station)

df["id"] = np.arange(df.shape[0])
df = df.reset_index()
# subsample_mask = (df.datetime.dt.minute == 0) & (df.datetime.dt.hour == 0)
# subsample_mask = df[subsample_mask]

df['day'] = df.시간.dt.day
df['hour'] = df.시간.dt.hour


# station = pn.widgets.Select(options=stations)

# param = pn.widgets.Select(options=params)
day = pn.widgets.IntSlider(name="day", value=1, start=1, end=29)
hour = pn.widgets.IntSlider(name="hour", value=0, start=0, end=23)
widgetbox = pn.WidgetBox(day, hour)

@pn.depends(day, hour, watch=True)
def plot_page(day, hour):
    points = hv.Points(
        df.query("day==@day and hour==@hour"), ["시간", 'sst']
    ).options(alpha=1, height=400, width=500, framewise=True)

    return points

ls = hv.link_selections.instance(
    selected_color="#bf0000", unselected_color="#ff9f9f", unselected_alpha=1
)


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
        df.loc[qc_mask, "sal"] = np.nan

    nan_count = (df.sal == np.nan).sum()
    return pn.pane.Markdown(str(nan_count), align="center", width=500)


# def update(event, save=True):
#     dest = Path('qc2') / file.name
#     df.to_csv(dest, index=False)
#     return 1
# button = pn.widgets.Button(name='Save')
# button.on_click(update)

dynamic_count = pn.bind(remove, ls.selection_param(df))
content = pn.Column(pn.Row(dynamic_count), widgetbox, dmap)
content

len(checkpoints)
app = pn.panel(content, center=True, widget_location="right_bottom")
app.servable()


# @pn.depends(station, param)
def plot_page(day, param):
    points = hv.Points(df.loc[df["시간"].dt.day == day], ["시간", param]).options(
        alpha=1, height=400, width=500, framewise=True
    )
    return points


points = hv.Points(df.loc[df["시간"].dt.day == day], ["시간", param]).options(
    alpha=1, height=400, width=500, framewise=True
)

# dmap = rasterize(
#         points,
#         # dynamic=False,
#         ).relabel("rasterized").options(
#         # height=1080, width=1600,
#         height=800, width=1600,
#         tools=['box_select'],
#         cmap='fire', cnorm='linear', framewise=True)


# station = pn.widgets.Select(options=stations)
# param = pn.widgets.Select(options=params)
# year = pn.widgets.IntSlider(name='Year', value=2019, start=2015, end=2021)

# widgetbox = pn.WidgetBox(station, param, year)

ls = hv.link_selections.instance(
    selected_color="#bf0000",
    # unselected_color='#ff9f9f',
    unselected_alpha=1,
)

dmap = ls(points)

checkpoints = []


def remove(data):
    # data = select_data_list[-1]
    if len(data) == 1:
        return
    if len(data) > df.shape[0] * 0.5:
        return

    checkpoints.append(data)
    qc_mask = df.id.isin(data.id)
    df.loc[qc_mask, param] = np.nan
    # dmap.event(station=station.value)
    return
    # return plot_page(station.value, param.value)

    if data.shape[0] < 1000000:
        qc_mask = df.datetime.isin(data.datetime)
        df.loc[qc_mask, "sal"] = np.nan

    nan_count = (df.sal == np.nan).sum()
    return pn.pane.Markdown(str(nan_count), align="center", width=500)


# def update(event, save=True):
#     dest = Path('qc2') / file.name
#     df.to_csv(dest, index=False)
#     return 1

# button = pn.widgets.Button(name='Save')
# button.on_click(update)
dynamic_count = pn.bind(remove, ls.selection_param(df))



# content = pn.Column(
#     pn.Row(dynamic_count),
#     # widgetbox,
#     dmap)

# app = pn.panel(content, center=True, widget_location='right_bottom')
# app.servable()

template = pn.template.MaterialTemplate(title="nia qc")
template.header.append(pn.Row(dynamic_count))
template.main.append(dmap)
template.servable()
