import os
from pathlib import Path
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
# conda install -c bokeh jupyter_bokeh
source_dir = Path('after_qc')


stations =  ['거제도', '모슬포', '부산', '완도', '제주', '진도', '추자도']
done = ['통영']


station = '거제도'
file = source_dir / f'{station}.csv'
usecols = ['datetime','sst','sal']

def load_data():
    df = pd.read_csv(
        file, parse_dates=['datetime'], infer_datetime_format=True, 
        usecols=usecols)
    return df

# df = pn.state.as_cached('sal', load_data)

df = load_data()
# pd.read_csv( 
#     Path('qc2')   / '통영.csv',
#     parse_dates=['datetime'], infer_datetime_format=True, 
#     usecols=usecols).sal.isna().sum()

df.sal.isna().sum()

# data = hv.Dataset(df, [('x', 'Longitude'), ('y', 'Latitude')],
#                      [('avg_prcp', 'Annual Precipitation (mm/yr)'),
#                       ('area_km2', 'Area'), ('latdeg', 'Latitude (deg)'),
#                       ('avg_temp_at_mean_elev', 'Annual Temperature at avg. altitude'), 
#                       ('mean_elev', 'Elevation')])

opts.defaults(opts.Points(tools=['box_select', 'lasso_select']))
points = hv.Points(df, ['datetime', 'sal'])
# static_sal = datashade(points, cmap='fire', cnorm="linear").relabel("Datashaded").options(height=720, width=1280, tools=['box_select'])
static_sal = rasterize(points).relabel("Datashaded").options(
    height=1080, width=1600, 
    tools=['box_select'], 
    cmap='fire', cnorm='linear')
# print(inspect.getsource(rasterize))

def plot_sal(data):   return hv.Points(df, ['datetime','sal']).options(alpha=1)


# active_tools=['box_select']
ls = hv.link_selections.instance(
    selected_color='#bf0000',
    unselected_color='#ff9f9f',
    unselected_alpha=1)

sal_map = ls(static_sal)


select_data_list = []
def remove(data): 
    # # print(data.index)
    # # global points
    # select_data_list.append(data)
    # if data.shape[0] < 100000:
    #     qc_mask = points.data.datetime.isin(data.datetime)
    #     points.data.loc[qc_mask, 'sal'] = 0.
    
    # nan_count = (points.data.sal == 0.).sum()

    select_data_list.append(data)
    if data.shape[0] < 1000000:
        qc_mask = df.datetime.isin(data.datetime)
        df.loc[qc_mask, 'sal'] = np.nan

    nan_count = (df.sal == np.nan).sum()
    return pn.pane.Markdown(
        str(nan_count),
        align='center', width=500
    )

def update(event, save=True): 
    dest = Path('qc2') / file.name
    df.to_csv(dest, index=False)
    return 1

button = pn.widgets.Button(name='Save')
button.on_click(update)

dynamic_count = pn.bind(remove, ls.selection_param(df))
content = pn.Column(
    pn.Row(dynamic_count, button),
    sal_map)

template = pn.template.MaterialTemplate(title='World Glaciers Explorer')
template.header.append(pn.Row(dynamic_count, button))
template.main.append(sal_map)
template.servable()
# bokeh_server = template.show(port=8989)
# bokeh_server.stop()




