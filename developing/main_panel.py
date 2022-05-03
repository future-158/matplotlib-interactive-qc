from ast import parse
import collections
import datetime as dte
import io
import itertools
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.dates import date2num, num2date
from matplotlib.widgets import (
    Button,
    CheckButtons,
    RadioButtons,
    RectangleSelector,
    Slider,
)

from omegaconf import OmegaConf

cfg = OmegaConf.load("conf/config.yml")

source = cfg.catalogue.raw
target = cfg.catalogue.clean

plt.rcParams["font.family"] = "Malgun Gothic"
# font = ImageFont.truetype(r'malgun.ttf', 30)


class QC:
    def __init__(self, df):
        plt.close("all")
        self.fig, self.ax = plt.subplots(figsize=(16, 8))
        self.fig.subplots_adjust(left=0.2, bottom=0.2)
        self.toggle_selector = RectangleSelector(
            self.ax,
            self.onselect,
            drawtype="box",
            useblit=True,
            button=[1],
            interactive=True,
        )

        self.df = df.copy()

        n_cols = len(cfg.columns.slice_var)
        h_ratio = 1 / (n_cols + 2)

        self.rax = {}
        self.radio = {}
        self.kv = {}

        for i, name in enumerate(cfg.columns.slice_var):
            self.rax[name] = plt.axes(
                [0, i * h_ratio, 0.1, (i + 1) * h_ratio],
                facecolor="lightgoldenrodyellow",
            )
            self.radio[name] = RadioButtons(
                self.rax[name], tuple(df[name].unique()), active=0
            )
            self.radio[name].on_clicked(self.on_clicked)

        # init kvs
        for name, btn in self.radio.items():
            self.kv[name] = btn.value_selected

    def on_clicked(self, *args, **kwargs):
        mask_list = []
        for name, btn in self.radio.items():
            val = btn.value_selected
            self.kv[name] = val  # upgrade self.kv
            mask = self.df[name] == val
            mask_list.append(mask)
        self.mask = np.logical_and.reduce(mask_list)

        self.ax.clear()
        self.xdata = self.df.loc[self.mask, "datetime"]
        self.ydata = self.df.loc[self.mask, "value"]

        self.pc = self.ax.scatter(
            x=self.xdata, y=self.ydata, c="b", label=self.kv["variable"]
        )

        self.offsets = self.pc.get_offsets()
        self.update()

    def dump(self):
        return self.df

    def update(self):
        self.ydata = self.df.loc[self.mask, "value"]
        self.offsets.mask = np.tile(np.isnan(self.ydata).values[:, np.newaxis], [1, 2])
        self.offsets.data[self.offsets.mask] = np.nan
        self.fig.canvas.draw_idle()

    def onselect(self, eclick, erelease):
        # change ydata
        x0 = eclick.xdata
        x1 = erelease.xdata

        y0 = eclick.ydata
        y1 = erelease.ydata
        x0, x1 = np.sort([x0, x1])
        y0, y1 = np.sort([y0, y1])

        dt0 = num2date(x0)
        dt0 = dt0.replace(tzinfo=None)
        ts_x0 = pd.Timestamp(dt0)

        dt1 = num2date(x1)
        dt1 = dt1.replace(tzinfo=None)
        ts_x1 = pd.Timestamp(dt1)

        idx_x = (self.df.datetime > ts_x0) & (self.df.datetime < ts_x1)
        idx_y = (self.df.value > y0) & (self.df.value < y1)

        idx = idx_x & idx_y
        idx = idx & self.mask
        self.df.loc[idx, "value"] = np.nan
        self.update()


df = pd.read_csv(source, parse_dates=["datetime"], infer_datetime_format=True)

df["year"] = df.datetime.dt.year.astype(str)  # radio_btn.value_selected return str.
assert all(
    [col in df.columns for col in [*cfg.columns.slice_var, cfg.columns.datetime_var]]
)
self = QC(df)

import numpy as np
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvas  # not needed for mpl >= 3.1
import panel as pn

pn.extension()
FigureCanvas(self.fig)  # not needed for mpl >= 3.1
mpl_pane = pn.pane.Matplotlib(self.fig, dpi=144, interactive=True)
pn.serve(
    mpl_pane,
    # static_dirs={'assets': './assets'}
)
