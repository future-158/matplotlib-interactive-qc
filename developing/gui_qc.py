import os
import sys
from pathlib import Path
import shutil
import matplotlib
from types import SimpleNamespace
from matplotlib.cbook import Grouper
from pandas.io import parsers
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.dates import date2num, num2date
from matplotlib.figure import Figure
from matplotlib.widgets import (
    Button,
    CheckButtons,
    RadioButtons,
    RectangleSelector,
    Slider,
)
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)
from PyQt5.QtGui import QPixmapCache
from functools import partial, update_wrapper
import logging
logging.basicConfig(encoding='cp949', level=logging.DEBUG)

doc_string = pd.read_csv.__doc__
pd.read_csv = partial(pd.read_csv, encoding='cp949')
pd.read_csv.__doc__  = doc_string


# comboBox = QtWidgets.QComboBox()
# radioButton = QtWidgets.QRadioButton()
# comboBox.currentData()


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# font = ImageFont.truetype(r'c:\windows\fonts\malgun.ttf', 30)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


# class MainWindow(QtWidgets.QMainWindow):
class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # grid = QVBoxLayout()
        grid = QGridLayout()

        self.is_grid = True
        self.setLayout(grid)

        self.setMinimumSize(1280, 720)
        # self.setMaximumSize(1920, 1080)
        # station_widget = QtWidgets.QComboBox()
        # station_widget.addItems(
        #     [
        #         "deprecated"
        #     ]
        # )
        # station_widget.currentTextChanged.connect(self.station_changed)
        
        param_widget = QtWidgets.QComboBox()
        param_widget.addItems(['기온(℃)', '수온(℃)', '기압(hPa)', '습도(%)'])        
        param_widget.currentTextChanged.connect(self.param_changed)

        reload_button = QtWidgets.QPushButton("load csv file")
        reload_button.clicked.connect(self.update_filename)

        saveButton = QtWidgets.QPushButton("save", self)
        saveButton.clicked.connect(self.save_df)

        closeButton = QtWidgets.QPushButton("close", self)
        closeButton.clicked.connect(self.close_app)

        # grab_btn=QtWidgets.QPushButton('ScreenShot')
        # grab_btn.clicked.connect(self.screenshot)


        # grid.addWidget(station_widget, 0, 0)
        grid.addWidget(param_widget, 0, 1)
        grid.addWidget(saveButton, 0, 2)
        grid.addWidget(reload_button, 0, 3)
        grid.addWidget(closeButton, 0, 4)
        # grid.addWidget(grab_btn, 0, 5)

        # self.setCentralWidget(station_widget)
        self.params = {
            0: 'obs_code',
            1: 'obs_name',
            2: 'obs_time',
            3: 'water_temperature',
            4: 'temperature',
            5: 'pressure',
            6: 'wind_direction',
            7: 'wind_speed',
            8: 'wind(U)',
            9: 'wind(V)',
            10: 'humidity',
            11: 'visibility'}

        # self.df = pd.read_csv('sample.csv')
        self.param_num = 8 # wind U
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        # grid.addWidget(self.canvas, 1,0)

        grid.addWidget(self.toolbar, 1, 0, 1, 5)
        grid.addWidget(self.canvas, 2, 0, 5, 5)
        # self.setCentralWidget(self.canvas)
        # rectangular selector
        self.toggle_selector = RectangleSelector(
            self.canvas.axes,
            self.onselect,
            drawtype="box",
            useblit=True,
            button=[1],
            interactive=True,
        )
        self.show()

    # def screenshot(self):
    #     screen = QtWidgets.QApplication.primaryScreen()
    #     screenshot = screen.grabWindow( w.winId() )

    #     img_dir = Path('screenshot')
    #     suffix = 0
        
    #     prefix = f'{self.port} {self.param_num}_'
    #     files = sorted([x for x in img_dir.glob('*.jpg') if x.stem.startswith(prefix)])

    #     if len(files):
    #         suffix = files[-1].stem.split('_')[-1]
    #         suffix  = int(suffix) + 1

    #     name = prefix + str(suffix) + '.jpg'
    #     save_path =  img_dir / name
    #     screenshot.save(save_path.as_posix(), 'jpg')

    def close_app(self):
        self.close()
    
    def param_changed(self, s):  # s is a str
        param_map = {
            'U': 8,
            'V': 9,
            '기온(℃)': 4,
            '수온(℃)': 3,
            '기압(hPa)': 5,
            '습도(%)': 10,
            '시정(20km)': 11}
        self.param_num = param_map[s]
        self.update_plot()

    def update_filename(self):
        self.fname = QFileDialog.getOpenFileName()[0]
        self.load_df()
        self.update_plot()


    def load_df(self):
        SOURCE = Path(self.fname)
        DEST = SOURCE.parent / f'{SOURCE.stem}_qc.csv'
        self.export_path = DEST
        self.df = pd.read_csv(
            SOURCE, parse_dates=["obs_time"])        

    def qc_vis(self):
        assert 'visibility' in self.df.columns
        length = 7
        vis  = (
            self.df
            [['visibility']] 
            # .assign(vis = lambda x: x.visibility.interpolate('linear', limit=60).rolling(10).mean())
            .assign(vis = lambda x: x.visibility.copy())
            .assign(**dict(
                grad = lambda x: x.vis.diff().abs() + x.vis.diff(-1).abs(),
                rolling_std = lambda x: x.vis.rolling(length, center=True).std(),
                nan_mask = lambda x: x.vis.isna().rolling(length, center=True).sum()
                ))
            .assign(qc_mask = lambda x: x.grad > 2 * x.rolling_std)
            .astype(dict(qc_mask=int))
            .assign(qc_mask = lambda x: np.where(x.nan_mask, 2, x.qc_mask))
            .assign(visibility = lambda x: np.where(x.qc_mask==1, np.nan, x.visibility))
            ['visibility']
            # .interpolate('linear', limit=60)
            # .rolling(10)
            # .mean()
            )
        
        na_count_before_qc = self.df.visibility.isna().sum()
        self.df.loc[vis.isna(),  'visibility'] = np.nan
        num_changed = self.df.visibility.isna().sum() - na_count_before_qc
        msg = '{} changed through qc'.format(num_changed)
        logging.debug(msg)
      

    def save_df(self):
        self.qc_vis() # update vis inplace
        """
        0: 'obs_code',
        1: 'obs_name',
        2: 'obs_time',
        3: 'water_temperature'
        4: 'temperature'
        5: 'pressure'
        6: 'wind_direction'
        7: 'wind_speed'
        8: 'wind(U)'
        9: 'wind(V)'
        10: 'humidity'
        11: 'visibility
        """

        
        data_cols = [col for i, col in enumerate(self.df.columns ) if i >=3]
        df_5T = self.df.groupby(['obs_name', pd.Grouper(key=self.params[2],freq='5T', closed='right', label='right')])[data_cols].mean().reset_index()
        df_5T = df_5T.iloc[:288]

        df_5T[self.params[7]] = np.sqrt(df_5T[self.params[8]] ** 2 + df_5T[self.params[9]] ** 2)
        wd =  np.rad2deg(np.arctan2(df_5T[self.params[8]], df_5T[self.params[9]]))
        df_5T[self.params[6]] = np.where(wd>=0, wd, 360 - np.abs(wd))
        df_5T = df_5T.sort_values([self.params[1],self.params[2]])
        df_5T = df_5T.join(self.df[[self.params[0]]], how='left')
        self.df[self.df.columns].to_csv(self.export_path, index=False, encoding='cp949')
        df_5T_dest = self.export_path.parent / f'{self.export_path.stem}_5T.csv'
        df_5T[self.df.columns].to_csv(df_5T_dest, index=False, encoding='cp949')
            
    def update_plot(self, memory={}):
        # self.xdata = pd.Index(self.df.query("obs_name==@self.port")["obs_time"])
        # self.ydata = self.df.loc[self.df.obs_name == self.port, self.param_num]
        self.xdata = pd.Index(self.df[self.params[2]])
        self.ydata = self.df[self.params[self.param_num]]
        ymin, ymax = self.ydata.min(), self.ydata.max()
        buffer = self.ydata.quantile(0.5) - self.ydata.quantile(0.4)
        buffer = max(buffer, 0.1) 
        ymin, ymax = ymin - buffer, ymax + buffer

        if (
            self.params[self.param_num] == memory.get("parameter")
            and self.fname == memory.get("fname")
            # and self.port == memory.get("port")
            
        ):
            self.line.set_data(self.xdata, self.ydata)

        else:
            self.canvas.axes.cla()  # Clear the canvas.
            (self.line,) = self.canvas.axes.plot(
                self.xdata,
                self.ydata,
                c="b",
                marker="o",
                markerfacecolor="r",
            )

            if self.ydata.isna().all():
                ylim = {
                    3: [-10,50],
                    4: [-30,50],
                    5: [950,1050],
                    8: [0,20],
                    9: [0,20],
                    10: [20,100],
                    11: [0,20000],
                }[self.param_num]
                self.canvas.axes.set_ylim(ylim)

        # self.toolbar.actions()[0].trigger()
        # ed.connect(home_callback)
        # Trigger the canvas to update and redraw.
        self.canvas.draw()

        # memory["port"] = self.port
        memory["parameter"] = self.params[self.param_num]
        memory["fname"] = self.fname
        # self.fig.canvas.draw_idle()

    def onselect(self, eclick, erelease):
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
        idx_x = (self.df[self.params[2]] > ts_x0) & (self.df[self.params[2]] < ts_x1)
        idx_y = (self.df[self.params[self.param_num]] > y0) & (self.df[self.params[self.param_num]] < y1)
        idx = idx_x & idx_y

        param_nums =  [6,7,8,9] if self.param_num  in [8,9] else [self.param_num]
        parameter = [self.params[num] for num in param_nums]
        # self.df.loc[(self.df.obs_name == self.port) & idx, parameter] = np.nan
        self.df.loc[idx, parameter] = np.nan
        self.update_plot()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec_())

