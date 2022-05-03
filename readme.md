# install
conda env create --file environment.yml --prefix venv

# explanation
need to be executed in code, spyder, pycharm interactive mode.

before calling QC(dataframe)
make sure dataframe follow pd.melt format
ex. ['datetime', 'id_var1', 'id_var2', 'id_var3', 'variable', 'value'] 

**datetime, variable, value must be included in dataframe.columns**

**conf/config.yml must contain additional slice_variables**
ex. slice_var: [station, year, variable]

matplotlit.widgets.radiobutton automatically generated for each slice_var

# TODO
use panel widget instead of matplotlib

# CHALLENGES
when using panel, matplotlib interactive bounding box not showing when dragged.




