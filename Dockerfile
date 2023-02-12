FROM jupyter/scipy-notebook
RUN pip install voila ipyfilechooser ipydatetime geopandas pyarrow geopy ipyleaflet
COPY . /home/jovyan/work