import pandas as pd
from typing import Iterable
from ipywidgets import HTML
from ipyleaflet import Map, TileLayer, Marker, MarkerCluster
import geopandas
from attribution import attribution_string

from functools import partial
import re
from column_names import *

def get_bounding_box(coordinates_list):
    min_lon, min_lat = float("inf"), float("inf")
    max_lon, max_lat = -float("inf"), -float("inf")
    
    for lat, lon in coordinates_list:
        if lat > max_lat: max_lat = lat
        if lat < min_lat: min_lat = lat
        if lon > max_lon: max_lon = lon
        if lon < min_lon: min_lon = lon
    
    return ((min_lat, min_lon), (max_lat, max_lon))

def get_position(name: str, reference_database: geopandas.GeoDataFrame):
    relevant_results: geo.GeoDataFrame = reference_database.loc[reference_database["name"] == name]
    if relevant_results.empty:
        relevant_results = reference_database.loc[reference_database["name"].str.contains(name)]
    return relevant_results.dissolve().to_crs(5514).centroid.to_crs("WGS84")

def get_marker_cluster(rbc_names: pd.Series):
    stations = pd.DataFrame({"name": pd.unique(rbc_names.str.extractall(r"(?P<station>\w+(?: +\w+)*)")["station"])})
    database = geopandas.read_parquet("./geo/stations.parquet")
    stations["position"] = stations["name"].apply(partial(get_position, reference_database = database))
    stations["marker"] = stations.apply(
        lambda row: Marker(location=[row["position"].y, row["position"].x], popup=HTML(row["name"])),
        axis="columns"
    )
    return MarkerCluster(markers=stations["marker"].tolist())

def get_map(data: pd.DataFrame):
    background = TileLayer(
        url='http://tiles.openrailwaymap.org/standard/{z}/{x}/{y}.png',
        attribution=attribution_string
    )
    leaflet_map = Map(zoom=12, scroll_wheel_zoom=False)
    leaflet_map.add_layer(background)
    markers = get_marker_cluster(data[RBC_NAME])
    leaflet_map.add_layer(markers)
    bounding_box = get_bounding_box(marker.location for marker in markers.markers)
    return leaflet_map, bounding_box