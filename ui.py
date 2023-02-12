from IPython.display import display, clear_output, Markdown
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Dropdown, Output
from map_tools import get_map

from datetime import date

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt

from typing import Iterable

from column_names import *
from modes import *
from errors import *
from parameters import *

def mode_highlighter_factory(mode: str):
    def highlight_mode(series, color="yellow"):
        s = pd.Series(data=False, index=series.index)
        return [f'background-color: {color}' if series[MODE] == mode else '' for column in s.index]
    return highlight_mode

def highlight_illegal_sr(series, color="yellow"):
    s = pd.Series(data=False, index=series.index)
    return [f'background-color: {color}' if series[ILLEGAL_SR] else '' for column in s.index]

def highlight_connection_loss(series, color="yellow"):
    s = pd.Series(data=False, index=series.index)
    return [f'background-color: {color}' if series[CONNECTION_LOSS] else '' for column in s.index]
    
highlighters = {
    TR_EVENT: mode_highlighter_factory(TR),
    ILLEGAL_SR_EVENT: highlight_illegal_sr,
    CONNECTION_LOSS_EVENT: highlight_connection_loss, 
}

def get_event_plot(dataset: pd.DataFrame) -> Figure:
    counts = get_event_counts(dataset)
    fig, ax = plt.subplots()
    if len(counts) > 0:
        counts.groupby("Datum").sum().plot.bar(ax=ax)
        ax.set_title("Počty událostí podle data")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        ax.text(.5, .5, "Žádná událost", ha="center", va="center")
    return fig

def get_event_counts(dataset: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(dataset.groupby([dataset[TIME].dt.date, TRAIN_NUMBER], dropna=False).size(), columns=[f"Počet"])
    df.index.set_names("Datum", level=TIME, inplace=True)
    return df

def render_summary(dataset: pd.DataFrame, mode: str, data: pd.DataFrame):
    counts = get_event_counts(dataset)
    plot = get_event_plot(dataset)
    display(Markdown(f"Počet nalazených událostí: {len(dataset)}"))
    display(Markdown(f"Počty událostí podle vlaků"))
    left, right = widgets.Output(), widgets.Output()
    with left:
        s = counts.T.style
        if len(counts):
            s.applymap_index(lambda v: "border-width: thin; border-style: solid", axis="columns")
            s.applymap(lambda v: "border-width: thin; border-style: solid")
        display(s)
    with right:
        display(plot)
    display(VBox([left, right]))
    display(Markdown(f"Celkem postižených vlaků: {counts.index.nunique()}"))

def render_list(dataset: pd.DataFrame, mode: str, data: pd.DataFrame):
    display(Markdown(f"Výpis všech nalezených událostí: {mode} - celkový počet záznamů: {len(dataset)}"))
    display(dataset[DISPLAY_COLUMNS])

def render_detail(dataset: pd.DataFrame, mode: str, data: pd.DataFrame):
    display(Markdown(f"Výpis všech záznamů pro vybraný vlak"))
    
    def change_selection(selector, new_value):
        selector.value = new_value

    full_selection = dataset[TIME].dt.date.unique().tolist()
    empty_selection = []
    
    date_selector = widgets.SelectMultiple(
        options=full_selection,
        value=full_selection,
        description='Datum',
        disabled=False
    )
    
    train_selector = widgets.Dropdown(description='Číslo vlaku')
    
    select_all_button = widgets.Button(description='Vybrat vše')
    unselect_all_button = widgets.Button(description='Zrušit výběr')
    
    output = widgets.Output()

    select_all_button.on_click(lambda _: change_selection(date_selector, full_selection))
    unselect_all_button.on_click(lambda _: change_selection(date_selector, empty_selection))
    
    def redraw(date_range: Iterable[date], train_number: int):
        record_filter = (data[TIME].dt.date.isin(date_range)) & (data[TRAIN_NUMBER] == train_number)
        subframe = data[record_filter].copy()
        subframe.sort_values(TIME, ascending=True, inplace=True)
        styler = subframe.style.apply(highlighters[mode], axis='columns')
        columns_to_hide = [column for column in subframe.columns if column not in DISPLAY_COLUMNS]
        styler.hide(columns_to_hide, axis="columns")
        return styler
    
    def update_date(change_event):
        date_range = change_event["new"]
        train_selector.value = None
        train_filter = dataset[TIME].dt.date.isin(date_range)
        train_selector.options = sorted(pd.unique(dataset.loc[train_filter, TRAIN_NUMBER]).dropna())
    
    def update_train(change_event):
        with output:
            clear_output()
            display(redraw(date_selector.value, change_event["new"]))
    
    date_selector.observe(update_date, names="value", type="change")
    train_selector.observe(update_train, names="value", type="change")
    
    update_date({"new": full_selection})
    
    display(HBox([date_selector, VBox([select_all_button, unselect_all_button])]))
    display(train_selector, output)
        
def render_ui(loaded_file_name: str, data: pd.DataFrame):
    datasets = {
        TR_EVENT: data[(data[MODE]==TR) & (data[PREV_MODE]!=data[MODE])],
        ILLEGAL_SR_EVENT: data[data[ILLEGAL_SR]],
        CONNECTION_LOSS_EVENT: data[data[CONNECTION_LOSS]]
    }
    
    accordion_fields = (
        {"name": "Souhrn", "renderer": render_summary},
        {"name": "Seznam", "renderer": render_list},
        {"name": "Detail", "renderer": render_detail}
    )

    accordions = {
        TR_EVENT: {field["name"]: (widgets.Output(), field["renderer"]) for field in accordion_fields},
        CONNECTION_LOSS_EVENT: {field["name"]: (widgets.Output(), field["renderer"]) for field in accordion_fields},
        ILLEGAL_SR_EVENT: {field["name"]: (widgets.Output(), field["renderer"]) for field in accordion_fields},
    }

    tab_outputs = {
        TR_EVENT: widgets.Output(),
        CONNECTION_LOSS_EVENT: widgets.Output(),
        ILLEGAL_SR_EVENT: widgets.Output(),
    }

    for mode, tab_output in tab_outputs.items():
        with tab_output:
            display(Markdown(f"### {mode} - nalezené události"))
            for field, (output, renderer) in accordions[mode].items():
                with output:
                    display(renderer(datasets[mode], mode, data))
                display(widgets.Accordion(children=[output], titles=[field]))
    
    rbcs = data[[RBC_ID, RBC_NAME]].drop_duplicates()
    rbc_output = widgets.Output()
    with rbc_output:
        display(
            Markdown(f"#### RBC v souboru: {len(rbcs)}"),
            rbcs.style.hide(axis="index")
        )

    time_range = pd.DataFrame({"Od": data[TIME].min(), "Do": data[TIME].max()}, index=pd.Index([0]))
    records_output = widgets.Output()
    with records_output:
        display(Markdown("#### Záznamy"))
        display(time_range.T.style.hide(axis="columns"))
    
    left, right, plot_output = Output(), Output(), Output()
    
    dropdown = Dropdown(
        description="Vyber datum",
        options = data[TIME].dt.date.unique().tolist()
    )
        
    def redraw_plot(change_event):
        with plot_output:
            clear_output()
            fig, ax = plt.subplots()
            data.loc[data[TIME].dt.date.isin([change_event["new"]])].resample("H", on=TIME)[TRAIN_NUMBER].nunique().plot(ax=ax)
            ax.set_title(f"Počty přihlášených vlaků {dropdown.value} podle hodin")
            ax.set_ylabel("Počet komunikujících vlaků")
            display(fig)
    dropdown.observe(redraw_plot, names="value", type="change")
    
    redraw_plot({"new": dropdown.options[0]})
    
    with right:
        display(dropdown)
    
    fig_left, ax = plt.subplots()
    data.resample("D", on=TIME)[TRAIN_NUMBER].nunique().plot.bar(ax=ax)
    ax.set_title("Počty přihlášených vlaků podle dní (celkově za den)")
    ax.set_ylabel("Počet komunikujících vlaků")
    ax.set_xlabel("Datum")
    ax.set_xticklabels(pd.Series(data.resample("D", on=TIME).groups.keys()).dt.strftime("%Y-%m-%d"))
    with left: display(fig_left)
    
    active_trains_output = widgets.Output()
    with active_trains_output:
        display(
            Markdown("#### Průběh počtu vlaků přihlášených k daným RBC"),
            HBox([left, VBox([right, plot_output]),])
        )

    day_selection_output = Output()
    
    with day_selection_output:
        display(dropdown)

    area_map, bounding_box = get_map(data)
    area_map.fit_bounds(area_map.bounds)
    tab = widgets.Tab()
    tab.children = list(tab_outputs.values())
    tab.titles = list(tab_outputs.keys())
    output = widgets.Output()
    with output:
        display(
            Markdown(f"### Soubor: '{loaded_file_name}'"),
            HBox([
                rbc_output,
                area_map,
            ]),
            HBox([
                records_output,
                active_trains_output
            ]),
            tab
        )
        area_map.fit_bounds(bounding_box)
    return output