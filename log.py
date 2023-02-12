import pandas as pd
import numpy as np

from column_names import *
from modes import *
from errors import *
from parameters import *

def load_data(file_name: str) -> pd.DataFrame:
    data = pd.read_excel(file_name, engine="openpyxl", header=1)
    return data.astype({
        TRAIN_NUMBER: pd.Int64Dtype(), LENGTH: pd.Int64Dtype(),
        MAX_SPEED: pd.Int64Dtype(), AXLE_LOAD: pd.Int64Dtype(),
    })

def parse_events(events_raw: pd.Series) -> pd.DataFrame:
    event_regex = r"RBC *(?P<rbc_id>\d+) *(?P<name>.*) *- *(?P<obu>\d*) *: *(?P<event>.*)"
    mode_regex = r"[mM][oóOÓ][dD] *(?P<mode>\w{2})"
    renaming_rules = {"rbc_id": RBC_ID, "name": RBC_NAME, "event": EVENT, "mode": MODE, "obu": OBU}
    
    event_details = events_raw.str.extract(event_regex)
    mode = events_raw.str.extract(mode_regex)
    result = event_details.join(mode)
    result.rename(renaming_rules, inplace=True, axis="columns")
    return result.astype({OBU: pd.Int64Dtype()})

def is_start_of_mission(event_descriptions: pd.Series) -> pd.Series:
    return event_descriptions.str.contains("vznik vlaku", case=False)

def is_end_of_mission(event_descriptions: pd.Series) -> pd.Series:
    return event_descriptions.str.contains("zánik vlaku", case=False)

def find_previous_non_empty(series: pd.Series, start_from_label: pd.Index) -> pd.Index:
    return series.iloc[0:series.index.get_loc(start_from_label)].last_valid_index()

def find_next_non_empty(series: pd.Series, start_from_label: pd.Index) -> pd.Index:
    return series.iloc[series.index.get_loc(start_from_label)+1:].first_valid_index()

def get_values_on_indices(data: pd.DataFrame, indices: pd.Index) -> pd.DataFrame:
    result = data.loc[data.index.intersection(indices), [MODE, TIME, RBC_ID, TRAIN_NUMBER]].reindex(indices)
    return result
    
def find_adjacent_modes(data: pd.DataFrame) -> pd.DataFrame:
    result_columns = [
        PREV_INDEX, PREV_MODE, PREV_TIME, PREV_NUMBER, PREV_RBC,
        NEXT_INDEX, NEXT_MODE, NEXT_TIME, NEXT_NUMBER, NEXT_RBC,
    ]
    result = pd.DataFrame(columns=result_columns)
    for obu_number, unsorted_subframe in data.groupby(by=OBU, dropna=False):
        subframe = unsorted_subframe.sort_values(by=TIME, ascending=True)
        prev_mode_indices = subframe.apply(
            lambda row: find_previous_non_empty(subframe[MODE], row.name), axis="columns", result_type="expand").astype(pd.Int64Dtype()
        )
        next_mode_indices = subframe.apply(
            lambda row: find_next_non_empty(subframe[MODE], row.name), axis="columns", result_type="expand").astype(pd.Int64Dtype()
        )

        prev_values = get_values_on_indices(subframe, prev_mode_indices)
        next_values = get_values_on_indices(subframe, next_mode_indices)
        
        prev_values[PREV_INDEX] = prev_values.index
        prev_values.index = subframe.index
        prev_values.rename({MODE: PREV_MODE, TIME: PREV_TIME, TRAIN_NUMBER: PREV_NUMBER, RBC_ID: PREV_RBC}, inplace=True, axis="columns")
        
        next_values[NEXT_INDEX] = next_values.index
        next_values.index = subframe.index
        next_values.rename({MODE: NEXT_MODE, TIME: NEXT_TIME, TRAIN_NUMBER: NEXT_NUMBER, RBC_ID: NEXT_RBC}, inplace=True, axis="columns")

        result_subframe = prev_values.join(next_values)
        result = pd.concat([result, result_subframe])
    return result

def compare_with_nans(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    result = (a == b)
    result[pd.isnull(a) & pd.isnull(b)] = True
    return result

def compare_adjacent_modes(data: pd.DataFrame) -> pd.DataFrame:
    prev_rbc_changes = (data[PREV_RBC] != data[RBC_ID]).fillna(True)
    next_rbc_changes = (data[RBC_ID] != data[NEXT_RBC]).fillna(True)
    prev_train_number_changes = (data[PREV_NUMBER] != data[TRAIN_NUMBER]).fillna(True)
    next_train_nember_changes = (data[TRAIN_NUMBER] != data[NEXT_NUMBER]).fillna(True)
    prev_time_differences = data[TIME] - data[PREV_TIME]
    next_time_differences = data[NEXT_TIME] - data[TIME]
    
    return pd.DataFrame({
        PREV_RBC_CHANGE: prev_rbc_changes,
        NEXT_RBC_CHANGE: next_rbc_changes,
        PREV_TIME_DIFFERENCE: prev_time_differences,
        NEXT_TIME_DIFFERENCE: next_time_differences,
        PREV_NUMBER_CHANGE: prev_train_number_changes,
        NEXT_NUMBER_CHANGE: next_train_nember_changes
    })

def is_illegal_sr(data, T_NVOVTRP, rel_eps=0.1) -> pd.Series:
    if rel_eps < 0:
        raise ValueError(f"Relative tollerance cannot be negative! {rel_eps} given")
    suspicious_transition = (data[MODE] == SR) & (data[PREV_MODE].isin([FS, OS])) & (data[NEXT_MODE].isin([OS]))
    lower_bound = T_NVOVTRP * (1 - rel_eps / 2)
    upper_bound = T_NVOVTRP * (1 + rel_eps / 2)
    return suspicious_transition & ((data[NEXT_TIME_DIFFERENCE] / np.timedelta64(1, "s")).between(lower_bound, upper_bound))
    
def is_connection_loss(data) -> pd.Series:
    return (data[EOM]) & (data[RBC_ID]==data[NEXT_RBC]) & (data[PREV_MODE]==FS) & (data[NEXT_MODE].isin([PT, SR]))

def analyse(file_name: str) -> pd.DataFrame:
    data = load_data(file_name)
    data.drop([OBU], axis="columns", inplace=True)
    event_details = parse_events(data[EVENT])
    data.drop([EVENT], axis="columns", inplace=True)
    data = data.join(event_details)
    data[SOM] = is_start_of_mission(data[EVENT])
    data[EOM] = is_end_of_mission(data[EVENT])
    data = data.join(find_adjacent_modes(data))
    data = data.join(compare_adjacent_modes(data))
    data[ILLEGAL_SR] = is_illegal_sr(data, T_NVOVTRP)
    data[CONNECTION_LOSS] = is_connection_loss(data)
    return data