import pandas as pd
import numpy as np
from pathlib import Path
from utils import setup_logging
import logging
import dask.dataframe as dd

N_PARTS = 10
META_ARG = { 
    "timestamp": "int64", 
    "action_type": object,
    "item_id": object,
    "cursor_time": float,
    "source": object,
    "user_answer": object,
    "platform": object,
    "user_id": "int64"
}

def get_only_lecture_events(events_df):
    """Returns only the lecture events."""
    logging.info("Fetching only lecture events")
    return events_df[events_df["item_id"].str.startswith("l")]

def remove_consecutive_events(lectures_df):
    """Removes consecutive events with the same item_id. Will only return the "enter" event.
    Must make sure that the next item event is correct wrt. time."""
    logging.info("Removing consecutive same lecture events")
    return lectures_df.sort_values("timestamp").groupby("user_id", group_keys=False).apply(lambda group: group[group.shift(1)["item_id"] != group["item_id"]], meta=META_ARG)

def check_time_and_user_sort(ednet_df):
    """Rudimentary check to see if data is sorted by timestamp and user_ids.
    IMPROVE: check if the consecutive user_ids are monotonically increasing"""
    return ednet_df["timestamp"].compute().is_monotonic_increasing and ednet_df["user_id"].compute().is_monotonic_increasing

def get_sequence_len_description(sequences):
    return sequences.groupby("user_id")["item_id"].count().describe()

if __name__ == "__main__":
    setup_logging("ednet_cleaning.log")
    ednet_path = Path("../EdNet")
    results_path = Path("./results")
    
    # Load data
    ednet = dd.from_pandas(pd.read_feather(ednet_path / "KT4_merged.feather"), npartitions=N_PARTS)
    
    #is_sorted = check_time_and_user_sort(ednet)
    #logging.info(f"User and timestamp is sorted: {is_sorted}")
    
    # Only lectures
    lectures_df = get_only_lecture_events(ednet)
    
    # Only enter event
    single_events = remove_consecutive_events(lectures_df)
    
    # Sequence description
    logging.info("User sequence length description")
    sequence_len = get_sequence_len_description(single_events).compute()
    logging.info(sequence_len)
    
    # STORING
    logging.info("Storing ednet sequence lengths description")
    sequence_len.to_frame().to_parquet(results_path / "ednet_kt4_lecture_seq_len_describe.parquet")
    
    logging.info("Storing ednet sequences")
    single_events.compute().to_parquet(ednet_path / "single_lecture_events_kt4.parquet")
    
    """
    2022-11-07 11:28:43,463 | INFO: 
    count    42828.000000    
    mean        12.196414
    std         27.497925
    min          1.000000
    25%          1.000000
    50%          3.000000
    75%         11.000000
    max        744.000000
    Name: item_id, dtype: float64 
    
    There was some difference (prev_mean: 12.1865, prev_std: 27.4635
    """