import pandas as pd
from pathlib import Path
import dask.dataframe as dd
import logging
import numpy as np
from utils import setup_logging
from segment_repetition_count import (explode_sequences_to_segments,
    calculate_consecutive_user_video_ids, calculate_segment_gap, calculate_session_ids, calculate_segment_repetition)
import logging
import sys
    
N_PARTS = 5

if __name__ == "__main__":
    partition_num = sys.argv[1]
    #setup_logging(Path("./logs") / f"part.{partition_num}.log")
    setup_logging(f"part.{partition_num}.log")
    
    # Loading data
    base_path = Path("~/fall_project/MOOCCubeX/")
    results_path = Path("./results")
    relations_path = base_path / "relations"
    partitions_path = relations_path / "clean_user2sequences_thresh_20_partitions_30"
    exploded_path = relations_path / "clean_user2sequences_thresh_20_partitions_30_exploded"
    
    in_out_name = f"part.{partition_num}.parquet"
    
    cleaned_user2video_part = dd.read_parquet(partitions_path / in_out_name).repartition(npartitions=N_PARTS)
    
    # Pre-processing
    segments = explode_sequences_to_segments(cleaned_user2video_part)
    logging.info("Storing exploded sequences")
    segments.to_parquet(exploded_path / in_out_name)
    logging.info("## COMPLETE  ##")
    