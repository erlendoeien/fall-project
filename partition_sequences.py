import pandas as pd
import numpy as np
from pathlib import Path
from utils import setup_logging
import dask.dataframe as dd
import logging
import pyarrow as pa

#GLOBALS
N_PARTS = 30

SEQ_STRUCT = pa.struct(
    [
        ("segment", pa.list_(pa.struct(
            [
            ("end_point", pa.float64()),
            ("start_point", pa.float64()),
            ("speed", pa.float64()),
            ("local_start_time", pa.int64())
        ],
            
            ))),
        ("video_id", pa.string()),
    ]
    )



if __name__ == "__main__":
    setup_logging("sequences_partition.log")
    base_path = Path("~/fall_project/MOOCCubeX/")
    results_path = Path("./results")
    relations_path = base_path / "relations"
    
    logging.info("Loading dataset")
    cleaned_user2video = dd.read_parquet(relations_path / "clean-user-video-sequences_thresh_20.parquet", npartitions=10)
    
    logging.info("Sorting and shuffling user_id for partitions")
    user_shuffled_ddf = cleaned_user2video.sort_values("user_id").shuffle(on="user_id", npartitions=N_PARTS, ignore_index=True)
    logging.info("Divisions")
    logging.info(user_shuffled_ddf.divisions)
    
    logging.info(f"Storing {N_PARTS=}")
    # Custom py arrow schemae for the sequences
    user_shuffled_ddf.to_parquet(relations_path / "clean_user2sequences_thresh_20_partitions_30", schema={"seq": SEQ_STRUCT})
    logging.info("## COMPLETE ##")

