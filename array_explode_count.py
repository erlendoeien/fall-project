import pandas as pd
from pathlib import Path
import dask.dataframe as dd
import logging
import numpy as np
from utils import setup_logging
from segment_repetition_count import (explode_sequences_to_segments,
    calculate_consecutive_user_video_ids, calculate_segment_gap, calculate_session_ids, calculate_segment_repetition,
        SEG_REP_COL, CONSEC_ID_COL, SESSION_ID_COL)
import logging
import sys
    
N_PARTS = 1

if __name__ == "__main__":
    partition_num = int(sys.argv[1])
    setup_logging(Path("./logs") / "explode_and_count" / f"part.{partition_num}.log")
    
    # FOR TESTING
    #setup_logging(f"part.{partition_num}.log")
    
    # Loading data
    base_path = Path("~/fall_project/MOOCCubeX/")
    results_path = Path("./results")
    relations_path = base_path / "relations"
    
    exploded_path = relations_path / "clean_user2sequences_thresh_20_partitions_30_exploded"
    session2video_id_path = relations_path / "session2video_id_clean_thresh_20_partitions_30"
    context_path = relations_path / "context_user2sequences_clean_thresh_partitions_30"
    session_len_path = results_path / "mooc_sequence_length"
    partitions_path = relations_path / "clean_user2sequences_thresh_20_partitions_30"
    
    in_out_name = f"part.{partition_num}.parquet"
    
    logging.info(f"Loading partition {in_out_name}")
    print(f"Loading partition {in_out_name}")
    cleaned_user2video_part = dd.read_parquet(partitions_path / in_out_name, npartitions=N_PARTS)
    
    # Pre-processing
    print("Exploding sequences to segments")
    segments_df = explode_sequences_to_segments(cleaned_user2video_part)
    
    
    # Pre-processing
    #print(segments_df.head())
    segments_consecutive_id = calculate_consecutive_user_video_ids(segments_df)
    #print(segments_consecutive_id.head())
    #print("Test uniqueness of consec IDS")
    #print(segments_consecutive_id.groupby("user_id")[CONSEC_ID_COL].apply(lambda x: x.is_unique))
    
    # TEST
    #segments_consecutive_id.to_parquet("test_partition_0.parquet")
        
    segments_consec_gap = calculate_segment_gap(segments_consecutive_id)
    print(segments_consec_gap.head())

    segments_interaction_sessions = calculate_session_ids(segments_consec_gap)
    print(segments_interaction_sessions.head())
    
    # Calucalate segment repetition
    segments_w_repetition_count = calculate_segment_repetition(segments_interaction_sessions)
    
    print(segments_w_repetition_count)
   
    # DATA DESCRIPTION and print
    ## Describe, value counts etc
    seg_rep_count_dist = segments_w_repetition_count.value_counts(SEG_REP_COL).reset_index().rename(columns={0:"frequency"})
    logging.info("## DATA ##")
    logging.info("Repetition count distribution")
    logging.info(seg_rep_count_dist.head(20))
    
    sessions_length_per_user = segments_w_repetition_count.reset_index().groupby(["user_id", CONSEC_ID_COL])[SESSION_ID_COL].nunique().to_frame().sum(level=[0])
    logging.info("Session length description")
    sessions_length_description = sessions_length_per_user.describe()
    logging.info(sessions_length_description)
    logging.info(f"Sessions over length 5 {(sessions_length_per_user[sessions_length_per_user['session_id'] > 5].shape[0] / sessions_length_per_user.count())}")
    logging.info(f"Sessions over length 10 {(sessions_length_per_user[sessions_length_per_user['session_id'] > 10].shape[0] / sessions_length_per_user.count())}")
         
    # STORING
    ## Fully-expanded segments df (segments_interaction_sessions)
    logging.info("Storing segments with context")
    segments_interaction_sessions.to_parquet(context_path / in_out_name)
    logging.info("Storing interaction session with context, left wise")
    segments_w_repetition_count.reset_index()\
        .merge(segments_interaction_sessions, on=["user_id", CONSEC_ID_COL, SESSION_ID_COL], how="left", validate="one_to_many")\
        .to_parquet(session2video_id_path / in_out_name)
    #segments_interaction_sessions[["user_id", "video_id", CONSEC_ID_COL, SESSION_ID_COL]]\
    #    .merge(segments_w_repetition_count, on=["user_id", CONSEC_ID_COL, SESSION_ID_COL], validate="many_to_one")\
    #    .to_parquet(session2video_id_path / in_out_name)
    
    logging.info("Storing interactions session lengths")
    sessions_length_per_user.to_parquet(session_len_path / in_out_name)
    logging.info("## COMPLETE ##")
    