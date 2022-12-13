import pandas as pd
from pathlib import Path
import dask.dataframe as dd
import logging
import numpy as np
#from pyarrow.parquet import ParquetFile
#import pyarrow as pa 
from utils import setup_logging

# GLOBALS
START_TYPE = "0"
END_TYPE = "1"
CONSEC_ID_COL = "video_consecutive_id"
GAP_COL = "gap"
SESSION_ID_COL = "session_id"
SEG_REP_COL = "seg_rep_count"

N_PARTS = 10

# META ARGS - Only for DASK
SEG_META_ARG = \
                    {'end_point': "float", 
                     'local_start_time':"int64", 
                     'speed': "float", 
                     'start_point': "float"}

CONSEC_META_ARG = ("video_id", "int64")

START_TIME_META_ARG = ("local_start_time", "int64")
SEG_EXPLODE_META = {"user_id": object, 
                    "video_id": object, 
                    "ccid": object,
                    "seq": object
                   }
SEG_EXT_META_ARG = dict({"user_id": object, 
                         "video_id": object, 
                         #"ccid": object,
                         #"index": "int64",
                         CONSEC_ID_COL: "int64",
                         GAP_COL: float,
                         SESSION_ID_COL: "int64"
                        }, **SEG_META_ARG)

# NOT USED YET 
SEG_REP_META_ARG = dict({SEG_REP_COL: "int64", 
                         "user_id": object, 
                         CONSEC_ID_COL: "int64", 
                         SESSION_ID_COL: "int64" })
    


# GENERAL STRUCTURE
# 1. Sequences of segments to segment rows with extracted segment info
# 2. Sort all segments by start timestamp
# 3. For each user, calculate the consecutive video id 
# 4. For each user, consecutive video_id, calculate the segment gap
# 5. For each user, consecutive video_id, calculate the interaction session id 
# 6. For each user, consecutive video_id, interaction session_id, calculate the segment repetition


def explode_sequences_to_segments(sequences_df, meta_arg=SEG_META_ARG):
    """Explodes each segment to a separate row, maintaining user_id and video_id.
    Param: sequences, Dask DataFrame
    Returns: DataFrame (reset_index)"""
    logging.info("Exploding each sequence into separate segment rows")
    # Slow implemenentation, but working
    exploded = sequences_df.groupby(["user_id", "video_id"])\
        .apply(lambda x:x["seq"].str["segment"].explode(), meta=("seq", object))
    logging.info("Extracting context from segments")
    return exploded.compute().apply(pd.Series).reset_index().drop(columns="level_2")

    #return dd.from_pandas(exploded.compute().apply(pd.Series).reset_index().drop(columns="level_2"), npartitions=N_PARTS)


    # PREVIOUS - SOME ERROR WITH THE MERGE
    exploded_sequences = sequences_df.merge(
            sequences_df["seq"].str["segment"].explode().to_frame(), 
            left_index=True,
            right_index=True)
    
    logging.info("Extracting segment keys to columns")
    return pd.concat(
        [exploded_sequences.drop(columns=["seq_x", "seq_y"]), 
         exploded_sequences["seq_y"].apply(pd.Series)#, meta=meta_arg)
        ], 
        axis=1)

    # ALTERNATIVE - Slower
    #return sequences_df.groupby(["user_id", "video_id"])\
    #            .apply(lambda x: x["seq"].str["segment"].explode())
    
    
def calculate_consecutive_user_video_ids(segments_df):
    """Calculates the consecutive video ids for each user to 
    generate the global interaction session ids"""
    # FIX CUMSUM META
    sorted_segments = segments_df.sort_values("local_start_time")#by="seq", key=lambda x: x.str["local_start_time"])
    logging.info("Calculating the consecutive video ids for each user")
    return sorted_segments.assign(video_consecutive_id=sorted_segments
        .groupby("user_id", group_keys=False)
        .apply(lambda group: (group["video_id"].shift() != group["video_id"]).cumsum(), #meta=CONSEC_META_ARG
              )
        )

def calculate_segment_gap(segments_df, video_id_col=CONSEC_ID_COL):
    """For each row, calculate the difference between the current row and the next.
    The end time stamp is calculated using true watch time, accounting for PBR."""
    logging.info("Calculating the segment gap for each user and consecutive video id")
    # FIX META #shift meta=START_TIME_META_ARG
    #computed_df = segments_df.compute()
    return segments_df.assign(
        gap=segments_df.
        groupby(["user_id", video_id_col])["local_start_time"].shift(-1,) -\
            (segments_df["local_start_time"] +\
             ((segments_df["end_point"] - segments_df["start_point"]) / segments_df["speed"]))
    )
    
    
    # DASK VERSION
    return segments_df.assign(
        gap=segments_df.
        groupby(["user_id", video_id_col])["local_start_time"].shift(-1, meta=START_TIME_META_ARG) -\
            (segments_df["local_start_time"] +\
             ((segments_df["end_point"] - segments_df["start_point"]) / segments_df["speed"]))
    )
    
def set_global_session_ids(x, session_thresh=3600, gap_col=GAP_COL, session_id_col=SESSION_ID_COL):
    """Very slow implementation"""
    session_counter = 0
    for idx in x.index:
        x.loc[idx, session_id_col] = session_counter
        if x.loc[idx, gap_col] >= session_thresh:
            session_counter += 1
    return x

def calculate_session_ids(segments_df, video_id_col=CONSEC_ID_COL):
    """Calculate interaction session ids, taking into consideration global timestamp and gap"""
    segments_df_w_session_id = segments_df.assign(session_id=np.nan)
    print("inside session", segments_df_w_session_id.head())
    logging.info("Calculating the interaction session ids for each user and consecutive video id.")
    return segments_df_w_session_id.groupby(["user_id", video_id_col])\
        .apply(set_global_session_ids)#, meta=SEG_EXT_META_ARG)

def count_seg_reps(intervals, overlap_thresh=5):
    """ Naïve approach - Counts the total number of overlapping intervals,
    given a threshold. 
    Improvement: Sweep line approach: https://www.baeldung.com/cs/finding-all-overlapping-intervals"""
    n = len(intervals)
    # sort intervals on start_time, then duration
    sorted_intervals = sorted([
        [start, end, (end - start), idx] for idx, (start, end) in enumerate(intervals)], 
        key=lambda x: (x[0], -x[2]))
    
    overlap_count = 0
    for idx, first in enumerate(sorted_intervals):
        for second in sorted_intervals[idx + 1:]:
            if max(first[0], second[0]) <= min(first[1], second[1])-overlap_thresh:
                overlap_count += 1

    return overlap_count

def seg_rep_wrapper(group):
    return count_seg_reps(group[["start_point", "end_point"]].values)

def calculate_segment_repetition(segments_df, 
                                 video_id_col=CONSEC_ID_COL,
                                 session_id_col=SESSION_ID_COL,
                                 seg_rep_col=SEG_REP_COL):
    """Expects a whitelist cleaned df whith session id. 
    For each interaction session, it will calculate the interaction session"""
    logging.info("Calculating segment repetition for each interaction session")
    # FIX META - Might not be possible due to multi index? , meta=(0, "int64")
    return segments_df.groupby(["user_id", video_id_col, session_id_col])\
            .apply(seg_rep_wrapper).rename(seg_rep_col)#.compute().reset_index()

if __name__ == "__main__":
    setup_logging("full_seg_rep_count.log")
    # Loading data
    base_path = Path("~/fall_project/MOOCCubeX/")
    results_path = Path("./results")
    relations_path = base_path / "relations"
    exploded_path = relations_path / "clean_user2sequences_thresh_20_partitions_30_exploded"
    
    # LOAD TEST SAMPLE - Already contains whitelisted users, ccid, video_id extracted
    #pf = ParquetFile(relations_path / "clean-user-video-sequences_thresh_20.parquet")
    #sampled_clean_set = next(pf.iter_batches(batch_size = 5000)) 
    #cleaned_user2video_sample = pa.Table.from_batches([sampled_clean_set]).to_pandas()
    #dd_clean_user2video_sample = dd.from_pandas(cleaned_user2video_sample, npartitions=N_PARTS)\
    #    .drop(columns="index")
    
    segments = dd.read_parquet(exploded_path, npartitions=N_PARTS).compute()
    #segments = dd.read_parquet(exploded_path / "part.0.parquet", npartitions=N_PARTS).compute()[:100]
    

    # Pre-processing
    #segments = explode_sequences_to_segments(cleaned_user2video)#.reset_index()
    print(segments.head())
    segments_consecutive_id = calculate_consecutive_user_video_ids(segments)
    print(segments_consecutive_id.head())
    #print(segments_consecutive_id.columns)
    segments_consec_gap = calculate_segment_gap(segments_consecutive_id)
    print(segments_consec_gap.head())
    segments_interaction_sessions = calculate_session_ids(segments_consec_gap)#.reset_index()
    #print(segments_interaction_sessions.columns)
    #logging.info("ABOUT TO CALCULATE")
    print(segments_interaction_sessions.head())
    
    # Calucalate segment repetition
    segments_w_repetition_count = calculate_segment_repetition(segments_interaction_sessions)
    
    print(segments_w_repetition_count)#.head())
   
    
    # DATA DESCRIPTION and print
    ## Describe, value counts etc
    seg_rep_count_dist = segments_w_repetition_count.value_counts(SEG_REP_COL).reset_index().rename(columns={0:"frequency"})
    #seg_rep_describe = segments_w_repetition_count[SEG_REP_COL].describe()
    logging.info("## DATA ##")
    logging.info("Repetition count distribution")
    logging.info(seg_rep_count_dist.head(20))
    
    sessions_length_per_user = segments_w_repetition_count.groupby(["user_id", CONSEC_ID_COL])[SESSION_ID_COL].nunique().to_frame().sum(level=[0])
    logging.info("Session length description")
    sessions_length_description = sessions_length_per_user.describe()
    logging.info(sessions_length_description)
    logging.info(f"Sessions over length 5 {(sessions_length_per_user[sessions_length_per_user['session_id'] > 5].shape[0] / sessions_length_per_user.count())}")
    logging.info(f"Sessions over length 10 {(sessions_length_per_user[sessions_length_per_user['session_id'] > 10].shape[0] / sessions_length_per_user.count())}")
    
    
    # STORING
    ## Fully-expanded segments df (segments_interaction_sessions)
    ## segments_w_repetition_count
    ## Segment length description + value counts
    logging.info("Storing segments with context")
    segments_interaction_sessions.to_parquet(results_path / "mooc_segments_w_context.parquet.gzip", compression="gzip")
    logging.info("Storing interaction session to video_id mapping")
    segments_interaction_sessions[["user_id", "video_id", CONSEC_ID_COL, SESSION_ID_COL]]\
        .merge(segments_w_repetition_count, on=["user_id", CONSEC_ID_COL, SESSION_ID_COL], validate="many_to_one")\
        .to_parquet(results_path / "mooc_interaction_session2video_id.parquet.gzip", compression="gzip")
    
    logging.info("Storing segments with repetition count")
    segments_w_repetition_count.to_parquet(results_path / "mooc_segments_w_repetition_count.parquet.gzip", compression="gzip")
    logging.info("Storing Repetition count distribution")
    seg_rep_count_dist.to_parquet(results_path / "mooc_segment_repetition_distribution.parquet")
    logging.info("Storing interactions session lengths")
    sessions_length_per_user.to_parquet(results_path / "mooc_interaction_sessions_length.parquet")
    logging.info("Storing interaction session length description")
    sessions_length_description.to_parquet(results_path / "mooc_interaction_sessions_description.parquet")
    logging.info("## COMPLETE ##")

    