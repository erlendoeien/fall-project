import pandas as pd
from pathlib import Path
import dask.dataframe as dd
import logging


def load_parquet_as_dd(file_name, **kwargs):
    logging.info(f"Loading parquet: {file_name}")
    return dd.from_pandas(pd.read_parquet(file_name), npartitions=10, **kwargs)

def merge_video_ccid(user2video, video_id2ccid):
    logging.info(f"Merging {user2video} with {video_id2ccid}")
    sequences = user2video.explode("seq").reset_index()
    return sequences.assign(video_id=sequences["seq"].str["video_id"]).merge(video_id2ccid)

def count_user_ccid_interactions(interactions_df, count_col, video_id="ccid"):
    """Must contain either the video_id or ccid to group by"""
    logging.info("Counting user-ccid interactions")
    return (interactions_df.groupby([video_id, "user_id"]).size()
                            .reset_index()
                            .rename(columns={0: count_col})
                            .sort_values(count_col, ascending=False))

def generate_blacklist_users(interaction_count_df, count_col, rep_thresh=20):
    """Generating a list of users who have watched any given ccid (not counting segments)
    more than `rep_thresh` times."""
    logging.info("Generating user blacklist")
    return interaction_count_df[interaction_count_df[count_col] > rep_thresh]["user_id"].unique()

def filter_users(user2video, blacklist):
    logging.info("Filtering out blacklisted users")
    return user2video[~user2video["user_id"].isin(blacklist)]

def count_segments(user2video):
    return user2video["seq"].str["segment"].str.len().sum()

def setup_logging():
    logging.basicConfig(filename="blacklist_cleaning.log", level=logging.DEBUG, filemode="w")


if __name__ == "__main__":
    setup_logging()
    # Loading data
    base_path = Path("~/fall_project/MOOCCubeX/")
    relations_path = base_path / "relations"
    user2video_simple_path, video_id2ccid_path = "user-video-simple.parquet.gzip", "video_id-ccid.txt"
    user2video_full = "user-video.json"
    #user2video_dd = load_parquet_as_dd(relations_path / user2video_path)
    video_id2ccid = dd.from_pandas(pd.read_csv(relations_path / "video_id-ccid.txt", sep="\t",
        names=["video_id", "ccid"]), npartitions=10)
    user2video_dd = dd.from_pandas(pd.read_json(relations_path / user2video_full, lines=True), npartitions=10)

    # Merging for blacklist generation with repetition threshold at 20
    user2video_ccid = merge_video_ccid(user2video_dd, video_id2ccid)
    # Counting views per user per ccid/video_id
    view_count_col = "num_watched"
    count_df = count_user_ccid_interactions(user2video_ccid, view_count_col)
    blacklist = generate_blacklist_users(count_df, view_count_col, 20)
    

    clean_user2video_ccid = filter_users(user2video_ccid, blacklist)

    num_segments = count_segments(user2video_dd)
    clean_num_segments = count_segments(clean_user2video_ccid)

    logging.info("Total number of segments before cleaning:", num_segments)
    logging.info("Total number of segments after cleaning:", clean_num_segments)
    logging.info(f"-----> Dataset reduction {(num_segments/clean_num_segments)*100:.2f}")
    logging.info("Number of blacklisted users", blacklist.shape[0])
    logging.info("Number of total unique users:", user2video_dd["user_id"].unique().shape[0])
