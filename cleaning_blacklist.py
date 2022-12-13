import pandas as pd
from pathlib import Path
import dask.dataframe as dd
import logging


def load_parquet_as_dd(file_name, **kwargs):
    logging.info(f"Loading parquet: {file_name}")
    return dd.read_parquet(file_name).repartition(npartitions=10)

def merge_video_ccid(user2video, video_id2ccid):
    """Will exclude some video_ids, as not every video_id has a mapping to a CCID"""
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
    return user2video[~user2video["user_id"].isin(blacklist.compute())]

def count_segments(user2video):
    return user2video["seq"].str["segment"].str.len().sum().compute()

def save_dd_df(df, filename):
    return df.compute().to_parquet(filename)


def setup_logging():
    logging.basicConfig(filename="blacklist_cleaning.log",
            format='%(asctime)s | %(levelname)s: %(message)s',
            level=logging.DEBUG, filemode="w")


if __name__ == "__main__":
    setup_logging()
    # Loading data
    base_path = Path("~/fall_project/MOOCCubeX/")
    relations_path = base_path / "relations"
    user2video_simple_path, video_id2ccid_path = "user-video-simple.parquet.gzip", "video_id-ccid.txt"
    user2video_full = "user-video.json"

    #user2video_dd = load_parquet_as_dd(relations_path / user2video_path)
    video_id2ccid = dd.from_pandas(pd.read_csv(relations_path / "video_id-ccid.txt", sep="\t",
        names=["video_id", "ccid"]), npartitions=20)
    user2video_dd = dd.read_json(relations_path / user2video_full).repartition(npartitions=20)

    # Merging for blacklist generation with repetition threshold at 20
    user2video_ccid = merge_video_ccid(user2video_dd, video_id2ccid)
    # Counting views per user per ccid/video_id
    view_count_col = "num_watched"
    count_df = count_user_ccid_interactions(user2video_ccid, view_count_col)
    blacklist = generate_blacklist_users(count_df, view_count_col, 20)
    

    clean_user2video_ccid = filter_users(user2video_ccid, blacklist)

    # Save files
    #logging.info("Saving blacklist")
    blacklist.to_csv("results/blacklist")
    #logging.info("Saving cleaned user2-video-exploded")
    save_dd_df(clean_user2video_ccid, relations_path / "clean-user-video-sequences.parquet")

    num_segments = count_segments(user2video_ccid)
    clean_num_segments = count_segments(clean_user2video_ccid)

    logging.info(f"Total number of segments before cleaning: {num_segments}")
    logging.info(f"Total number of segments after cleaning: {clean_num_segments}")
    logging.info(f"\n-----> Dataset reduction {(1-(clean_num_segments/num_segments)*100):.2f}")
    logging.info(f"Number of blacklisted users: {blacklist.shape[0].compute()}")
    logging.info(f"Number of total unique users: {user2video_dd['user_id'].unique().shape[0].compute()}")
