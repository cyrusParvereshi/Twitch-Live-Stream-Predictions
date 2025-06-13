import pandas as pd
import glob
import os


def find_snapshot_files(snapshot_dir, pattern="all-*.csv"):
    """Find all snapshot CSV files matching the pattern in the specified directory."""
    search_path = os.path.join(snapshot_dir, pattern)
    return glob.glob(search_path)


def load_snapshots(csv_files):
    """Load all CSVs into pandas DataFrames."""
    dataframes = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, dtype={"id": str, "user_id": str})
            dataframes.append(df)
        except Exception as e:
            print(f"Could not read {f}: {e}")
    return dataframes


def merge_snapshots(dfs):
    """Merge, sort, and deduplicate all snapshots by stream 'id', keeping the latest."""
    if not dfs:
        print("No valid dataframes loaded.")
        return None
    big_df = pd.concat(dfs, ignore_index=True)
    # Ensure snapshot_time is parsed as datetime for sorting
    big_df["snapshot_time"] = pd.to_datetime(big_df["snapshot_time"], errors="coerce")
    # Sort by stream id and snapshot_time, so the latest occurrence comes last
    big_df.sort_values(["id", "snapshot_time"], inplace=True)
    # Drop duplicates, keeping only the latest snapshot for each stream id
    unique_df = big_df.drop_duplicates("id", keep="last")
    return unique_df


def save_merged_snapshot(df, output_path):
    """Save the merged and deduplicated DataFrame to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Merged CSV saved to {output_path}")


def create_raw_csv():
    # Set this to the 'data' directory relative to this script
    SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    OUTPUT_FILE = "twitch_streams_latest.csv"
    OUTPUT_PATH = os.path.join(SNAPSHOT_DIR, OUTPUT_FILE)
    csv_files = find_snapshot_files(SNAPSHOT_DIR, "all-*.csv")
    if not csv_files:
        print(
            "No snapshot CSVs found in the data folder. Please check your SNAPSHOT_DIR and file naming pattern."
        )
        return

    dfs = load_snapshots(csv_files)
    merged_df = merge_snapshots(dfs)
    if merged_df is not None:
        save_merged_snapshot(merged_df, OUTPUT_PATH)
        print(
            f"Merged {len(csv_files)} snapshots, resulting in {len(merged_df)} unique streams."
        )


if __name__ == "__main__":
    create_raw_csv()
