import requests
import time
import datetime
import os
import inspect
import configparser
import csv
from tqdm import tqdm


def setup():
    """Reads credentials (client_id and client_secret) from creds.ini."""
    config = configparser.ConfigParser()
    config.read("creds.ini")
    client_id = config["AppCredentials"]["client_id"]
    client_secret = config["AppCredentials"]["client_secret"]
    return client_id, client_secret


def get_oauth_token(client_id, client_secret):
    """Requests an OAuth token using the client credentials flow."""
    url = "https://id.twitch.tv/oauth2/token"
    params = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
    }
    resp = requests.post(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    return data["access_token"]


def save_streams_to_csv(platform, client_id, oauth_token):
    """
    Fetches all live streams from Twitch and saves them as a CSV snapshot,
    including a snapshot_time column to track collection time.
    """
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d-%H-%M-%S")
    snapshot_time = datetime.datetime.utcnow().isoformat()
    # Data directory
    DATA_DIR = os.path.join(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))),
        "data",
    )
    os.makedirs(DATA_DIR, exist_ok=True)

    csv_name = f"{platform}-{date}.csv"
    csv_path = os.path.join(DATA_DIR, csv_name)

    columns = [
        "id",
        "user_id",
        "user_login",
        "user_name",
        "game_id",
        "game_name",
        "type",
        "title",
        "viewer_count",
        "started_at",
        "language",
        "thumbnail_url",
        "is_mature",
        "snapshot_time",
    ]

    headers = {"Client-ID": client_id, "Authorization": f"Bearer {oauth_token}"}
    url = "https://api.twitch.tv/helix/streams"
    params = {"first": 100}  # Max allowed by Twitch API

    total_streams = 0
    page = 0
    cursor = None

    with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()

        print(f" Getting Twitch livestream snapshot ({snapshot_time})...")

        with tqdm(unit="page ") as pbar:  # set progress bar
            while True:
                if cursor:
                    params["after"] = cursor
                else:
                    params.pop("after", None)
                resp = requests.get(url, headers=headers, params=params)
                if resp.status_code != 200:
                    print(f"Error: HTTP {resp.status_code} - {resp.text}")
                    break
                data = resp.json()
                streams = data.get("data", [])
                for stream in streams:
                    row = {
                        col: stream.get(col, "")
                        for col in columns
                        if col != "snapshot_time"
                    }
                    row["snapshot_time"] = snapshot_time
                    writer.writerow(row)
                total_streams += len(streams)
                page += 1
                pbar.update(1)
                pbar.set_postfix({"streams": total_streams})

                # Pagination: continue if there's a next page, else break
                cursor = data.get("pagination", {}).get("cursor")
                if not cursor or not streams:
                    break

    print(f"Finished. Fetched {total_streams} streams across {page} pages.")
    print(f"Saved snapshot to {csv_path}")


if __name__ == "__main__":
    client_id, client_secret = setup()
    oauth_token = get_oauth_token(client_id, client_secret)
    save_streams_to_csv("all", client_id, oauth_token)
