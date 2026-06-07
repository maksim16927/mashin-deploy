#!/usr/bin/env python3
"""
Google Drive Resumable Upload with Detailed Logging
- Uses Drive API resumable upload (chunked)
- Logs progress (percent, MB/s, ETA) to console and file
- Resumes automatically on transient failures

Prereqs:
  pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib tqdm
  Create OAuth client credentials (Desktop) and save as credentials.json next to this script.
Usage:
  python tools/drive_resumable_upload.py --file /path/big_archive.zip --folder-id <DriveFolderID> --chunk-size 128 --name big_archive.zip
"""

import argparse
import math
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

LOG_PATH = Path(__file__).with_name("drive_upload.log")
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def setup_logger(verbose: bool = True):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
        ],
    )
    logging.info("Logging to %s", LOG_PATH)


def get_creds(credentials_path: Path, token_path: Path) -> Credentials:
    creds: Optional[Credentials] = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(creds.to_json())
            logging.info("Saved token to %s", token_path)
    return creds


def format_size(n_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while n_bytes >= 1024 and i < len(units) - 1:
        n_bytes /= 1024.0
        i += 1
    return f"{n_bytes:.2f} {units[i]}"


def upload_file(service, file_path: Path, folder_id: Optional[str], name: Optional[str], chunk_mb: int) -> str:
    file_size = file_path.stat().st_size
    media = MediaFileUpload(str(file_path), chunksize=chunk_mb * 1024 * 1024, resumable=True)

    body = {"name": name or file_path.name}
    if folder_id:
        body["parents"] = [folder_id]

    request = service.files().create(body=body, media_body=media, fields="id, name, size")

    start_time = time.time()
    last_log_time = start_time
    last_bytes = 0

    logging.info("Starting upload: %s (%s)", file_path.name, format_size(file_size))
    logging.info("Chunk size: %d MB", chunk_mb)

    response = None
    while response is None:
        try:
            status, response = request.next_chunk()
            now = time.time()
            if status:
                uploaded = int(status.resumable_progress)
                elapsed = now - start_time
                since_last = now - last_log_time
                speed = (uploaded - last_bytes) / max(since_last, 1e-6)
                overall_speed = uploaded / max(elapsed, 1e-6)
                remaining = max(file_size - uploaded, 0)
                eta = remaining / max(overall_speed, 1e-9)
                pct = (uploaded / file_size) * 100 if file_size > 0 else 0

                if since_last >= 1.0 or uploaded == file_size:
                    logging.info(
                        "Progress: %6.2f%% | %s / %s | cur %.2f MB/s | avg %.2f MB/s | ETA %s",
                        pct,
                        format_size(uploaded),
                        format_size(file_size),
                        speed / (1024 * 1024),
                        overall_speed / (1024 * 1024),
                        time.strftime("%H:%M:%S", time.gmtime(eta)),
                    )
                    last_log_time = now
                    last_bytes = uploaded
        except Exception as e:
            logging.warning("Transient error: %s. Retrying in 5s...", e)
            time.sleep(5)

    file_id = response.get("id")
    logging.info("Upload finished: %s (id=%s)", response.get("name"), file_id)
    return file_id


def main():
    parser = argparse.ArgumentParser(description="Google Drive Resumable Uploader with Logs")
    parser.add_argument("--file", required=True, help="Path to local file to upload")
    parser.add_argument("--folder-id", default=None, help="Destination Drive folder ID")
    parser.add_argument("--name", default=None, help="Name of the file on Drive")
    parser.add_argument("--chunk-size", type=int, default=128, help="Chunk size in MB (default: 128)")
    parser.add_argument("--credentials", default="credentials.json", help="Path to OAuth credentials.json")
    parser.add_argument("--token", default="token.json", help="Path to store OAuth token")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logger(verbose=args.verbose)

    file_path = Path(args.file).expanduser().resolve()
    if not file_path.exists():
        logging.error("File not found: %s", file_path)
        sys.exit(1)

    credentials_path = Path(args.credentials).expanduser().resolve()
    token_path = Path(args.token).expanduser().resolve()
    if not credentials_path.exists():
        logging.error("Missing OAuth credentials: %s", credentials_path)
        logging.error("Create Desktop OAuth client at https://console.cloud.google.com/apis/credentials and download credentials.json")
        sys.exit(2)

    creds = get_creds(credentials_path, token_path)
    service = build("drive", "v3", credentials=creds, cache_discovery=False)

    file_id = upload_file(service, file_path, args.folder_id, args.name, args.chunk_size)

    print("\n=== SUMMARY ===")
    print(f"Local file: {file_path}")
    print(f"Drive file ID: {file_id}")
    print(f"Logs saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()
