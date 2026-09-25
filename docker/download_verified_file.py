#!/usr/bin/env python3
"""Download a large file concurrently with HTTP ranges and verify SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen


def _fetch_range(
    url: str,
    destination: Path,
    start: int,
    end: int,
    timeout: int,
    retries: int,
) -> None:
    expected_size = end - start + 1
    for attempt in range(retries):
        try:
            request = Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urlopen(request, timeout=timeout) as response:
                if response.status != 206:
                    raise RuntimeError(f"range request returned HTTP {response.status}")
                payload = response.read()
            if len(payload) != expected_size:
                raise RuntimeError(
                    f"range {start}-{end} returned {len(payload)} bytes, "
                    f"expected {expected_size}"
                )
            with destination.open("r+b") as output:
                output.seek(start)
                output.write(payload)
            return
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(attempt + 1)


def download_verified_file(
    url: str,
    destination: Path,
    expected_sha256: str,
    workers: int = 16,
    timeout: int = 120,
    retries: int = 3,
) -> None:
    with urlopen(Request(url, method="HEAD"), timeout=timeout) as response:
        total_size = int(response.headers["Content-Length"])
        if response.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise RuntimeError("download server does not advertise byte ranges")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        output.truncate(total_size)

    chunk_size = math.ceil(total_size / workers)
    ranges = [
        (start, min(start + chunk_size - 1, total_size - 1))
        for start in range(0, total_size, chunk_size)
    ]

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _fetch_range,
                    url,
                    destination,
                    start,
                    end,
                    timeout,
                    retries,
                )
                for start, end in ranges
            ]
            for future in futures:
                future.result()

        digest = hashlib.sha256()
        with destination.open("rb") as downloaded:
            for block in iter(lambda: downloaded.read(1024 * 1024), b""):
                digest.update(block)
        actual_sha256 = digest.hexdigest()
        if actual_sha256.lower() != expected_sha256.lower():
            raise RuntimeError(
                f"SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
            )
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("destination", type=Path)
    parser.add_argument("sha256")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    print(f"Downloading {args.url} with {args.workers} workers", flush=True)
    download_verified_file(
        args.url,
        args.destination,
        args.sha256,
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
    )
    print(f"Verified {args.destination}", flush=True)


if __name__ == "__main__":
    main()
