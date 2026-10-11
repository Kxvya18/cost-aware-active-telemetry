from __future__ import annotations

import argparse
import concurrent.futures
import os
import urllib.request
from pathlib import Path


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def fetch_part(url: str, path: Path, start: int, end: int) -> tuple[int, int]:
    expected = end - start + 1
    present = path.stat().st_size if path.exists() else 0
    if present == expected:
        return start, expected
    if present > expected:
        path.unlink()
        present = 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={start + present}-{end}"})
    mode = "ab" if present else "wb"
    with urllib.request.urlopen(request, timeout=180) as response, path.open(mode) as output:
        if response.status != 206:
            raise RuntimeError(f"Server ignored range for {path.name} with status {response.status}")
        while True:
            block = response.read(4 * 1024 * 1024)
            if not block:
                break
            output.write(block)
    actual = path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"Range {start}-{end} has {actual} bytes, expected {expected}")
    return start, actual


def download(url: str, output: Path, workers: int) -> None:
    size = remote_size(url)
    part_dir = output.with_suffix(output.suffix + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    chunk = (size + workers - 1) // workers
    ranges = []
    for index in range(workers):
        start = index * chunk
        end = min(size - 1, (index + 1) * chunk - 1)
        if start <= end:
            ranges.append((part_dir / f"part-{index:03d}", start, end))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_part, url, path, start, end) for path, start, end in ranges]
        for future in concurrent.futures.as_completed(futures):
            start, count = future.result()
            print(f"completed range at {start:,}, {count:,} bytes", flush=True)
    temporary = output.with_suffix(output.suffix + ".assembling")
    with temporary.open("wb") as target:
        for path, _, _ in ranges:
            with path.open("rb") as source:
                while True:
                    block = source.read(8 * 1024 * 1024)
                    if not block:
                        break
                    target.write(block)
    if temporary.stat().st_size != size:
        raise RuntimeError("Assembled byte count does not match server Content-Length")
    os.replace(temporary, output)
    print(f"verified byte count for {output}: {size:,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    download(args.url, args.output, args.workers)


if __name__ == "__main__":
    main()
