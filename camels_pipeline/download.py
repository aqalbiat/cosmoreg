from __future__ import annotations

import os
import time
from pathlib import Path

import requests

from .provenance import file_identity


HTML_PREFIXES = (b"<!doctype html", b"<html", b"\n<html")


def _is_probably_html(path: Path) -> bool:
    with path.open("rb") as fh:
        prefix = fh.read(512).lstrip().lower()
    return prefix.startswith(HTML_PREFIXES)


def _format_bytes(n_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    value = float(n_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{n_bytes} B"


def download_file(
    url: str,
    destination: Path,
    overwrite: bool = False,
    retries: int = 3,
    timeout: int = 30,
    progress_interval: int = 5,
) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        if _is_probably_html(destination):
            raise ValueError(f"{destination} looks like HTML, not data. Re-run with overwrite after checking the URL.")
        return file_identity(destination, url)
    tmp = destination.with_suffix(destination.suffix + ".download")
    if tmp.exists():
        tmp.unlink()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"Attempt {attempt}/{retries}: {url}", flush=True)
            with requests.get(
                url,
                stream=True,
                timeout=(10, timeout),
                headers={"User-Agent": "camels-pipeline/1.0"},
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                if "text/html" in content_type.lower():
                    raise ValueError(f"{url} returned HTML content-type {content_type}.")
                total = int(response.headers.get("Content-Length") or 0)
                if total:
                    print(f"Expected size: {_format_bytes(total)}", flush=True)
                downloaded = 0
                last_report = time.monotonic()
                with tmp.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 1024 * 4):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= progress_interval:
                            if total:
                                pct = 100.0 * downloaded / total
                                print(
                                    f"Downloaded {_format_bytes(downloaded)} / {_format_bytes(total)} ({pct:.1f}%)",
                                    flush=True,
                                )
                            else:
                                print(f"Downloaded {_format_bytes(downloaded)}", flush=True)
                            last_report = now
                print(f"Finished transfer: {_format_bytes(downloaded)}", flush=True)
            if _is_probably_html(tmp):
                raise ValueError(f"{url} downloaded an HTML page instead of a dataset.")
            os.replace(tmp, destination)
            return file_identity(destination, url)
        except (requests.RequestException, TimeoutError, ValueError, OSError) as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink()
            print(f"Download attempt failed: {exc}", flush=True)
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {url}: {last_error}") from last_error
