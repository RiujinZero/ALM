#!/usr/bin/env python
"""Copy a byte range of one source file to local disk, surviving mount drops.

Google Drive's streaming mount cannot sustain multi-GB sequential reads:
the connection drops with OSError(EIO) and even remounts fail mid-stream.
Training also re-reads corpora every epoch, so live streaming is not
viable downstream either. This copier lands exactly the byte ranges the
corpus builder needs onto the local filesystem once, with:

- resume: progress is the size of the .part file; rerunning continues
  from there (input is re-seeked, so Drive re-fetches the range);
- self-healing: on I/O error, remount (configurable command), reopen,
  seek back to the last good offset, keep going;
- generous retry budget (long backoff) because Drive sometimes needs
  minutes, not seconds, to come back.

The final file is renamed from .part to its real name only when complete,
so a partial copy is never mistaken for a full one.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_CHUNK_BYTES = 4 * 1024 * 1024
_MAX_IO_RETRIES = 24
_DEFAULT_REMOUNT_CMD = "sudo -n mount -t drvfs G: /mnt/g"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bytes", type=int, default=0,
                        help="copy at most this many bytes; 0 means the whole file")
    parser.add_argument("--remount-cmd", type=str, default=_DEFAULT_REMOUNT_CMD)
    args = parser.parse_args()

    total = args.bytes if args.bytes > 0 else args.source.stat().st_size
    part = args.output.with_suffix(args.output.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    if done >= total:
        print(f"already complete: {part} has {done} >= {total} bytes")
        part.rename(args.output)
        return
    print(f"copying {total:,} bytes -> {part} (resuming at {done:,})", flush=True)
    retries = 0
    started = time.perf_counter()
    with part.open("ab") as out:
        while done < total:
            try:
                with args.source.open("rb") as src:
                    src.seek(done)
                    remaining = total - done
                    for _ in range((remaining + _CHUNK_BYTES - 1) // _CHUNK_BYTES):
                        chunk = src.read(min(_CHUNK_BYTES, total - done))
                        if not chunk:
                            break
                        out.write(chunk)
                        out.flush()
                        done += len(chunk)
                        if done % (100 * _CHUNK_BYTES) < _CHUNK_BYTES:
                            rate = done / max(time.perf_counter() - started, 1e-9)
                            print(
                                f"  {done / 1e9:.2f}/{total / 1e9:.2f} GB "
                                f"({rate / 1e6:.1f} MB/s)",
                                flush=True,
                            )
                retries = 0
            except OSError as error:
                retries += 1
                if retries > _MAX_IO_RETRIES:
                    raise
                wait = min(2 ** min(retries, 6), 60)
                print(
                    f"  I/O error ({error}); retry {retries}/{_MAX_IO_RETRIES} "
                    f"after remount and {wait}s",
                    flush=True,
                )
                time.sleep(wait)
                if args.remount_cmd:
                    subprocess.run(
                        args.remount_cmd, shell=True, check=False, capture_output=True
                    )
    part.rename(args.output)
    print(f"done: {args.output} ({done:,} bytes in "
          f"{time.perf_counter() - started:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
