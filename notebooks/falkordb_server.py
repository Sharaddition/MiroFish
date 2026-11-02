"""Keeps an embedded FalkorDB (FalkorDBLite) running on a TCP port until this process is stopped.

Colab and Kaggle cannot run Docker, so the notebook starts this as a background process instead of a FalkorDB
container. The server lives and dies with this process; its data is one file (``--path``) that the notebook copies to
Drive / /kaggle/working so a finished run's graph is not lost with the VM.

    python notebooks/falkordb_server.py --path /content/graph_db/mirofish.db --port 6390
"""
import argparse
import os
import signal
import sys
import time

from redislite.falkordb_client import FalkorDB


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/content/graph_db/mirofish.db")
    parser.add_argument("--port", default="6390")
    parser.add_argument("--save-every", default="30", help="seconds between snapshots of the data file while changes keep coming")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.path)), exist_ok=True)
    # Left alone, FalkorDBLite writes its file only on a clean shutdown, so a session that is cut off (Colab and Kaggle do
    # that) would lose the whole graph. "save N 1" snapshots it every N seconds as long as something changed.
    db = FalkorDB(args.path, serverconfig={"port": str(args.port), "save": f"{args.save_every} 1"})
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    print(f"FalkorDB (embedded) listening on 127.0.0.1:{args.port}, data file {args.path}", flush=True)
    while not stop:
        time.sleep(1)
    try:
        db.shutdown()   # flush the data file
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
