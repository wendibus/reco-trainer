#!/bin/sh
set -eu

exec python3 local_worker.py --port 8766
