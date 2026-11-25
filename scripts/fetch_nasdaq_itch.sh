#!/usr/bin/env bash
# Downloads NASDAQ's free full-day TotalView-ITCH 5.0 sample (30 Dec 2019,
# 3,524,013,057 bytes) into data/, which git ignores. NASDAQ publishes these
# sample files for developers at https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/
#
# The file is never decompressed to disk: the replay tool reads it from gzip,
# and gzip's CRC-32 check over the whole stream verifies the download.
#
#   scripts/fetch_nasdaq_itch.sh
#   gzip -dc data/12302019.NASDAQ_ITCH50.gz | build/bin/itch_replay -
set -euo pipefail

cd "$(dirname "$0")/.."
FILE=12302019.NASDAQ_ITCH50.gz
URL="https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/${FILE}"
SIZE=3524013057

mkdir -p data
curl -fL --retry 5 --retry-delay 5 -C - -o "data/${FILE}" "${URL}"
actual=$(wc -c < "data/${FILE}" | tr -d ' ')
if [ "${actual}" != "${SIZE}" ]; then
    echo "size mismatch: got ${actual} bytes, expected ${SIZE}" >&2
    exit 1
fi
echo "data/${FILE}: ${actual} bytes (size verified; gzip verifies the CRC when the file is replayed)"
