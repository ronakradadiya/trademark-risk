#!/bin/bash
# Parallel-download TRTYRAP zips via the USPTO ODP API.
# Reads urls.tsv (url<TAB>filename) and writes to zips/.
set -u
API_KEY="${USPTO_API_KEY:-cyejlqysvdzjlmclxilfybmrocyeot}"
CONCURRENCY="${CONCURRENCY:-2}"
RETRY_429_SLEEP="${RETRY_429_SLEEP:-5}"
mkdir -p zips
: > download.log

download_one() {
    local url="$1" name="$2" dest="zips/$2"
    if [ -s "$dest" ]; then
        echo "SKIP $name" >> download.log
        return 0
    fi
    local code attempt=0
    while [ $attempt -lt 5 ]; do
        code=$(curl -sSL -H "x-api-key: $API_KEY" -o "$dest" -w '%{http_code}' "$url")
        if [ "$code" = "200" ]; then
            echo "OK   $name $(stat -f%z "$dest")" >> download.log
            return 0
        fi
        rm -f "$dest"
        if [ "$code" = "429" ]; then
            sleep $((RETRY_429_SLEEP * (attempt + 1)))
            attempt=$((attempt+1))
            continue
        fi
        echo "FAIL $name http=$code" >> download.log
        return 1
    done
    echo "FAIL $name http=429 (exhausted retries)" >> download.log
    return 1
}

active=0
while IFS=$'\t' read -r url name; do
    [ -z "$url" ] && continue
    download_one "$url" "$name" &
    active=$((active+1))
    if [ "$active" -ge "$CONCURRENCY" ]; then
        wait -n
        active=$((active-1))
    fi
done < urls.tsv
wait

ok=$(grep -c ^OK download.log || true)
skipped=$(grep -c ^SKIP download.log || true)
failed=$(grep -c ^FAIL download.log || true)
echo "---" >> download.log
echo "done: $ok ok, $skipped skipped, $failed failed" >> download.log
echo "done: $ok ok, $skipped skipped, $failed failed"
