#!/usr/bin/env bash
# Parallel range download: pfetch.sh <url> <out> [connections=16]
# PyPI / GitHub are throttled per connection from this machine (~100 KB/s); N ranges give ~N times that.
set -euo pipefail
URL="$1"; OUT="$2"; N="${3:-16}"
[ -s "$OUT" ] && { echo "exists $OUT"; exit 0; }
SIZE=$(curl -sIL "$URL" | tr -d '\r' | awk 'tolower($1)=="content-length:"{s=$2} END{print s}')
[ -n "$SIZE" ] || { echo "no content-length for $URL" >&2; exit 1; }
TMP="$OUT.parts.$$"; mkdir -p "$TMP"
CH=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  S=$((i*CH)); E=$((S+CH-1)); [ "$E" -ge "$SIZE" ] && E=$((SIZE-1))
  [ "$S" -ge "$SIZE" ] && break
  curl -sL --retry 5 --retry-delay 2 -r "$S-$E" -o "$TMP/$(printf %03d "$i")" "$URL" &
done
wait
cat "$TMP"/* > "$OUT.tmp.$$"
[ "$(stat -c %s "$OUT.tmp.$$")" = "$SIZE" ] || { echo "size mismatch for $OUT" >&2; rm -rf "$TMP" "$OUT.tmp.$$"; exit 1; }
mv "$OUT.tmp.$$" "$OUT"; rm -rf "$TMP"
echo "fetched $OUT ($SIZE bytes)"
