#!/usr/bin/env bash
# pmp.sh <out> <interval_s> <prog> [args...]
set -u
out="$1"; interval="$2"; shift 2
S=$(dirname "$0")  # pmp.py lives next to this script
PMP_OUT="$out" PMP_PIDFILE="$out.started" gdb -q -batch -x "$S/pmp.py" --args "$@" > "$out.gdb.log" 2>&1 &
gpid=$!
# find the target (child of gdb)
for _ in $(seq 100); do
  tpid=$(pgrep -P "$gpid" | head -1); [ -n "$tpid" ] && break; sleep 0.1
done
n=0; echo "target=$tpid" >&2
while kill -0 "$tpid" 2>/dev/null; do
  sleep "$interval"; kill -USR2 "$tpid" 2>/dev/null && n=$((n+1))
done
echo "sent=$n" >&2; wait "$gpid"
