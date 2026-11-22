#!/usr/bin/env bash
# Monitor HugATree LiDAR bundle build (read-only; safe while build runs).
set -euo pipefail

BUNDLE_DIR="${1:-/home/catt/Dev/HugATree/static/data/lidar}"
INTERVAL="${INTERVAL:-30}"

pack="$BUNDLE_DIR/crops.pack.zst"
det="$BUNDLE_DIR/detections.bin"
idx="$BUNDLE_DIR/crops.idx"
meta="$BUNDLE_DIR/meta.json"

print_status() {
  local ts pack_b det_b idx_b running
  ts="$(date -Is)"
  pack_b=0
  det_b=0
  idx_b=0
  [[ -f "$pack" ]] && pack_b=$(wc -c <"$pack" | tr -d ' ')
  [[ -f "$det" ]] && det_b=$(wc -c <"$det" | tr -d ' ')
  [[ -f "$idx" ]] && idx_b=$(wc -c <"$idx" | tr -d ' ')
  running=0
  if pgrep -f "data-build-hugatree-bundle" >/dev/null 2>&1; then
    running=1
  fi

  local pack_mb det_count meta_note done_note
  pack_mb=$(awk "BEGIN{printf \"%.2f\", $pack_b/1048576}")
  det_count=""
  if [[ -f "$det" && $(wc -c <"$det" | tr -d ' ') -ge 12 ]]; then
    det_count=$(od -An -tu4 -j8 -N4 "$det" 2>/dev/null | tr -d ' ' || true)
  fi
  if [[ -f "$meta" ]] && command -v python3 >/dev/null; then
    meta_note=$(python3 -c "
import json
from pathlib import Path
p = Path('$meta')
try:
    m = json.loads(p.read_text())
except Exception:
    raise SystemExit
parts = []
if m.get('detection_count') is not None:
    parts.append('manifest_detections=' + str(m['detection_count']))
if m.get('total_pack_bytes') is not None:
    parts.append('manifest_pack_mb=' + format(m['total_pack_bytes'] / 1048576, '.1f'))
print(' '.join(parts))
" 2>/dev/null || true)
  fi
  if [[ "$running" -eq 1 ]]; then
    done_note="BUILD_RUNNING"
  elif [[ -f "$idx" ]]; then
    done_note="COMPLETE"
  else
    done_note="IDLE_OR_FAILED"
  fi

  local det_count_note=""
  if [[ -n "$det_count" ]]; then
    det_count_note="rows_in_detections_bin=$det_count"
  fi
  printf '%s  pack=%sMB  detections_bin=%sMB  idx=%s  status=%s  %s  %s\n' \
    "$ts" "$pack_mb" "$(awk "BEGIN{printf \"%.1f\", $det_b/1048576}")" \
    "$( [[ -f "$idx" ]] && echo yes || echo no)" "$done_note" "$det_count_note" "${meta_note:-}"

  if [[ "${LOG_PROGRESS:-}" == "1" ]]; then
    printf '{"ts":"%s","pack_bytes":%s,"detections_bytes":%s,"idx":%s,"running":%s}\n' \
      "$ts" "$pack_b" "$det_b" "$( [[ -f "$idx" ]] && echo true || echo false)" "$running" \
      >>"$BUNDLE_DIR/build.progress.jsonl"
  fi
}

usage() {
  echo "Usage: $0 [bundle_dir]"
  echo "  INTERVAL=30  loop seconds (set INTERVAL=0 for one shot)"
  echo "  LOG_PROGRESS=1  append JSON lines to bundle_dir/build.progress.jsonl"
  echo "  (No percent estimate — pack bytes/crop vary; watch pack MB growth.)"
  echo
  echo "Examples:"
  echo "  watch -n 30 $0"
  echo "  LOG_PROGRESS=1 INTERVAL=30 $0   # background-friendly logger"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$INTERVAL" -eq 0 ]]; then
  print_status
  exit 0
fi

while true; do
  print_status
  if [[ "$INTERVAL" -le 0 ]]; then
    break
  fi
  if ! pgrep -f "data-build-hugatree-bundle" >/dev/null 2>&1; then
    print_status
    break
  fi
  sleep "$INTERVAL"
done