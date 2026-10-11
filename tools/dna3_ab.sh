#!/bin/sh
# The A/B/C experiment of docs/17 section 19.3: does promoting evolved solutions to instructions (macros) break the
# plateau? Run on a cluster node from a directory for the runs (e.g. ~/universe2/dna3/ab), with the repository
# checked out (REPO, default: the repository this script lives in) and a built engine binary (BIN).
#
#   sh dna3_ab.sh epoch1 BIN GPU SEED           fresh ladder world E1_s<SEED>, 2 x 10^6 ticks, checkpoint, op census
#   sh dna3_ab.sh harvest SEED                  extractor on E1_s<SEED>.ckpt -> solutions, macro table (A) and its
#                                               empty-body twin (C); prints the table
#   sh dna3_ab.sh epoch2 BIN GPU SEED A|B|C     fork of E1_s<SEED> at its last checkpoint: A with the macros, B as
#                                               it is, C with the donor codons taken but empty bodies; 2 x 10^6 ticks
#   sh dna3_ab.sh report SEED                   harvest of the three arms at their end and the activity measure
#
# Worlds: O7's size (10^8 cells, 640 x 640 slots of 256 cells, land-cell 64), days 1 to 6, --ladder, --in-fresh,
# --shade --T 0, mu 0.001, 4,096 plant seeds in the sea; census every 10,240 ticks, checkpoint every 102,400.
# Nothing here touches D7 or D8. Every run writes NAME.txt (census), NAME.err, NAME.top, NAME.ops, NAME.ckpt.
set -e
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${PY:-python3}
PLANT="ATG AGA GCT CTT TTA GGT"
E1_TICKS=${E1_TICKS:-2000000}
E2_TICKS=${E2_TICKS:-2000000}
WORLD="--days 1,2,3,4,5,6 --ladder --in-fresh --shade --T 0 --N 104857600 --G 256 --W 640 --land-cell 64 --mu 0.001"
OUT="--census-every 10240 --checkpoint-every 102400 --no-hash --depth-cap 50"

run() {   # NAME BIN GPU args...
  n=$1; b=$2; g=$3; shift 3
  echo "start $n $(date -u +%FT%TZ)" >> "$n.log"
  "$b" "$@" --gpu "$g" $OUT --checkpoint "$n.ckpt" --top-genes "$n.top" --op-census "$n.ops" --out "$n.json" > "$n.txt" 2> "$n.err"
  echo "exit $? $(date -u +%FT%TZ) $(tail -1 "$n.err" | cut -c1-160)" >> "$n.log"
}

cmd=$1; shift
case $cmd in
epoch1)
  BIN=$1; GPU=$2; SEED=$3
  run "E1_s$SEED" "$BIN" "$GPU" $WORLD --seed "$SEED" --ticks "$E1_TICKS" --n-seed 4096 --seed-terrain sea --genome "$PLANT" ;;
harvest)
  SEED=$1; N="E1_s$SEED"
  $PY "$REPO/tools/dna3_extract_solutions.py" --mode ladder --ckpt "$N.ckpt" --verify --json "${N}_solutions.json" \
      --promote "${N}_macros.json" --max-macros 8 | tee "${N}_harvest.txt"
  # arm C: the same donor codons with empty bodies (a NOP that costs its unit)
  sed -E 's/^(M[0-9]+ [A-Z]+) .*$/\1/' "${N}_macros.txt" > "${N}_macros_C.txt"
  echo "--- ${N}_macros.txt"; cat "${N}_macros.txt"; echo "--- ${N}_macros_C.txt"; cat "${N}_macros_C.txt" ;;
epoch2)
  BIN=$1; GPU=$2; SEED=$3; ARM=$4; E1="E1_s$SEED"; N="E2_s${SEED}_$ARM"
  T=$((E1_TICKS + E2_TICKS))
  cp "$E1.ckpt" "$N.ckpt"   # each arm continues its own copy of the epoch-1 checkpoint
  case $ARM in
  A) run "$N" "$BIN" "$GPU" $WORLD --seed "$SEED" --ticks "$T" --resume "$N.ckpt" --fork --macros "${E1}_macros.txt" ;;
  B) run "$N" "$BIN" "$GPU" $WORLD --seed "$SEED" --ticks "$T" --resume "$N.ckpt" ;;
  C) run "$N" "$BIN" "$GPU" $WORLD --seed "$SEED" --ticks "$T" --resume "$N.ckpt" --fork --macros "${E1}_macros_C.txt" ;;
  *) echo "arm A, B or C"; exit 1 ;;
  esac ;;
report)
  SEED=$1; E1="E1_s$SEED"
  for ARM in A B C; do
    N="E2_s${SEED}_$ARM"; M=""
    [ "$ARM" = A ] && M="--macros ${E1}_macros.txt"
    [ "$ARM" = C ] && M="--macros ${E1}_macros_C.txt"
    $PY "$REPO/tools/dna3_extract_solutions.py" --mode ladder --ckpt "$N.ckpt" $M --verify --json "${N}_solutions.json" > "${N}_harvest.txt" || true
    $PY "$REPO/tools/dna3_activity.py" "$N.ops" --seed-genome "$PLANT" --out "${N}_activity" > "${N}_activity.txt" 2>&1 || true
    echo "== $N: $(head -2 "${N}_harvest.txt" | tail -1)"
  done ;;
*) echo "usage: sh dna3_ab.sh epoch1 BIN GPU SEED | harvest SEED | epoch2 BIN GPU SEED A|B|C | report SEED"; exit 1 ;;
esac
