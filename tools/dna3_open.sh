#!/bin/sh
# Open-ended DNA-3 worlds (docs/17 section 14): no planned end. The run checkpoints every CKPT ticks; started again
# with the same arguments it resumes from NAME.ckpt and appends to its census, top-gene and slot-map files, so a
# world can be continued indefinitely (or stopped at any time and picked up later).
#
#   sh dna3_open.sh NAME BIN GPU CENSUS CKPT "SEED ARGS" WORLD ARGS...
#
# SEED ARGS (e.g. --n-seed 4096 --genome "...") are used only for a fresh start. WORLD ARGS must stay the same
# between restarts (the checkpoint checks N, G, seed and the days). --ticks is set to 10^12 (no end).
NAME=$1; BIN=$2; GPU=$3; CE=$4; CK=$5; SEEDARGS=$6; shift 6
if [ -f "$NAME.ckpt" ]; then START="--resume $NAME.ckpt"; else START="$SEEDARGS"; : > "$NAME.txt"; fi   # a fresh start truncates
echo "start $(date -u +%FT%TZ) $START" >> "$NAME.log"
eval "$BIN \"\$@\" --gpu $GPU --ticks 1000000000000 --census-every $CE --checkpoint $NAME.ckpt --checkpoint-every $CK \
      --top-genes $NAME.top --slot-map $NAME.map --slot-map-mode ${MODES:-plant,clade,traits} --out $NAME.json --no-hash \
      --depth-cap 50 $START" >> "$NAME.txt" 2>> "$NAME.err"
echo "exit $? $(date -u +%FT%TZ)" >> "$NAME.log"
