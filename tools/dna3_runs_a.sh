#!/bin/sh
# DNA-3 first worlds (docs/17 section 11), series A: 104,857,600 cells = 640 x 640 slots of 256 cells, seed 1,
# mu 0.001, double strand, damage and inflow at their defaults, census every 100 (long runs 200), slot maps.
# Run on one GPU from ~/universe2/dna3/runs:  sh dna3_runs_a.sh GPU [names...]
GPU=$1; shift
B=../dna_world3
REPL="ATG GCT CTT TTA GGT"
PLANT="ATG AGA GCT CTT TTA GGT"
C="--N 104857600 --G 256 --seed 1 --mu 0.001 --no-hash --depth-cap 50 --gpu $GPU"
run() {   # name, ticks, census_every, map modes, args...
  n=$1; t=$2; ce=$3; mm=$4; shift 4
  echo "start $n $(date +%T)"
  $B $C --ticks $t --census-every $ce --top-genes $n.top --slot-map $n.map --slot-map-mode $mm --out $n.json "$@" > $n.txt 2> $n.err
  gzip -f $n.txt $n.top
  echo "end $n $(date +%T) $(tail -1 $n.err | cut -c1-120)"
}
for name in "$@"; do
  case $name in
  A0) run A0_dna2 20000 100 clade --dna2 --n-seed 4096 --genome "$REPL" ;;
  A1) run A1_day1 20000 100 clade --days 1 --n-seed 4096 --genome "$REPL" ;;
  A2) run A2_day2 20000 100 clade,gene --days 2 --W 640 --n-seed 4096 --genome "$REPL" ;;
  A3) run A3_day12 20000 100 clade,energy --days 1,2 --W 640 --n-seed 4096 --genome "$REPL" ;;
  A4) run A4_plant 20000 100 clade,plant,energy --dna3 --W 640 --land-cell 64 --n-seed 4096 --seed-terrain sea --genome "$PLANT" ;;
  A5) run A5_nonplant 100000 200 clade,plant --dna3 --W 640 --land-cell 64 --n-seed 4096 --seed-terrain sea --genome "$REPL" ;;
  A6) run A6_soup 200000 200 clade,plant --dna3 --W 640 --land-cell 64 ;;
  A7) run A7_one_seed 20000 100 clade --days 1,2 --W 640 --n-seed 1 --seed-slots 6720 --genome "$REPL" ;;
  esac
done
echo "all done $(date +%T)"
