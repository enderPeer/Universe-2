#!/bin/bash
# reference vs dna_world_v2 on seeded small worlds; every tick hashed; census lines and births files compared
cd ~/universe2/dna/seeded/v
E=../dna_world_v2; R="python3 ../dna_ref.py"
G1="ATG TTC GGA GCA AAA ATA CTC"; G2="ATG GCA TTC GGA AAA ATA CTC"
run() { name=$1; shift
  $R "$@" --hash-every 1 --census-every 10 --births $name.ref.births > $name.ref.txt 2> $name.ref.err
  $E "$@" --hash-every 1 --census-every 10 --births $name.eng.births --log-births $name.eng.log --log-filter all --top-genes $name.eng.top --slot-map $name.eng.map > $name.eng.txt 2> $name.eng.err
  $E "$@" --hash-every 1 --census-every 10 --depth-cap 3 --log-births $name.eng2.log > $name.eng2.txt 2> $name.eng2.err
  h1=$(grep -c '^tick' $name.ref.txt); c1=$(grep -c '^census' $name.ref.txt)
  if cmp -s $name.ref.txt $name.eng.txt && cmp -s $name.ref.births $name.eng.births && cmp -s <(grep '^tick' $name.ref.txt) <(grep '^tick' $name.eng2.txt); then ok=OK; else ok=MISMATCH; fi
  echo "$name $ok hashes=$h1 census=$c1 $(tail -1 $name.ref.err | sed 's/secs=.*//')"
}
run v1 --N 65536 --G 256 --ticks 300 --seed 1 --blank --n-seed 128 --genome "$G1" --mu 0
run v2 --N 65536 --G 256 --ticks 300 --seed 2 --blank --n-seed 128 --genome "$G1" --mu 0.01
run v3 --N 65536 --G 256 --ticks 300 --seed 3 --n-seed 32 --genome "$G1" --mu 0.001
run v4 --N 262144 --G 1024 --ticks 200 --seed 4 --n-seed 64 --genome "$G1" --mu 0
run v5 --N 65536 --G 256 --ticks 300 --seed 5 --blank --n-seed 128 --genome "$G1" --mu 0 --no-double --D 10
run v6 --N 16384 --G 256 --ticks 200 --seed 1
run v7 --N 65536 --G 256 --ticks 300 --seed 7 --blank --n-seed 128 --genome "$G2" --mu 0.001
run v8 --N 65536 --G 256 --ticks 400 --seed 8 --blank --n-seed 64 --genome "$G1" --mu 0.001 --F 4
echo verify-done
