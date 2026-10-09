#!/bin/bash
# (a) v3 hash check against the reference outputs of the verification suite, (b) --log-codons reruns of four pilots,
# (c) the 40-minute 3e9 run: G 256, blank, 1,000,000 seeds of ATG TTC GGA GCA AAA ATA CTC, mu 0.001, double strand,
#     D and F auto, T 64, census every 100, every DIVIDE birth logged with codons, top genes, --max-seconds 2400.
cd ~/universe2/dna/seeded
export CUDA_VISIBLE_DEVICES=0
E=$HOME/universe2/dna/seeded/dna_world_v3
G1="ATG TTC GGA GCA AAA ATA CTC"
cd v
$E --N 65536 --G 256 --ticks 300 --seed 1 --blank --n-seed 128 --genome "$G1" --mu 0 --hash-every 1 --census-every 10 --births v1.v3.births --log-births v1.v3.log --log-codons --depth-cap 3 > v1.v3.txt 2> v1.v3.err
$E --N 65536 --G 256 --ticks 300 --seed 2 --blank --n-seed 128 --genome "$G1" --mu 0.01 --hash-every 1 --census-every 10 --births v2.v3.births --log-births v2.v3.log --log-codons --max-seconds 1000 > v2.v3.txt 2> v2.v3.err
$E --N 65536 --G 256 --ticks 400 --seed 8 --blank --n-seed 64 --genome "$G1" --mu 0.001 --F 4 --hash-every 1 --census-every 10 --births v8.v3.births --log-births v8.v3.log --log-codons > v8.v3.txt 2> v8.v3.err
for n in v1 v2 v8; do
  if cmp -s <(grep '^tick' $n.ref.txt) <(grep '^tick' $n.v3.txt) && cmp -s $n.ref.births $n.v3.births; then echo "$n v3 OK $(grep -c ^tick $n.v3.txt) hashes"; else echo "$n v3 MISMATCH"; fi
done
cmp -s v2.ref.txt v2.v3.txt && echo "v2 v3 census identical" || echo "v2 v3 census differs"
cd ../runs
run() { name=$1; shift
  $E --N 99999744 --ticks 20000 --seed 1 --no-hash --census-every 10 --depth-cap 10 --log-births $name.log --log-codons --top-genes $name.top --out $name.json "$@" > $name.txt 2> $name.err
  echo "$name $(grep -o 'ticks_per_second=[0-9.]*' $name.err) $(grep -o 'live=.*' $name.err)"
}
run c256_35511          --G 256 --blank --n-seed 35511 --genome "$G1" --mu 0
run c256_35511_mu001    --G 256 --blank --n-seed 35511 --genome "$G1" --mu 0.001
run c256_35511_mu01     --G 256 --blank --n-seed 35511 --genome "$G1" --mu 0.01
run c256_35511_nodouble --G 256 --blank --n-seed 35511 --genome "$G1" --mu 0 --no-double --D 950
date +%s > big.start
$E --N 2999999488 --G 256 --ticks 900000 --seed 1 --no-hash --census-every 100 --depth-cap 10 --blank --n-seed 1000000 --genome "$G1" --mu 0.001 --log-births big.log --log-codons --top-genes big.top --out big.json --max-seconds 2400 > big.txt 2> big.err
date +%s > big.end
echo "big $(grep -o 'ticks_per_second=[0-9.]*' big.err) $(grep -o 'live=.*' big.err)"
echo big-done
