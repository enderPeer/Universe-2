#!/bin/bash
cd ~/universe2/dna && nvcc -O3 -std=c++17 -arch=sm_86 -o dna_world_v5 dna_world_v5.cu 2>&1 | head -5
cd ~/universe2/dna/seeded
export CUDA_VISIBLE_DEVICES=1
E=$HOME/universe2/dna/dna_world_v5
G1="ATG TTC GGA GCA AAA ATA CTC"
$E --N 99999744 --G 256 --blank --n-seed 35511 --mu 0 --ticks 20000 --seed 1 --no-hash --census-every 10 --depth-cap 10 --log-births src_35511.log --log-codons --top-genes src_35511.top --out src_35511.json --genome "$G1" > src_35511.txt 2> src_35511.err
echo "src_35511 $(grep -o 'ticks_per_second=[0-9.]*' src_35511.err) $(grep -o 'live=.*' src_35511.err)"
# v4 hash check on the v1 configuration against the reference output copied from adler40
$E --N 65536 --G 256 --ticks 300 --seed 1 --blank --n-seed 128 --genome "$G1" --mu 0 --hash-every 1 --census-every 10 --births v1.v5.births --log-births v1.v5.log --log-codons > v1.v5.txt 2> v1.v5.err
cmp -s <(grep ^tick v1.ref.txt) <(grep ^tick v1.v5.txt) && cmp -s v1.ref.births v1.v5.births && echo "v1.v5 OK" || echo "v1.v5 MISMATCH"
echo src-done
