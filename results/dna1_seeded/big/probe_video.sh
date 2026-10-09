#!/bin/bash
# the video run (slot map every 20 ticks) at 1e8 cells, then the 3e9 probes
cd ~/universe2/dna/seeded/runs
export CUDA_VISIBLE_DEVICES=0
E=../dna_world_v2
G1="ATG TTC GGA GCA AAA ATA CTC"
$E --N 99999744 --G 256 --ticks 20000 --seed 1 --no-hash --census-every 20 --depth-cap 10 --blank --n-seed 35511 --genome "$G1" --mu 0.001 --log-births video.log --top-genes video.top --slot-map video.map --out video.json > video.txt 2> video.err
echo "video $(grep -o 'ticks_per_second=[0-9.]*' video.err) $(grep -o 'live=.*' video.err)"
bash ../big_probe.sh
