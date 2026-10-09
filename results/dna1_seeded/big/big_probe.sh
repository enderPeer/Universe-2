#!/bin/bash
# 3e9 probes (short): the as-specified configuration (G 1024, soup, 1M seeds, mu 0.001) and the G 256 blank one, to
# document sterility at full size and to measure the tick rate before the 40-minute run.
cd ~/universe2/dna/seeded/runs
export CUDA_VISIBLE_DEVICES=0
E=../dna_world_v2
G1="ATG TTC GGA GCA AAA ATA CTC"
$E --N 2999999488 --G 1024 --ticks 2000 --seed 1 --no-hash --census-every 10 --depth-cap 10 --n-seed 1000000 --genome "$G1" --mu 0.001 --log-births big_spec.log --top-genes big_spec.top --out big_spec.json > big_spec.txt 2> big_spec.err
echo "big_spec $(grep -o 'ticks_per_second=[0-9.]*' big_spec.err) $(grep -o 'live=.*' big_spec.err)"
$E --N 2999999488 --G 256 --ticks 500 --seed 1 --no-hash --census-every 100 --depth-cap 10 --blank --n-seed 1000000 --genome "$G1" --mu 0.001 --log-births big_probe.log --top-genes big_probe.top --out big_probe.json > big_probe.txt 2> big_probe.err
echo "big_probe $(grep -o 'ticks_per_second=[0-9.]*' big_probe.err) $(grep -o 'live=.*' big_probe.err)"
echo probe-done
