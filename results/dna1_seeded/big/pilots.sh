#!/bin/bash
# DNA-1 seeded pilots at N = 99,999,744 cells, 20,000 ticks each (docs/12). GPU 0 of adler40.
cd ~/universe2/dna/seeded/runs
export CUDA_VISIBLE_DEVICES=0
E=../dna_world_v2
G1="ATG TTC GGA GCA AAA ATA CTC"; G2="ATG GCA TTC GGA AAA ATA CTC"
N=99999744
run() { name=$1; shift
  if [ -f $name.err ]; then echo "skip $name"; return; fi
  $E --N $N --ticks 20000 --seed 1 --no-hash --census-every 10 --depth-cap 10 --log-births $name.log --top-genes $name.top --out $name.json "$@" > $name.txt 2> $name.err
  echo "$name $(grep -o 'ticks_per_second=[0-9.]*' $name.err) $(grep -o 'live=.*' $name.err)"
}
# group 1: as specified (G 1024, random soup, 4,096 seeds)
run p0            --G 1024 --n-seed 4096 --genome "$G1" --mu 0
run p0_mu001      --G 1024 --n-seed 4096 --genome "$G1" --mu 0.001
run p0_mu01       --G 1024 --n-seed 4096 --genome "$G1" --mu 0.01
run p0_nodouble   --G 1024 --n-seed 4096 --genome "$G1" --mu 0 --no-double --D 950
run p0_g2         --G 1024 --n-seed 4096 --genome "$G2" --mu 0
# group 2: G 256 (the sweep's slot size), 390,624 slots
run s256_soup4096 --G 256 --n-seed 4096 --genome "$G1" --mu 0
run b256_4096     --G 256 --blank --n-seed 4096 --genome "$G1" --mu 0
run b256_35511    --G 256 --blank --n-seed 35511 --genome "$G1" --mu 0
run b256_97656    --G 256 --blank --n-seed 97656 --genome "$G1" --mu 0
run b256_130208   --G 256 --blank --n-seed 130208 --genome "$G1" --mu 0
run b256_195312   --G 256 --blank --n-seed 195312 --genome "$G1" --mu 0
run b256_35511_F0 --G 256 --blank --n-seed 35511 --genome "$G1" --mu 0 --F 0
run b256_195312_F0 --G 256 --blank --n-seed 195312 --genome "$G1" --mu 0 --F 0
run b256_35511_mu001 --G 256 --blank --n-seed 35511 --genome "$G1" --mu 0.001
run b256_35511_mu01  --G 256 --blank --n-seed 35511 --genome "$G1" --mu 0.01
run b256_35511_nodouble --G 256 --blank --n-seed 35511 --genome "$G1" --mu 0 --no-double --D 950
run b256_35511_g2    --G 256 --blank --n-seed 35511 --genome "$G2" --mu 0
# group 3: G 1024 blank, lifetime 64 and 256
run b1024_48828      --G 1024 --blank --n-seed 48828 --genome "$G1" --mu 0
run b1024_48828_T256 --G 1024 --blank --n-seed 48828 --genome "$G1" --mu 0 --T 256
run b1024_8878_T256  --G 1024 --blank --n-seed 8878 --genome "$G1" --mu 0 --T 256
run s1024_4096_T256  --G 1024 --n-seed 4096 --genome "$G1" --mu 0 --T 256
echo pilots-done
