#!/bin/bash
# heritability of mutant genes seen in the mu runs, in the B3 configuration (1e8 cells, G 256, blank, 35,511 seeds, mu 0)
cd ~/universe2/dna/seeded
export CUDA_VISIBLE_DEVICES=0
E=$HOME/universe2/dna/dna_world_v3
run() { name=$1; shift
  $E --N 99999744 --G 256 --blank --n-seed 35511 --mu 0 --ticks 20000 --seed 1 --no-hash --census-every 10 --depth-cap 10 --log-births $name.log --top-genes $name.top --out $name.json --genome "$@" > $name.txt 2> $name.err
  echo "$name $(grep -o 'ticks_per_second=[0-9.]*' $name.err) $(grep -o 'live=.*' $name.err)"
}
run m_orig  "ATG TTC GGA GCA AAA ATA CTC"
run m_ggg   "ATG TTC GGG GCA AAA ATA CTC"
run m_seek  "ATG TTC GGA GCA AAA ATA CTC GAA"
run m_jnzb  "ATG TTC GTA GCA AAA ATA CTC"
run m_ttt   "ATG TTT GGA GCA AAA ATA CTC"
echo mutants-done
