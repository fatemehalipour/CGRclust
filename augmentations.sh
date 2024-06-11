#!/bin/bash
#SBATCH --account=def-khill22
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=128000M
#SBATCH --time=04:00:00

module avail python
module load python/3.10
#virtualenv --no-download ENV
source ENV/bin/activate

#pip install -r requirements.txt
python3 src/cluster.py --dataset="1_Cypriniformes.fasta" --weak_mutation_rate=1e-4 --strong_mutation_rate=1e-3
python3 src/cluster.py --dataset="1_Cypriniformes.fasta" --weak_mutation_rate=1e-4 --strong_mutation_rate=1e-2
python3 src/cluster.py --dataset="1_Cypriniformes.fasta" --weak_mutation_rate=1e-3 --strong_mutation_rate=1e-2
python3 src/cluster.py --dataset="1_Cypriniformes.fasta" --weak_mutation_rate=None --strong_mutation_rate=None --weak_fragmentation_perc=0.99 --strong_mutation_rate=0.8
python3 src/cluster.py --dataset="1_Cypriniformes.fasta" --weak_mutation_rate=None --strong_mutation_rate=None --weak_fragmentation_perc=0.95 --strong_mutation_rate=0.8
python3 src/cluster.py --dataset="1_Cypriniformes.fasta" --weak_mutation_rate=None --strong_mutation_rate=None --weak_fragmentation_perc=0.95 --strong_mutation_rate=0.7
python3 src/cluster.py --dataset="1_Cypriniformes.fasta" --weak_mutation_rate=None --strong_mutation_rate=None --weak_fragmentation_perc=0.95 --strong_mutation_rate=0.6
python3 src/cluster.py --dataset="1_Cypriniformes.fasta" --weak_mutation_rate=None --strong_mutation_rate=None --weak_fragmentation_perc=0.95 --strong_mutation_rate=0.5
python3 src/cluster.py --dataset="1_Cypriniformes.fasta" --weak_mutation_rate=None --strong_mutation_rate=None --weak_fragmentation_perc=0.9 --strong_mutation_rate=0.8

