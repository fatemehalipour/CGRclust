#!/bin/bash
#SBATCH --account=def-khill22
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=128000M
#SBATCH --time=02:00:00

module avail python
module load python/3.10
virtualenv --no-download ENV
source ENV/bin/activate

pip install -r requirements.txt
python3 src/cluster.py --dataset="Cypriniformes.fasta"

