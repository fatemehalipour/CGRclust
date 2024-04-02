#!/bin/bash
#SBATCH --account=def-khill22
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=128000M
#SBATCH --time=06:00:00

module avail python
module load python/3.10
#virtualenv --no-download ENV
source ENV/bin/activate

#pip install -r requirements.txt
python3 src/cluster.py --dataset="1_Cypriniformes" --num_epochs=150 --lr=7e-5 --weight=0.7
python3 src/cluster.py --dataset="2_Cyprinoidei" --num_epochs=150 --lr=7e-5 --weight=0.7
python3 src/cluster.py --dataset="3_Cyprinidae" --num_epochs=150 --lr=7e-5 --weight=0.7
python3 src/cluster.py --dataset="4_Cyprininae" --num_epochs=150 --lr=7e-5 --weight=0.7
python3 src/cluster.py --dataset="5_Fungi" --num_epochs=150 --lr=7e-5 --weight=0.7
python3 src/cluster.py --dataset="6_Insetcs" --num_epochs=150 --lr=7e-5 --weight=0.7
python3 src/cluster.py --dataset="7_Protists" --num_epochs=150 --lr=7e-5 --weight=0.7
python3 src/cluster.py --dataset="8_Astrovirus" --num_epochs=150 --lr=7e-5 --weight=0.7
