#!/bin/bash
#SBATCH -n 24
#SBATCH --time 12:00:00             
#SBATCH --mem-per-cpu=4096
#SBATCH --gpus=v100:1
#SBATCH -J tpu_net
#SBATCH -o ./slurm_out/tpu_net_%j.out


# load modules
module load python_gpu/3.11.2
source /cluster/scratch/jafluri/tpu_venv/bin/activate

# run
python training.py --data_path ../../predict-ai-model-runtime/npz_all/npz/layout/xla/default/ --data_path ../../predict-ai-model-runtime/npz_all/npz/layout/xla/random/ --epochs 5 --cosine_annealing --cache --layout_network --batch_size 4 --list_size 16
