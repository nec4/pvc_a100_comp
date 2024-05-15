#! /bin/bash
#SBATCH --job-name cifar_10_a100_16_gpu_f32_bench_opt
#SBATCH -o cifar_10_a100_bench_16_gpu_f32_opt.out
#SBATCH -t 00:50:00
#SBATCH -p gpu-a100
#SBATCH --gres gpu:4
#SBATCH -N 4
#SBATCH --mem-per-cpu 1G
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node 4

source ~/.bashrc
module load cuda/11.8
module load anaconda3/2023.09
module load impi/2021.11

conda activate base

scontrol show hostname
hnode=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$(scontrol getaddrs $hnode | cut -d' ' -f 2 | cut -d':' -f 1)
export MASTER_PORT=29500
export PATH=/sbin:$PATH # to find ldconfig

echo $MASTER_ADDR
echo $MASTER_PORT

mpirun -v -np 16 -ppn 4 -print-rank-map -prepend-rank python script_cifar_10_a100_bench.py --partition gpu-a100 --hardware nvidia --gpus 16 --nodes 4 --dtype f32 --optimize true 
