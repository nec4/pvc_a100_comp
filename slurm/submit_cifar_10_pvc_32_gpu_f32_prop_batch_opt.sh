#! /bin/bash
#SBATCH --job-name cifar_10_pvc_bench_32_gpu_f32_prop_batch
#SBATCH -o cifar_10_pvc_bench_32_gpu_f32_prop_batch.out
#SBATCH -t 00:50:00
#SBATCH -p gpu-pvc
#SBATCH --gres gpu:4
#SBATCH -N 4
#SBATCH --mem-per-cpu 1G
#SBATCH --cpus-per-task 16
#SBATCH --ntasks-per-node 8
#SBATCH --exclude bgi1004

source ~/.bashrc
module load intel/2024.0.0
module load impi
module load intel_AI_tools

export CCL_ROOT=/sw/compiler/intel/oneapi/ccl/2021.12
export LD_LIBRARY_PATH=$I_MPI_ROOT/lib:$LD_LIBRARY_PATH
export CCL_WORKER_COUNT=1
hnode=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$(scontrol getaddrs $hnode | cut -d' ' -f 2 | cut -d':' -f 1)
export MASTER_PORT=29500
export FI_PROVIDER=psm3
export I_MPI_OFFLOAD=1
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ZE_AFFINITY_MASK=0.0,0.1,1.0,1.1,2.0,2.1,3.0,3.1

conda activate intel_pytorch_2.1.0a0

scontrol show hostname

mpirun -v -np 32 -ppn 8 -print-rank-map -prepend-rank python script_cifar_10_pvc_bench.py --gpus 32 --nodes 4 --dtype f32 --batch_size 256 --optimize true
