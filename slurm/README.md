# Slurm Submission Scripts

---

This directory contains example scripts for running training/inference on
CIFAR10 and Reddit datasets over multiple nodes on ZIB A100 and PVC compute
partitions. Note that 2x PVC tiles are used in comparison with 1X A100,
therefore batch sizes are halved in the former jobs.

Jobs can be submitted to SLURM scheduling using:

```
sbatch JOB_SCRIPT.sh
```
