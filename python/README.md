# Training and Inference Scripts

---

These scripts define the training/inference routines for A100 and PVC jobs for
CIFAR10 and Reddit datasets. They are separated by task (inference/training) and
hardware (intel, nvidia) due to the nature of extension packages/imports/comm
backends -- otherwise the tasks should be the same between the two. Tasks are
distributed and pinned using Intel MPI and `mpirun` (see `slurm` directory
above).


