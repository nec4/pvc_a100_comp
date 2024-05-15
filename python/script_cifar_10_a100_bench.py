import torch
from typing import Tuple
import os
import numpy as np
import torchvision
from torchvision.transforms import v2
import logging
from datetime import datetime
import argparse
from torch.utils.data import Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


dtype_dict = {"f32": torch.float32, "bf16": torch.bfloat16}


def cleanup():
    dist.destroy_process_group()


def prepare_data(
    rank,
    world_size,
    batch_size=32,
    pin_memory=False,
    num_workers=0,
    dtype=torch.float32,
):
    DATA = "/scratch/usr/bzfbnick/intel_bench/pytorch/distributed/"
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            v2.ToDtype(dtype, scale=False),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA,
        train=True,
        transform=transform,
        download=True,
    )
    train_dataset = Subset(train_dataset, np.arange(4096))  # take just first 32 batches
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        sampler=sampler,
    )
    return dataloader


def main():

    parser = argparse.ArgumentParser(description="RESNET/CIFAR10 DDP script")
    parser.add_argument(
        "--hardware",
        type=str,
        default="intel",
        choices=["nvidia", "intel"],
        help="hardware vendor",
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="Number of training epochs per run"
    )
    parser.add_argument(
        "--repeats", type=int, default=5, help="Number of training runs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=16384,
        help="Number of CIFAR 10 images to use for training dataset",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/scratch/usr/bzfbnick/pytorch_benchmarking/pytorch-hpc/examples/fit/local_datasets",
        help="Directory containing CIFAR10 images",
    )
    parser.add_argument(
        "--optimize",
        type=bool,
        default=False,
        help="If 'true', model optimizations/compilations will be made before training.",
    )
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus", type=int, default=4, help="Number of gpus")
    parser.add_argument(
        "--partition", type=str, default="gpu-pvc", help="Cluster partition name"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="f32",
        choices=["f32", "bf16"],
        help="Model/data precision",
    )

    args = parser.parse_args()

    dtype = dtype_dict[args.dtype]

    mpi_world_size = int(os.environ.get("PMI_SIZE", -1))
    mpi_rank = int(os.environ.get("PMI_RANK", -1))
    mpi_local_rank = int(os.environ.get("MPI_LOCALRANKID", -1))
    if mpi_world_size > 0:
        os.environ["RANK"] = str(mpi_rank)
        os.environ["WORLD_SIZE"] = str(mpi_world_size)
    else:
        # set the default rank and world size to 0 and 1
        os.environ["RANK"] = str(os.environ.get("RANK", 0))
        os.environ["WORLD_SIZE"] = str(os.environ.get("WORLD_SIZE", 1))

    # os.environ["MASTER_ADDR"] = args.dist_url  # your master address
    # os.environ["MASTER_PORT"] = args.dist_port  # your master port
    # Initialize the process group with (n)ccl backend
    dist.init_process_group(
        backend="ccl" if args.hardware == "intel" else "nccl",
        world_size=mpi_world_size,
        rank=mpi_rank,
    )

    global_rank = dist.get_rank()
    device = "cuda:{}".format(mpi_local_rank)

    train_loader = prepare_data(global_rank, mpi_world_size, dtype=dtype)

    model = torchvision.models.resnet50()
    model.to(device)
    model.to(dtype)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    if args.optimize == True:
        model = torch.compile(model)

    model = DDP(model, device_ids=[device])

    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer.zero_grad()

    if global_rank == 0:
        logname = f"cifar_10_{args.partition}_nodes_{args.nodes}_gpus_{args.gpus}_batchsize_{args.batch_size}_epochs_{args.epochs}_optimize_{args.optimize}_dtype_{args.dtype}"
        logging.basicConfig(
            filename=logname + ".log",
            format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )
        logging.FileHandler(logname + ".log")

    for repeat in range(args.repeats):
        for epoch in range(args.epochs):
            for idx, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            if global_rank == 0:
                logging.info(f"Epoch {epoch} finished")
        if global_rank == 0:
            logging.info(f"Run {repeat} finished")

    cleanup()


if __name__ == "__main__":
    main()
