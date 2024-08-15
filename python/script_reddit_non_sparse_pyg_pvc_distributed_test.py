import torch
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch
import logging
import os
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import ChebConv

dtype_dict = {"f32": torch.float32, "bf16": torch.bfloat16}


class Cheb(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 8,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(ChebConv(in_channels, hidden_channels, 8))
        for _ in range(num_layers - 2):
            self.convs.append(ChebConv(hidden_channels, hidden_channels, 8))
        self.convs.append(ChebConv(hidden_channels, out_channels, 8))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reddit/PyG DDP script")
    parser.add_argument(
        "--hardware",
        type=str,
        default="intel",
        choices=["nvidia", "intel"],
        help="hardware vendor",
    )
    parser.add_argument(
        "--hierarchy",
        type=str,
        default="COMPOSITE",
        #choices=["COMPOSITE", "FLAT", "COMBINED"],
        help="The set ZE_FLAT_DEVICE_HIERARCHY environment variable value",
    )
    parser.add_argument(
        "--epochs", type=int, default=25, help="Number of training epochs per run"
    )
    parser.add_argument(
        "--repeats", type=int, default=5, help="Number of training runs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Training batch size"
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=65536,
        help="size of training dataset",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/scratch/usr/bzfbnick/intel_bench/pytorch/pyg/Reddit",
        help="Directory containing CIFAR10 images",
    )
    parser.add_argument(
        "--optimize",
        type=bool,
        default=False,
        help="If 'true', IPEX model optimizations/compilations will be made before training. Only valid for Intel hardware.",
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

    dist.init_process_group("ccl", rank=mpi_rank, world_size=mpi_world_size)

    device = f"xpu:{mpi_local_rank}"

    dataset = Reddit(args.dataset_root)
    data = dataset[0]
    # data = data.to(rank, 'x', 'y')  # Move to device for faster feature fetch.

    # Split indices into `world_size` many chunks:
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx[: args.dataset_size]
    train_idx = train_idx.split(train_idx.size(0) // mpi_world_size)[mpi_rank]

    kwargs = dict(
        data=data,
        batch_size=args.batch_size,
        num_neighbors=[8, 8, 8, 8],
        drop_last=True,
        num_workers=1,
    )
    train_loader = NeighborLoader(
        input_nodes=train_idx,
        shuffle=True,
        **kwargs,
    )

    model = Cheb(dataset.num_features, 256, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    opt_dtype = dtype if args.dtype == "bf16" else None
    if args.optimize == True:
        model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=opt_dtype)

    model = DistributedDataParallel(model, device_ids=[device])

    if mpi_rank == 0:
        logname = f"reddit_pyg_{args.partition}_nodes_{args.nodes}_gpus_{args.gpus}_batchsize_{args.batch_size}_epochs_{args.epochs}_optimize_{args.optimize}_dtype_{args.dtype}_hierarchy_{args.hierarchy}"
        logging.basicConfig(
            filename=logname + ".log",
            format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )
        logging.FileHandler(logname + ".log")

    for repeat in range(args.repeats):
        dist.barrier()

        for epoch in range(args.epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch.x.to(device), batch.edge_index.to(device))[
                    : batch.batch_size
                ]
                loss = F.cross_entropy(out, batch.y.to(device)[: batch.batch_size])
                loss.backward()
                optimizer.step()

            torch.xpu.synchronize(device)
            dist.barrier()

            if mpi_rank == 0:
                logging.info(f"Epoch {epoch} finished")
        if mpi_rank == 0:
            logging.info(f"Run {repeat} finished")

    dist.destroy_process_group()
