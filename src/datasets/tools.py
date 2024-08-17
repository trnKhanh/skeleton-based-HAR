from src.datasets.NTUDataset import NTUDataset
from src.datasets.KineticDataset import KineticDataset
from torch.utils.data import DataLoader


def load_dataset(args, init_seed):
    train_datasets = []
    valid_datasets = []
    train_dataloaders = []
    valid_dataloaders = []
    for feature in args.features:
        if args.train:
            if args.dataset == "ntu":
                train_datasets.append(
                    NTUDataset(
                        data_path=args.data_path,
                        extra_data_path=args.extra_data_path,
                        mode="train",
                        split=args.split,
                        features=feature,
                        length_t=args.length_t,
                        p_interval=args.p_intervals,
                        load_to_ram=args.load_to_ram,
                    )
                )
            elif args.dataset == "kinetic":
                train_datasets.append(
                    KineticDataset(
                        data_path=args.data_path,
                        extra_data_path=args.extra_data_path,
                        mode="train",
                        features=feature,
                        length_t=args.length_t,
                        p_interval=args.p_intervals,
                        load_to_ram=args.load_to_ram,
                    )
                )

            train_dataloaders.append(
                DataLoader(
                    dataset=train_datasets[-1],
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    worker_init_fn=init_seed,
                )
            )
        if args.dataset == "ntu":
            valid_datasets.append(
                NTUDataset(
                    data_path=args.data_path,
                    extra_data_path=args.extra_data_path,
                    mode="valid",
                    split=args.split,
                    features=feature,
                    length_t=args.length_t,
                    load_to_ram=args.load_to_ram,
                )
            )
        elif args.dataset == "kinetic":
            valid_datasets.append(
                KineticDataset(
                    data_path=args.data_path,
                    extra_data_path=args.extra_data_path,
                    mode="valid",
                    features=feature,
                    length_t=args.length_t,
                    load_to_ram=args.load_to_ram,
                )
            )

        valid_dataloaders.append(
            DataLoader(
                dataset=valid_datasets[-1],
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=init_seed,
            )
        )
    return train_dataloaders, valid_dataloaders
