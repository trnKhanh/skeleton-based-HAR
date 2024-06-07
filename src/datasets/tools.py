from src.datasets.NTUDataset import NTUDataset
from torch.utils.data import DataLoader


def load_dataset(args):
    train_datasets = []
    valid_datasets = []
    train_dataloaders = []
    valid_dataloaders = []
    for feature in args.features:
        if args.train:
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
            train_dataloaders.append(
                DataLoader(
                    dataset=train_datasets[-1],
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
            )

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

        valid_dataloaders.append(
            DataLoader(
                dataset=valid_datasets[-1],
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        )
    return train_dataloaders, valid_dataloaders
