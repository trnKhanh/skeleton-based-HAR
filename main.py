import os
import json
from argparse import ArgumentParser

import numpy as np

from src.datasets.NTUDataset import NTUDataset
from src.models.ctrgcn import Model

from src.utils.optims import StepSchedule
from src.utils.engines import (
    train_one_epoch,
    valid_one_epoch,
    valid_ensemble_one_epoch,
)
from src.datasets.tools import load_dataset
from src.utils.checkpoints import load_checkpoint, save_checkpoint

import torch
from torchsummary import summary
from torch.utils.data import DataLoader
from torch import optim, nn
import random


def init_seed(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--steps", type=int, default=[35, 55], nargs="*")
    parser.add_argument("--decay-rate", type=float, default=0.1)
    parser.add_argument(
        "--graph", default="src.graph.ntu_graph.Graph", type=str
    )
    parser.add_argument("--dataset", default="ntu", choices=["ntu", "ucla"])
    parser.add_argument("--num-points", default=25, type=int)

    parser.add_argument(
        "--score-path", default="", type=str, help="Where to save score"
    )
    parser.add_argument("--train", action="store_true", help="Whether to train")
    parser.add_argument(
        "--valid", action="store_true", help="Whether to validate"
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="Device to use (default: cpu)"
    )
    parser.add_argument(
        "--log-path", default="", type=str, help="Where to save log"
    )
    parser.add_argument(
        "--eval-log-path",
        default="",
        type=str,
        help="Where to save final evaluation log",
    )
    # Dataset
    parser.add_argument(
        "--load-to-ram",
        action="store_true",
        help="Whether to load all dataset to RAM first",
    )
    parser.add_argument(
        "--features",
        default=["j"],
        nargs="+",
        type=str,
        help="Features to use as inputs. If model ensemble is used, this should match the order of given models",
    )
    parser.add_argument(
        "--length-t",
        default=64,
        type=int,
        help="Number of frames in each sample",
    )
    parser.add_argument(
        "--p-intervals",
        default=[1],
        type=float,
        nargs="+",
        help="Percentage of cropped length",
    )
    parser.add_argument(
        "--data-path", required=True, type=str, help="Path to dataset"
    )
    parser.add_argument(
        "--extra-data-path",
        default="",
        type=str,
        help="Path to extra dataset (i.e NTU120)",
    )
    parser.add_argument(
        "--split",
        default="x-subject",
        choices=["x-subject", "x-view", "x-setup"],
        help="Split evaluation (default: x-subject)",
    )
    parser.add_argument(
        "--batch-size", default=64, type=int, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--num-workers",
        default=1,
        type=int,
        help="Number of workers used to load data (default: 1)",
    )
    # Optimizer
    parser.add_argument(
        "--base-lr",
        default=0.005,
        type=float,
        help="Base learning rate (default: 0.005)",
    )
    parser.add_argument(
        "--target-lr",
        default=0.0001,
        type=float,
        help="Target learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--warmup-epochs",
        default=40,
        type=int,
        help="Warm up epochs (default: 40)",
    )
    parser.add_argument(
        "--max-epochs",
        default=80,
        type=int,
        help="Max epochs in cosine schedule (default: 80)",
    )
    # Training
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="Epochs to train (defult: 1000)",
    )
    parser.add_argument(
        "--start-epoch", default=1, type=int, help="Start epoch (defult: 1)"
    )
    # Checkpoint
    parser.add_argument(
        "--save-path", default="", type=str, help="Where to save checkpoint"
    )
    parser.add_argument(
        "--save-freq",
        default=10,
        type=int,
        help="How often to save checkpoint (default: 10)",
    )
    parser.add_argument(
        "--save-best-path",
        default="",
        type=str,
        help="Where to save best checkpoint",
    )
    parser.add_argument(
        "--save-best-acc-path",
        default="",
        type=str,
        help="Where to save highest accuracy checkpoint",
    )
    parser.add_argument(
        "--resume", default="", type=str, help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--load-ckpt", default="", type=str, help="Load checkpoint"
    )
    # Model
    parser.add_argument(
        "--ensemble",
        default=[],
        nargs="*",
        type=str,
        help="List of models used in multi-stream",
    )
    parser.add_argument(
        "--alphas",
        default=[],
        nargs="*",
        type=float,
        help="weight for each model when ensembling",
    )
    parser.add_argument(
        "--num-classes",
        default=60,
        type=int,
        help="Number of classes (default: 60)",
    )
    parser.add_argument(
        "--dropout-rate",
        default=0,
        type=float,
        help="Dropout rate (default: 0)",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Whether to use adaptive graph for graph edges",
    )
    return parser.parse_args()


def main(args):
    init_seed(args.seed)
    train_dataloaders, valid_dataloaders = load_dataset(
        args, init_seed=init_seed
    )
    args.device = torch.device(args.device)

    print("=" * os.get_terminal_size().columns)
    print("Datasets")
    print(f"  Data path: {args.data_path}")
    if len(args.extra_data_path) > 0:
        print(f"  Extra data path: {args.extra_data_path}")
    for id in range(len(args.features)):
        print("-" * os.get_terminal_size().columns)
        print(f"  Feature: {args.features[id]}")
        if args.train:
            print(f"  Train size: {len(train_dataloaders[id].dataset)} samples")
        print(f"  Valid size: {len(valid_dataloaders[id].dataset)} samples")
        print("-" * os.get_terminal_size().columns)
    print("=" * os.get_terminal_size().columns)

    num_features = args.features[0].count(",") + 1
    model = Model(
        in_channels=3 * num_features,
        num_class=args.num_classes,
        num_point=args.num_points,
        graph=args.graph,
        graph_args=dict(labeling_mode="spatial"),
    )
    model.to(args.device)
    num_params = sum([p.numel() for p in model.parameters()])
    optimizer = optim.SGD(
        model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0004
    )

    warmup_steps = args.warmup_epochs
    max_steps = args.max_epochs
    lr_scheduler = StepSchedule(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        base_lr=args.base_lr,
        steps=args.steps,
        decay_rate=args.decay_rate,
        cur_step=args.start_epoch - 1,
    )
    if len(args.load_ckpt) > 0 and os.path.isfile(args.load_ckpt):
        state_dict = torch.load(args.load_ckpt, map_location=args.device)
        if "model" in state_dict:
            model.load_state_dict(state_dict["model"])
        else:
            model.load_state_dict(state_dict)
        print(f"Loaded model from {args.load_ckpt}")

    min_loss = np.Inf
    max_acc = -np.Inf

    if len(args.resume) > 0 and os.path.isfile(args.resume):
        args.start_epoch, min_loss = load_checkpoint(
            args.resume,
            model,
            optimizer,
            lr_scheduler,
            args.device,
        )
        print(f"Resuming from epoch {args.start_epoch}, min_loss={min_loss}")

    loss_fn = nn.CrossEntropyLoss()

    if args.train:
        print("=" * os.get_terminal_size().columns)
        print("Model")
        print(model)
        print(f"  Number of parameters: {num_params}")
        print("=" * os.get_terminal_size().columns)
        log = []
        if len(args.log_path) > 0:
            os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
        if len(args.save_path) > 0:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        for e in range(args.start_epoch, args.epochs + 1):
            train_avg_loss, train_acc = train_one_epoch(
                epoch=e,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                dataloader=train_dataloaders[0],
                device=args.device,
                lr_schedule=lr_scheduler,
            )
            valid_avg_loss, valid_acc, _, _, score = valid_one_epoch(
                model=model,
                loss_fn=loss_fn,
                dataloader=valid_dataloaders[0],
                device=args.device,
            )
            if len(args.log_path) > 0:
                log.append(
                    {
                        "epoch": e,
                        "train_avg_loss": train_avg_loss,
                        "train_acc": train_acc,
                        "valid_avg_loss": valid_avg_loss,
                        "valid_acc": valid_acc,
                    }
                )
                with open(args.log_path, "w", encoding="utf-8") as f:
                    json.dump(log, f, ensure_ascii=False, indent=2)

            if valid_avg_loss < min_loss and len(args.save_best_path) > 0:
                save_checkpoint(
                    args.save_best_path,
                    model,
                    optimizer,
                    lr_scheduler,
                    e,
                    valid_avg_loss,
                )

            if valid_acc > max_acc and len(args.save_best_acc_path) > 0:
                save_checkpoint(
                    args.save_best_acc_path,
                    model,
                    optimizer,
                    lr_scheduler,
                    e,
                    valid_avg_loss,
                )
            min_loss = min(min_loss, valid_avg_loss)
            max_acc = max(max_acc, valid_acc)

            if len(args.save_path) > 0:
                if (e % args.save_freq) == 0 or e == args.epochs:
                    save_checkpoint(
                        args.save_path,
                        model,
                        optimizer,
                        lr_scheduler,
                        e,
                        min_loss,
                    )

    if args.valid:
        valid_avg_loss, valid_acc, preds, labels, score = valid_one_epoch(
            model=model,
            loss_fn=loss_fn,
            dataloader=valid_dataloaders[0],
            device=args.device,
        )
        if len(args.score_path):
            if score is not None:
                score_np = score.cpu().detach().numpy()

                os.makedirs(os.path.dirname(args.score_path), exist_ok=True)
                np.save(args.score_path, score_np)
                print(f"Saved scores to {args.score_path}")

        print(f"Evalution accuracy: {valid_acc}")
        if len(args.eval_log_path) > 0:
            os.makedirs(os.path.dirname(args.eval_log_path), exist_ok=True)
            with open(args.eval_log_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"preds": preds, "labels": labels},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )


if __name__ == "__main__":
    args = create_args()
    main(args)
