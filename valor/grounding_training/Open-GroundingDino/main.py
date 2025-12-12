# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import warnings

warnings.filterwarnings("ignore")
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import math
from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import BestMetricHolder
import util.misc as utils

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch

from groundingdino.util.utils import clean_state_dict


import math, torch
import torch.distributed as dist
from torch.utils.data import Sampler
from torch.utils.data.dataset import ConcatDataset


def _concat_spans(concat: ConcatDataset):
    """Return [(start, length), ...] for each child dataset."""
    sizes = [len(ds) for ds in concat.datasets]
    starts, s = [], 0
    for L in sizes:
        starts.append(s)
        s += L
    spans = list(zip(starts, sizes))
    return spans, sizes


class BalancedDistributedConcatSampler(Sampler[int]):
    """
    Uniform over sub-datasets; uniform within chosen sub-dataset. DDP-aware.

    - Keeps epoch length ~ len(concat) by default
    - Works with your existing BatchSampler/DataLoader
    - Call set_epoch(epoch) each epoch (you already do this)
    """

    def __init__(
        self,
        concat: ConcatDataset,
        num_samples_per_epoch: int | None = None,
        seed: int = 0,
        shuffle: bool = True,
    ):
        assert isinstance(
            concat, ConcatDataset
        ), "BalancedDistributedConcatSampler expects a ConcatDataset"
        assert (
            dist.is_available() and dist.is_initialized()
        ), "torch.distributed must be initialized before creating this sampler."

        self.concat = concat
        self.spans, self.sizes = _concat_spans(concat)
        self.num_datasets = len(self.sizes)
        self.shuffle = shuffle

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        # Default epoch size ~ len(concat); round up to be divisible by world_size
        total_default = sum(self.sizes)
        total_desired = (
            total_default if num_samples_per_epoch is None else num_samples_per_epoch
        )
        self.total_size = (
            int(math.ceil(total_desired / self.world_size)) * self.world_size
        )
        self.num_samples = self.total_size // self.world_size

        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Build the full (global) index list of length total_size
        # Step 1: choose a dataset uniformly
        # Step 2: choose an example uniformly within that dataset
        indices = []
        for _ in range(self.total_size):
            ds_id = int(torch.randint(self.num_datasets, (1,), generator=g))
            start, L = self.spans[ds_id]
            idx_in_ds = int(torch.randint(L, (1,), generator=g))
            indices.append(start + idx_in_ds)

        if self.shuffle:
            perm = torch.randperm(self.total_size, generator=g).tolist()
            indices = [indices[i] for i in perm]

        # Shard across ranks
        indices = indices[self.rank : self.total_size : self.world_size]
        assert len(indices) == self.num_samples
        return iter(indices)


def coco_eval_list_to_dict(vals, prefix="test/"):
    """
    Map the 12-element COCO eval list to readable metric names.
    vals: list like test_stats['coco_eval_bbox']
    """
    if vals is None:
        return {}
    names = [
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR@1",
        "AR@10",
        "AR@100",
        "ARs",
        "ARm",
        "ARl",
    ]
    out = {}
    for k, v in zip(names, vals):
        out[f"{prefix}{k}"] = float(v)
    if len(vals) >= 9:
        out[f"{prefix}mAR"] = float(vals[8])  # alias for AR@100
    if len(vals) >= 1:
        out[f"{prefix}mAP"] = float(vals[0])  # alias for COCO AP
    return out


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )

    # dataset parameters
    parser.add_argument(
        "--datasets", type=str, required=True, help="path to datasets json"
    )
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--fix_size", action="store_true")

    # training parameters
    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument("--note", default="", help="add some notes to the experiment")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--pretrain_model_path", help="load from other checkpoint")
    parser.add_argument("--finetune_ignore", type=str, nargs="+")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--enable_eval",
        action="store_true",
        help="Run validation/evaluation after each training epoch.",
    )
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--find_unused_params", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--save_log", action="store_true")

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--local_rank", type=int, help="local rank for DistributedDataParallel"
    )
    parser.add_argument(
        "--local-rank", type=int, help="local rank for DistributedDataParallel"
    )
    parser.add_argument("--amp", action="store_true", help="Train with mixed precision")
    parser.add_argument("--cosine_lr", action="store_true",
                        help="Use cosine annealing LR schedule")
    parser.add_argument("--lr_min", type=float, default=1e-6,
                        help="Minimum LR for cosine annealing (eta_min)")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Number of warmup epochs (linear warmup to base lr)")
    parser.add_argument("--warmup_lr_init", type=float, default=1e-6,
                        help="Initial LR at start of warmup (linearly increases to base lr)")
    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


@torch.no_grad()
def evaluate_loss_only(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader,
    device: torch.device,
    epoch: int,
    wo_class_error=False,
    args=None,
    logger=None,
):
    # mirror train_one_epoch setup, but eval + no optimizer
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter(
            "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
        )
    header = "Eval (loss-only): [{}]".format(epoch)
    print_freq = 10

    _cnt = 0

    for samples, targets in metric_logger.log_every(
        data_loader, print_freq, header, logger=logger
    ):
        samples = samples.to(device)

        # Pull the same fields your train loop uses
        captions = [t["caption"] for t in targets] if "caption" in targets[0] else None
        cap_list = (
            [t["cap_list"] for t in targets] if "cap_list" in targets[0] else None
        )

        # Only tensor fields go to device (same as train)
        targets = [
            {k: v.to(device) for k, v in t.items() if torch.is_tensor(v)}
            for t in targets
        ]

        with torch.cuda.amp.autocast(enabled=args.amp):
            # Call model/criterion exactly like train_one_epoch
            if captions is not None:
                outputs = model(samples, captions=captions)
                loss_dict = criterion(outputs, targets, cap_list, captions)
            else:
                outputs = model(samples)
                # fall back to (outputs, targets) signature if your criterion supports it
                try:
                    loss_dict
                except NameError:
                    loss_dict = criterion(outputs, targets)

            weight_dict = criterion.weight_dict

            losses = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )

        # reduce across ranks for logging
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping eval".format(loss_value))
            print(loss_dict_reduced)
            break

        # identical keys to train loop (minus lr)
        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        if "class_error" in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced["class_error"])

        _cnt += 1
        if args.debug and _cnt % 15 == 0:
            print("BREAK!" * 5)
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats (eval-only):", metric_logger)
    resstat = {
        k: meter.global_avg
        for k, meter in metric_logger.meters.items()
        if meter.count > 0
    }
    return resstat


def main(args):
    utils.setup_distributed(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, "w") as f:
            json.dump(vars(args), f, indent=2)
            
    cfg_dict = cfg._cfg_dict.to_dict()
    _parser = get_args_parser()
    # collect argparse defaults without parsing
    parser_defaults = {
        a.dest: _parser.get_default(a.dest)
        for a in _parser._actions
        if hasattr(a, "dest") and a.dest
    }

    for k, v in cfg_dict.items():
        if not hasattr(args, k):
            # arg isn't defined by argparse → take from config
            setattr(args, k, v)
            continue

        cur = getattr(args, k)
        default = parser_defaults.get(k, None)

        # If the current value still equals the argparse default (i.e., not overridden via CLI),
        # or is None, let the config supply it. Otherwise keep the CLI value.
        if cur == default or cur is None:
            setattr(args, k, v)

    # update some new args temporally
    if not getattr(args, "debug", None):
        args.debug = False

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(
        output=os.path.join(args.output_dir, "info.txt"),
        distributed_rank=args.rank,
        color=False,
        name="detr",
    )

    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: " + " ".join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, "w") as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))

    with open(args.datasets) as f:
        dataset_meta = json.load(f)
    if args.use_coco_eval:
        args.coco_val_path = dataset_meta["val"][0]["anno"]

    logger.info("world size: {}".format(args.world_size))
    logger.info("rank: {}".format(args.rank))
    logger.info("local_rank: {}".format(args.local_rank))
    logger.info("args: " + str(args) + "\n")

    wandb_run = None
    if utils.is_main_process():
        try:
            import wandb  # lazy import

            wandb_run = wandb.init(
                # Set via env vars if you like:
                #   export WANDB_PROJECT="my-project"
                #   export WANDB_ENTITY="my-team"   (optional)
                #   export WANDB_RUN_NAME="exp1"    (optional)
                project=os.environ.get("WANDB_PROJECT", "gd_train"),
                entity=os.environ.get("WANDB_ENTITY", None),
                name=os.environ.get("WANDB_RUN_NAME", "11k_uniform_cosine_warmup"),
                tags=(os.environ.get("WANDB_TAGS", "").split() or None),
                config=vars(args),  # logs hyperparams
                resume="allow",
                mode=os.environ.get("WANDB_MODE", "online"),  # "offline" if needed
            )
            # Make charts step by epoch
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("test/*", step_metric="epoch")
            wandb.define_metric("ours/*", step_metric="epoch")
        except ImportError:
            raise ImportError(
                "wandb is required but not installed. Run `pip install wandb`."
            )

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    logger.debug("build model ... ...")
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)
    logger.debug("build model, done.")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params
        )
        model._set_static_graph()
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("number of params:" + str(n_parameters))
    logger.info(
        "params before freezing:\n"
        + json.dumps(
            {n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
            indent=2,
        )
    )

    param_dicts = get_param_dict(args, model_without_ddp)

    # freeze some layers
    if args.freeze_keywords is not None:
        for name, parameter in model.named_parameters():
            for keyword in args.freeze_keywords:
                if keyword in name:
                    parameter.requires_grad_(False)
                    break
    logger.info(
        "params after freezing:\n"
        + json.dumps(
            {n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
            indent=2,
        )
    )

    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay
    )

    logger.debug("build dataset ... ...")
    run_validation = args.eval or args.enable_eval
    if not args.eval:
        num_of_dataset_train = len(dataset_meta["train"])
        if num_of_dataset_train == 1:
            dataset_train = build_dataset(
                image_set="train", args=args, datasetinfo=dataset_meta["train"][0]
            )
        else:
            from torch.utils.data import ConcatDataset

            dataset_train_list = []
            for idx in range(len(dataset_meta["train"])):
                dataset_train_list.append(
                    build_dataset(
                        image_set="train",
                        args=args,
                        datasetinfo=dataset_meta["train"][idx],
                    )
                )
            dataset_train = ConcatDataset(dataset_train_list)
        logger.debug("build dataset, done.")
        logger.debug(
            f"number of training dataset: {num_of_dataset_train}, samples: {len(dataset_train)}"
        )

    dataset_val = None
    data_loader_eval_ours = None
    if run_validation:
        dataset_val = build_dataset(
            image_set="val", args=args, datasetinfo=dataset_meta["val"][0]
        )

        if "eval_ours" in dataset_meta and len(dataset_meta["eval_ours"]) > 0:
            if len(dataset_meta["eval_ours"]) == 1:
                dataset_eval_ours = build_dataset(
                    image_set="val",  # use val transforms (no heavy augs)
                    args=args,
                    datasetinfo=dataset_meta["eval_ours"][0],
                )
            else:
                from torch.utils.data import ConcatDataset

                dataset_eval_ours = ConcatDataset(
                    [
                        build_dataset("val", args=args, datasetinfo=info)
                        for info in dataset_meta["eval_ours"]
                    ]
                )

            if args.distributed:
                sampler_eval_ours = DistributedSampler(dataset_eval_ours, shuffle=False)
            else:
                sampler_eval_ours = torch.utils.data.SequentialSampler(
                    dataset_eval_ours
                )

            data_loader_eval_ours = DataLoader(
                dataset_eval_ours,
                4,  # match your existing val batch size
                sampler=sampler_eval_ours,
                drop_last=False,
                collate_fn=utils.collate_fn,
                num_workers=args.num_workers,
            )

    if args.distributed:
        if run_validation:
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        if not args.eval:
            sampler_train = DistributedSampler(dataset_train)
            if isinstance(dataset_train, ConcatDataset):
                sampler_train = BalancedDistributedConcatSampler(
                    dataset_train,
                    num_samples_per_epoch=len(
                        dataset_train
                    ),  # keep epoch length ~ same
                    seed=args.seed,
                    shuffle=True,
                )
            # else:
            # Fall back to vanilla DDP sampler if it's a single dataset
            # sampler_train = DistributedSampler(dataset_train, shuffle=True)
    else:
        if run_validation:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        if not args.eval:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if not args.eval:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True
        )
        data_loader_train = DataLoader(
            dataset_train,
            batch_sampler=batch_sampler_train,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
        )

    data_loader_val = None
    if run_validation:
        data_loader_val = DataLoader(
            dataset_val,
            4,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
        )

    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            steps_per_epoch=len(data_loader_train),
            epochs=args.epochs,
            pct_start=0.2,
        )

    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_drop_list
        )

    elif getattr(args, "cosine_lr", False):
        from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR

        remaining_epochs = max(1, args.epochs - int(args.warmup_epochs))
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=remaining_epochs,
            eta_min=args.lr_min,
        )

        if args.warmup_epochs and args.warmup_epochs > 0:
            base_lr = float(args.lr)
            init_lr = float(args.warmup_lr_init)

            def warmup_lambda(epoch):
                # epoch: 0..warmup_epochs-1
                if epoch >= args.warmup_epochs:
                    return 1.0
                if base_lr <= 0:
                    return 1.0
                start = init_lr / base_lr
                progress = float(epoch + 1) / float(args.warmup_epochs)
                return start + (1.0 - start) * progress

            warmup = LambdaLR(optimizer, lr_lambda=warmup_lambda)
            lr_scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[args.warmup_epochs],
            )
        else:
            lr_scheduler = cosine

    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    base_ds = get_coco_api_from_dataset(dataset_val) if run_validation else None

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False
        )

    output_dir = Path(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir, "checkpoint.pth")):
        args.resume = os.path.join(args.output_dir, "checkpoint.pth")
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False
        )

        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")["model"]
        from collections import OrderedDict

        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict(
            {
                k: v
                for k, v in utils.clean_state_dict(checkpoint).items()
                if check_keep(k, _ignorekeywordlist)
            }
        )

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

    if args.eval:
        os.environ["EVAL_FLAG"] = "TRUE"
        test_stats, coco_evaluator = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            args.output_dir,
            wo_class_error=wo_class_error,
            args=args,
        )
        if args.output_dir:
            utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth"
            )

        log_stats = {**{f"test_{k}": v for k, v in test_stats.items()}}

        # Loss-only eval on odvg set (no gradients, no mAP)
        if data_loader_eval_ours is not None:
            ours_stats = evaluate_loss_only(
                model,
                criterion,
                data_loader_eval_ours,
                device,
                args,
                logger=(logger if args.save_log else None),
            )
            if args.output_dir and utils.is_main_process():
                (output_dir / "eval_ours").mkdir(exist_ok=True)
                with (output_dir / "eval_ours" / "latest.json").open("w") as f:
                    json.dump(ours_stats, f, indent=2)

            # Merge into log_stats and wandb
            log_stats.update({f"ours_{k}": v for k, v in ours_stats.items()})

            if utils.is_main_process() and wandb_run is not None:
                import wandb

                wandb.log(
                    {
                        **{f"ours/{k}": float(v) for k, v in ours_stats.items()},
                        "epoch": epoch,
                    }
                )

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return

    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=False)

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            wo_class_error=wo_class_error,
            lr_scheduler=lr_scheduler,
            args=args,
            logger=(logger if args.save_log else None),
        )
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]

        if not args.onecyclelr:
            lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (
                epoch + 1
            ) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                weights = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }

                utils.save_on_master(weights, checkpoint_path)

        if run_validation:
            # eval
            test_stats, coco_evaluator = evaluate(
                model,
                criterion,
                postprocessors,
                data_loader_val,
                base_ds,
                device,
                args.output_dir,
                wo_class_error=wo_class_error,
                args=args,
                logger=(logger if args.save_log else None),
            )

            map_regular = test_stats["coco_eval_bbox"][0]
            _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
            if _isbest:
                checkpoint_path = output_dir / "checkpoint_best_regular.pth"
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
            }

            # Optional: loss-only evaluation for odvg "eval_ours"
            if data_loader_eval_ours is not None:
                ours_stats = evaluate_loss_only(
                    model,
                    criterion,
                    data_loader_eval_ours,
                    device,
                    epoch,
                    wo_class_error=wo_class_error,
                    args=args,
                    logger=(logger if args.save_log else None),
                )
                log_stats.update({f"ours_{k}": v for k, v in ours_stats.items()})
                if utils.is_main_process() and wandb_run is not None:
                    import wandb

                    wandb.log(
                        {**{f"ours/{k}": float(v) for k, v in ours_stats.items()}},
                        step=epoch,
                    )
        else:
            log_stats = {f"train_{k}": v for k, v in train_stats.items()}

        try:
            log_stats.update({"now_time": str(datetime.datetime.now())})
        except:
            pass

        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats["epoch_time"] = epoch_time_str

        if run_validation and utils.is_main_process() and wandb_run is not None:
            import wandb

            wandb_payload = {}

            # 5a) log everything from train_stats as train/<key>
            for k, v in train_stats.items():
                wandb_payload[f"train/{k}"] = (
                    float(v) if isinstance(v, (int, float)) else v
                )

            if run_validation:
                # 5b) log everything from test_stats except the coco list
                for k, v in test_stats.items():
                    if k == "coco_eval_bbox":
                        continue
                    wandb_payload[f"test/{k}"] = (
                        float(v) if isinstance(v, (int, float)) else v
                    )

                # 5c) expand COCO list into friendly names (AP, AP50, AP75, AR@100, etc.)
                wandb_payload.update(
                    coco_eval_list_to_dict(
                        test_stats.get("coco_eval_bbox"), prefix="test/"
                    )
                )

            # (optional) time per epoch
            wandb_payload["epoch_time/seconds"] = float(epoch_time)

            # finally log with epoch as the step
            wandb.log({**wandb_payload, "epoch": epoch})

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if run_validation and coco_evaluator is not None:
                (output_dir / "eval").mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ["latest.pth"]
                    if epoch % 50 == 0:
                        filenames.append(f"{epoch:03}.pth")
                    for name in filenames:
                        torch.save(
                            coco_evaluator.coco_eval["bbox"].eval,
                            output_dir / "eval" / name,
                        )
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get("copyfilelist")
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove

        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)

    if utils.is_main_process() and wandb_run is not None:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
