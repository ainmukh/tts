import argparse
import collections
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

import tts.loss as module_loss
import tts.model as module_arch
from tts.model import Vocoder
from tts.datasets import LJSpeechDataset, get_dataloaders
from tts.collator import LJSpeechCollator
from tts.trainer import Trainer
from tts.utils import prepare_device, ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    # exit(print(config.__dict__))
    logger = config.get_logger("train")

    # setup data_loader instances
    batch_size = config['data']['train']['batch_size']
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    model = config.init_obj(config["arch"], module_arch)
    if config["warm_start"] != "":
        print("Starting from checkpoint", config["warm_start"])
        check_point = torch.load(config["warm_start"])
        model.load_state_dict(check_point["state_dict"])
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    vocoder = Vocoder().to(device).eval()

    # get function handles of loss and metrics
    loss_module = config.init_obj(config["loss"], module_loss).to(device)
    metrics = []

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj(config["optimizer"], torch.optim, trainable_params)
    lr_scheduler = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        model,
        vocoder,
        loss_module,
        optimizer,
        config,
        device,
        dataloaders['train'],
        valid_data_loader=dataloaders['val'],
        lr_scheduler=lr_scheduler,
        len_epoch=config["trainer"].get("len_epoch", None),
        sr=config["preprocessing"]["sr"]
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
