import os
import pathlib
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl
import yaml
from pytorch_lightning.loggers import TensorBoardLogger

from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.pl_modules import FastMriDataModule

sys.path.append('.')
from networks.pl_modules import ResidualVarNetModule


def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    # use random masks for train transform, fixed masks for val transform
    train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask)
    test_transform = VarNetDataTransform()
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ------------
    # model
    # ------------
    model = ResidualVarNetModule(
        organize_type=args.organize_type,
        backbone_type=args.backbone_type,
        transform_type=args.transform_type,
        num_blocks=args.num_blocks,
        pools=args.pools,
        chans=args.chans,
        sens_pools=args.sens_pools,
        sens_chans=args.sens_chans,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        # lr_step_size=args.lr_step_size,
        # lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser()
    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced_fraction"),
        default="random",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[4],
        type=int,
        help="Acceleration rates to use for masks",
    )

    # baisc path config
    path_config = pathlib.Path("./fastmri_dirs.yaml")

    # data config
    data_path = fetch_dir("data_path", path_config)
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,  # path to fastMRI data
        mask_type="random",  # VarNet uses equispaced mask
        challenge="multicoil",  # only multicoil implemented for VarNet
        batch_size=1,  # number of samples per batch
        num_workers=0,  # number of workers for PyTorch dataloader if 0 single
        test_path=None,  # path for test split, overwrites data_path
    )

    # module config
    parser = ResidualVarNetModule.add_model_specific_args(parser)
    parser.set_defaults(
        organize_type="cascade",  # type of organization for VarNet blocks, such as cascade or recurrent
        residual_type="add", # type of residual path between VarNet blocks, such as add, mean or none
        backbone_type="unet",  # type of backbone for VarNet, such as unet,nestedunet,attentionunet,SwinUnet,RSTB
        transform_type="fourier",  # type of transformation for K-Space, such as fourier, automap or direct
        consistency_type="soft",  # type of data consistency for K-Space, such as soft, hard or none
        num_blocks=3,  # number of blocks for VarNet
        pools=4,  # number of pooling layers for U-Net
        chans=18,  # number of top-level channels for U-Net
        sens_pools=4,  # number of pooling layers for sense est. U-Net
        sens_chans=8,  # number of top-level channels for sense est. U-Net
        max_epochs=50,  # number of total epochs
        warmup_epochs=3, # number of warmup epochs at beginning
        lr=0.001,  # base learning rate in Adam or AdamW
        # lr_step_size=40,  # epoch at which to decrease learning rate
        # lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.01,  # weight regularization strength
    )

    # trainer config
    default_root_dir = fetch_dir("default_root_dir", path_config)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        accelerator='gpu',
        # gpus=1,  # number of gpus to use
        # precision=16, 
        # amp_backend='apex',
        # amp_level='02',
        deterministic=True,  # makes things slower, but deterministic
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=50,  # max number of epochs
        seed=42,  # random seed
    )

    args = parser.parse_args()

    # tensorboard config
    # tensorboard_dir = default_root_dir / "tensorboard"
    tensorboard_dir = fetch_dir("tensorboard_dir", path_config)
    if not tensorboard_dir.exists():
        tensorboard_dir.mkdir(parents=True)
    
    args.logger = TensorBoardLogger(tensorboard_dir, name="HDR-VarNet")
    # args.log_every_n_steps = 16

    # checkpoint config
    checkpoint_dir = default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory
    # if args.resume_from_checkpoint is None:
    #     ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
    #     if ckpt_list:
    #         args.resume_from_checkpoint = str(ckpt_list[-1])

    args_dict = {key: str(value) for key, value in args.__dict__.items()}
    with open(args.default_root_dir / "args_dict.yaml", "w") as f:
        yaml.dump(args_dict, f)

    return args


def run_cli():
    args = build_args()
    cli_main(args)


if __name__ == "__main__":
    run_cli()
