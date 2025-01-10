import yaml
import logging
from argparse import ArgumentParser
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from models.hf_wrapper import HFCLIPWrapper
from data_loader.arrow_load_stream_for_clip import TextImageArrowStream
from IndexKits.index_kits.sampler import DistributedSamplerWithStartIndex, BlockDistributedSampler
from pytorch_lightning.strategies import DDPStrategy
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(hparams):
    # 初始化wandb
    wandb_logger = WandbLogger(
        project="clip-training",  # 项目名称
        name=hparams.run_name,    # 运行名称
        version=hparams.version,  # 版本
        log_model=True,          # 是否记录模型检查点
        save_dir=hparams.default_root_dir,
        # 记录所有超参数
        config={
            "batch_size": hparams.batch_size,
            "learning_rate": 5e-5,  # 从模型配置中获取
            "model_name": "clip-vit-base-patch32",
            "image_size": 224,
            "max_epochs": hparams.max_epochs,
            "num_devices": hparams.num_devices,
            # 添加其他你想跟踪的参数
        }
    )

    # 创建数据集和dataloader
    dataset = TextImageArrowStream(
        args=hparams,
        image_size=224,
        resize_ratio=0.8,
        resolution=1024,
        random_flip=False,
        log_fn=logger.info,
        index_file="dataset/porcelain/jsons/porcelain_mt.json",
        multireso=True,
        batch_size=hparams.batch_size,
        world_size=getattr(hparams, 'num_devices', 1),
    )

    # 设置分布式训练参数
    world_size = getattr(hparams, 'num_devices', 1)
    rank = getattr(hparams, 'local_rank', 0)

    # 创建采样器
    if True:  # multireso
        sampler = BlockDistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=rank, 
            seed=hparams.seed,
            shuffle=True, 
            drop_last=True, 
            batch_size=hparams.batch_size
        )
    else:
        sampler = DistributedSamplerWithStartIndex(
            dataset, 
            num_replicas=world_size, 
            rank=rank, 
            seed=hparams.seed,
            shuffle=False, 
            drop_last=True
        )
        
    # 创建数据加载器
    train_dataloader = DataLoader(
        dataset, 
        batch_size=hparams.batch_size, 
        shuffle=False, 
        sampler=sampler,
        num_workers=hparams.num_workers, 
        pin_memory=True, 
        drop_last=True
    )

    # 创建模型
    model = HFCLIPWrapper(
        model_name="/mnt/data/clip-vit-base-patch32",  # 改回使用标准CLIP模型
        minibatch_size=hparams.minibatch_size
    )
    
    
    model.train_dataloader = lambda: train_dataloader
    
    # 创建检查点保存目录
    checkpoint_dir = Path(hparams.default_root_dir) / "checkpoints" / f"run_{hparams.run_name}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 配置训练器
    trainer = Trainer(
        precision="16-mixed",
        max_epochs=hparams.max_epochs,
        accelerator=hparams.accelerator,
        devices=hparams.num_devices,
        strategy=DDPStrategy(find_unused_parameters=True) if hparams.num_devices > 1 else None,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        log_every_n_steps=hparams.log_every_n_steps,
        enable_checkpointing=True,
        default_root_dir=hparams.default_root_dir,
        logger=wandb_logger,  # 使用wandb logger
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=str(checkpoint_dir),  # 使用新的检查点目录
                filename="{epoch}-{step}-{train_loss:.4f}",
                save_top_k=-1,  # -1 表示保存所有检查点
                monitor="train_loss",
                mode="min",
                every_n_epochs=hparams.save_every_n_epochs if not hparams.save_every_n_steps else None,
                every_n_train_steps=hparams.save_every_n_steps,
                save_last=True,
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ]
    )
    
    trainer.fit(model)
    
    # 完成训练后结束wandb运行
    wandb.finish()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    
    # 添加trainer相关的参数
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--strategy', type=str, default='auto')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--log_every_n_steps', type=int, default=50)
    parser.add_argument('--enable_checkpointing', action='store_true')
    parser.add_argument('--default_root_dir', type=str, default='./checkpoints')
    
    # 添加版本参数
    parser.add_argument('--version', type=str, default=None,
                      help='Experiment version for logging')
    
    # 添加wandb相关参数
    parser.add_argument('--run_name', type=str, default=None,
                      help='Name for this training run')
    parser.add_argument('--wandb_project', type=str, default='clip-training',
                      help='WandB project name')
    
    # 在 ArgumentParser 中添加新参数
    parser.add_argument('--save_every_n_epochs', type=int, default=1,
                      help='Save checkpoint every n epochs')
    parser.add_argument('--save_every_n_steps', type=int, default=None,
                      help='Save checkpoint every n steps (if set, overrides save_every_n_epochs)')
    
    # 在 ArgumentParser 中添加文件名格式参数
    parser.add_argument('--checkpoint_filename', type=str, 
                      default='{epoch}-{step}-{train_loss:.4f}',
                      help='Checkpoint filename format')
    
    args = parser.parse_args()
    if args.minibatch_size < 1:
        args.minibatch_size = args.batch_size

    main(args) 