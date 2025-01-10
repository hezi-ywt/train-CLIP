import yaml
import logging
from argparse import ArgumentParser
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from models.jina_wrapper import JinaCLIPWrapper
from data_loader.arrow_load_stream_for_clip import TextImageArrowStream
from pytorch_lightning.strategies import DDPStrategy
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(hparams):
    # 初始化wandb
    wandb_logger = WandbLogger(
        project=hparams.wandb_project,
        name=hparams.run_name,
        version=hparams.version,
        log_model=True,
        save_dir=hparams.default_root_dir,
        config={
            "batch_size": hparams.batch_size,
            "learning_rate": hparams.learning_rate,
            "model_name": "jina-clip-v2",
            "image_size": 512,
            "max_epochs": hparams.max_epochs,
            "num_devices": hparams.num_devices,
        }
    )
    
    # 创建数据集
    dataset = TextImageArrowStream(
        args=hparams,
        image_size=512,
        resize_ratio=0.8,
        resolution=1024,
        random_flip=False,
        log_fn=logger.info,
        index_file=hparams.index_file,
        multireso=True,
        batch_size=hparams.batch_size,
        world_size=getattr(hparams, 'num_devices', 1),
        return_raw_image=False,
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=hparams.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # 创建模型
    model = JinaCLIPWrapper(
        model_name="/mnt/data/jina-clip-v2",
        minibatch_size=hparams.minibatch_size,
        learning_rate=hparams.learning_rate,
        warmup_steps=hparams.warmup_steps,
        max_steps=len(train_dataloader) * hparams.max_epochs,
        truncate_dim=512
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
        logger=wandb_logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="{epoch}-{step}-{loss:.4f}",
                save_top_k=-1,
                monitor="loss",
                mode="min",
                every_n_epochs=hparams.save_every_n_epochs if not hparams.save_every_n_steps else None,
                every_n_train_steps=hparams.save_every_n_steps,
                save_last=True,
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ]
    )
    
    trainer.fit(model)
    wandb.finish()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--index_file', type=str, default="dataset/porcelain/jsons/porcelain_mt.json",
                      help='Path to dataset index file')
    
    # 训练相关参数
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    
    # Trainer相关参数
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--strategy', type=str, default='auto')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--log_every_n_steps', type=int, default=50)
    parser.add_argument('--default_root_dir', type=str, default='./checkpoints')
    
    # 检查点保存相关参数
    parser.add_argument('--save_every_n_epochs', type=int, default=1)
    parser.add_argument('--save_every_n_steps', type=int, default=None)
    
    # wandb相关参数
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--version', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='jina-clip-training',
                      help='WandB project name')
    
    args = parser.parse_args()
    if args.minibatch_size < 1:
        args.minibatch_size = args.batch_size
    
    main(args) 