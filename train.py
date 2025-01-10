import yaml
import logging
from argparse import ArgumentParser
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from models import CLIPWrapper
from data_loader.arrow_load_stream_for_clip import TextImageArrowStream
from IndexKits.index_kits.sampler import DistributedSamplerWithStartIndex, BlockDistributedSampler

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(hparams):
    config_dir = 'models/configs/ViT.yaml' if 'ViT' in hparams.model_name else 'models/configs/RN.yaml'
    with open(config_dir) as fin:
        config = yaml.safe_load(fin)[hparams.model_name]

    if hparams.minibatch_size < 1:
        hparams.minibatch_size = hparams.batch_size

    # 创建数据集和dataloader
    dataset = TextImageArrowStream(
        args=hparams,
        image_size=config.get('image_size', 224),
        resize_ratio=config.get('resize_ratio', 0.8),
        resolution=config.get('resolution', 1024),
        random_flip=config.get('random_flip', False),
        log_fn=logger.info,
        index_file=config.get('index_file', "dataset/porcelain/jsons/porcelain_mt.json"),
        multireso=config.get('multireso', True),
        batch_size=hparams.batch_size,
        world_size=getattr(hparams, 'num_devices', 1),
    )

    # 设置分布式训练参数
    world_size = getattr(hparams, 'num_devices', 1)
    rank = getattr(hparams, 'local_rank', 0)

    # 创建采样器
    if config.get('multireso', True):
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

    # 创建模型并传入dataloader
    model = CLIPWrapper(hparams.model_name, config, hparams.minibatch_size)
    model.train_dataloader = lambda: train_dataloader  # 添加train_dataloader方法
    
    # 配置训练器
    trainer = Trainer(
        precision="16-mixed",
        max_epochs=hparams.max_epochs,
        accelerator=hparams.accelerator,
        devices=hparams.num_devices,
        strategy=hparams.strategy if hparams.num_devices > 1 else None,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        log_every_n_steps=hparams.log_every_n_steps,
        enable_checkpointing=hparams.enable_checkpointing,
        default_root_dir=hparams.default_root_dir,
    )
    
    trainer.fit(model)  # 不需要再传入dataloader

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    
    # 添加trainer相关的参数
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--strategy', type=str, default='auto')
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--log_every_n_steps', type=int, default=50)
    parser.add_argument('--enable_checkpointing', action='store_true')
    parser.add_argument('--default_root_dir', type=str, default='./checkpoints')
    
    args = parser.parse_args()

    main(args)
