import torch
from pytorch_lightning import Trainer
from models.hf_wrapper import HFCLIPWrapper
from data_loader.arrow_load_stream_for_clip import TextImageArrowStream
from torch.utils.data import DataLoader
import logging
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    # 创建验证数据集
    val_dataset = TextImageArrowStream(
        args=args,
        image_size=224,
        resize_ratio=0.8,
        resolution=1024,
        random_flip=False,
        log_fn=logger.info,
        index_file=args.val_index_file,  # 使用验证集的索引文件
        multireso=True,
        batch_size=args.batch_size,
        world_size=1,
    )
    
    # 创建数据加载器
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 加载模型
    model = HFCLIPWrapper.load_from_checkpoint(
        args.checkpoint_path,
        model_name="/mnt/data/clip-vit-large-patch14",
        minibatch_size=args.batch_size
    )
    model.eval()

    # 配置评估器
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=1,
        precision="16-mixed",
    )
    
    # 运行评估
    results = trainer.validate(model, val_dataloader)
    
    # 打印结果
    print("Validation Results:")
    print(f"Image-to-Text Accuracy: {results[0]['val_acc_i']:.4f}")
    print(f"Text-to-Image Accuracy: {results[0]['val_acc_t']:.4f}")
    print(f"Average Accuracy: {results[0]['val_acc']:.4f}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--val_index_file', type=str, required=True,
                      help='Path to validation dataset index file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--accelerator', type=str, default='gpu')
    
    args = parser.parse_args()
    main(args) 