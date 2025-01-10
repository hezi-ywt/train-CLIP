import torch
import torch.nn.functional as F
from models.hf_wrapper import HFCLIPWrapper
from torch.utils.data import Dataset, DataLoader
import logging
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
from pathlib import Path
from torchvision import transforms
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FolderDataset(Dataset):
    def __init__(self, folder_path, image_size=224):
        self.folder_path = Path(folder_path)
        self.image_size = image_size
        
        # 定义图像转换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                               (0.26862954, 0.26130258, 0.27577711))
        ])
        
        # 获取所有图像文件
        self.image_files = []
        self.text_files = []
        
        # 支持多种图像格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
        
        for ext in image_extensions:
            for img_path in self.folder_path.glob(ext):
                txt_path = img_path.with_suffix('.txt')
                if txt_path.exists():
                    self.image_files.append(img_path)
                    self.text_files.append(txt_path)
        
        if len(self.image_files) == 0:
            logger.warning(f"No image-text pairs found in {folder_path}")
            logger.info(f"Supported image formats: {', '.join(image_extensions)}")
            logger.info("Files in directory:")
            for f in self.folder_path.iterdir():
                logger.info(f"  {f.name}")
        else:
            logger.info(f"Found {len(self.image_files)} image-text pairs")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 读取图像
        image_path = self.image_files[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            # 转换图像为tensor
            image = self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
        
        # 读取文本
        text_path = self.text_files[idx]
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except Exception as e:
            logger.error(f"Error loading text {text_path}: {e}")
            raise
            
        return {
            'image': image,  # 现在是tensor
            'prompt': text,
            'image_path': str(image_path),
            'text_path': str(text_path)
        }

def compute_metrics(image_features, text_features, logit_scale):
    # 计算相似度矩阵
    logits = (logit_scale * image_features @ text_features.t()).cpu()
    
    # 获取排名
    ground_truth = torch.arange(len(logits))
    rankings = torch.argsort(logits, dim=-1, descending=True)
    
    # 计算指标
    metrics = {}
    
    # 图像到文本
    i2t_ranks = []
    for i in range(len(rankings)):
        rank = torch.where(rankings[i] == i)[0][0]
        i2t_ranks.append(rank.item())
    
    # 文本到图像
    t2i_ranks = []
    rankings = torch.argsort(logits.t(), dim=-1, descending=True)
    for i in range(len(rankings)):
        rank = torch.where(rankings[i] == i)[0][0]
        t2i_ranks.append(rank.item())
    
    # 计算R@K
    for k in [1, 5, 10]:
        metrics[f'I2T_R@{k}'] = 100 * len([r for r in i2t_ranks if r < k]) / len(i2t_ranks)
        metrics[f'T2I_R@{k}'] = 100 * len([r for r in t2i_ranks if r < k]) / len(t2i_ranks)
    
    # 计算中位数排名
    metrics['I2T_median_rank'] = np.median(i2t_ranks) + 1
    metrics['T2I_median_rank'] = np.median(t2i_ranks) + 1
    
    return metrics, logits, i2t_ranks, t2i_ranks

def evaluate_checkpoint(checkpoint_path, dataset, batch_size=32, num_workers=4):
    """评估单个检查点的性能"""
    logger.info(f"\nEvaluating checkpoint: {Path(checkpoint_path).name}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 加载模型
    model = HFCLIPWrapper.load_from_checkpoint(
        checkpoint_path,
        model_name="/mnt/data/clip-vit-base-patch32",
        # model_name="/mnt/data/clip-vit-large-patch14",
        minibatch_size=batch_size
    )
    model.cuda()
    model.eval()

    # 收集所有特征
    all_image_features = []
    all_text_features = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            image = batch['image'].cuda()
            text = batch['prompt']
            
            text_inputs = model.processor(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(image.device)
            
            image_features = F.normalize(model.model.get_image_features(image), dim=-1)
            text_features = F.normalize(model.model.get_text_features(**text_inputs), dim=-1)
            
            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
    
    image_features = torch.cat(all_image_features)
    text_features = torch.cat(all_text_features)
    
    metrics, _, _, _ = compute_metrics(
        image_features.cuda(), 
        text_features.cuda(),
        model.model.logit_scale.exp()
    )
    
    # 清理GPU内存
    del model
    torch.cuda.empty_cache()
    
    return metrics

def plot_metrics(results_df, save_path='checkpoint_comparison.png'):
    """绘制性能对比图"""
    plt.figure(figsize=(15, 10))
    
    # 绘制R@1指标
    plt.subplot(2, 1, 1)
    plt.plot(results_df.index, results_df['I2T_R@1'], 'b-', label='Image-to-Text R@1')
    plt.plot(results_df.index, results_df['T2I_R@1'], 'r-', label='Text-to-Image R@1')
    plt.title('R@1 Performance Comparison')
    plt.xlabel('Checkpoint')
    plt.ylabel('Recall@1 (%)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # 绘制中位数排名
    plt.subplot(2, 1, 2)
    plt.plot(results_df.index, results_df['I2T_median_rank'], 'b--', label='Image-to-Text Median Rank')
    plt.plot(results_df.index, results_df['T2I_median_rank'], 'r--', label='Text-to-Image Median Rank')
    plt.title('Median Rank Comparison')
    plt.xlabel('Checkpoint')
    plt.ylabel('Median Rank')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main(args):
    # 创建输出目录
    output_dir = Path(args.output_dir)
    logger.info(f"Creating output directory: {output_dir.absolute()}")  # 打印完整路径
    
    try:
        output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Successfully created output directory")
    except Exception as e:
        logger.error(f"Error creating output directory: {e}")
        raise
    
    # 创建数据集
    dataset = FolderDataset(
        folder_path=args.folder_path,
        image_size=224
    )
    
    # 获取所有检查点文件
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob('*.ckpt'))
    
    if not checkpoints:
        logger.error(f"No checkpoints found in {checkpoint_dir}")
        return
        
    logger.info(f"Found {len(checkpoints)} checkpoints")
    
    # 评估所有检查点
    results = []
    for ckpt in checkpoints:
        metrics = evaluate_checkpoint(
            ckpt,
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        metrics['checkpoint'] = ckpt.name
        results.append(metrics)
    
    # 转换为DataFrame并保存
    df = pd.DataFrame(results)
    df.set_index('checkpoint', inplace=True)
    
    # 保存CSV结果
    csv_path = output_dir / 'checkpoint_comparison.csv'
    df.to_csv(csv_path)
    logger.info(f"\nResults saved to {csv_path}")
    
    # 绘制对比图
    plot_path = output_dir / 'checkpoint_comparison.png'
    plot_metrics(df, save_path=plot_path)
    logger.info(f"Performance plot saved to {plot_path}")
    
    # 保存详细结果到文本文件
    txt_path = output_dir / 'evaluation_results.txt'
    with open(txt_path, 'w') as f:
        f.write("Checkpoint Evaluation Results\n")
        f.write("===========================\n\n")
        
        # 打印最佳结果
        f.write("Best Checkpoints:\n")
        for metric in ['I2T_R@1', 'T2I_R@1']:
            best_ckpt = df[metric].idxmax()
            f.write(f"\nBest {metric}:\n")
            f.write(f"Checkpoint: {best_ckpt}\n")
            f.write(f"Score: {df.loc[best_ckpt, metric]:.2f}%\n")
            f.write("\nDetailed metrics for this checkpoint:\n")
            for col in df.columns:
                f.write(f"{col}: {df.loc[best_ckpt, col]:.2f}\n")
            f.write("\n")
        
        # 添加所有检查点的完整结果
        f.write("\nComplete Results for All Checkpoints:\n")
        f.write("===================================\n\n")
        f.write(df.to_string())
    
    logger.info(f"Detailed results saved to {txt_path}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                      help='Directory containing checkpoint files')
    parser.add_argument('--folder_path', type=str, required=True,
                      help='Path to folder containing images and text files')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='eval_results',
                      help='Directory to save evaluation results')
    
    args = parser.parse_args()
    main(args) 