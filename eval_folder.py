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

def save_retrieval_results(logits, dataset, save_dir):
    """保存检索结果"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # 对每个查询保存top-k结果
    k = 5  # 每个查询保存前5个结果
    
    # 图像到文本
    i2t_dir = save_dir / 'image_to_text'
    i2t_dir.mkdir(exist_ok=True)
    
    rankings = torch.argsort(logits, dim=-1, descending=True)
    for i in range(len(rankings)):
        query_image = dataset.image_files[i].name
        top_k_texts = [dataset.text_files[idx].read_text() for idx in rankings[i][:k]]
        
        with open(i2t_dir / f"{query_image}.txt", 'w', encoding='utf-8') as f:
            f.write(f"Query Image: {query_image}\n\nTop {k} Retrieved Texts:\n")
            for rank, text in enumerate(top_k_texts, 1):
                f.write(f"\n{rank}. {text}\n")
    
    # 文本到图像
    t2i_dir = save_dir / 'text_to_image'
    t2i_dir.mkdir(exist_ok=True)
    
    rankings = torch.argsort(logits.t(), dim=-1, descending=True)
    for i in range(len(rankings)):
        query_text = dataset.text_files[i].name
        with open(t2i_dir / f"{query_text}", 'w', encoding='utf-8') as f:
            f.write(f"Query Text: {dataset.text_files[i].read_text()}\n\nTop {k} Retrieved Images:\n")
            for rank, idx in enumerate(rankings[i][:k], 1):
                f.write(f"\n{rank}. {dataset.image_files[idx].name}\n")

def main(args):
    # 创建数据集
    dataset = FolderDataset(
        folder_path=args.folder_path,
        image_size=224
    )
    
    dataloader = DataLoader(
        dataset,
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
    model.cuda()
    model.eval()

    # 收集所有特征
    all_image_features = []
    all_text_features = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            image = batch['image'].cuda()
            text = batch['prompt']
            
            # 处理文本
            text_inputs = model.processor(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(image.device)
            
            # 提取特征
            image_features = F.normalize(model.model.get_image_features(image), dim=-1)
            text_features = F.normalize(model.model.get_text_features(**text_inputs), dim=-1)
            
            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
    
    # 合并特征
    image_features = torch.cat(all_image_features)
    text_features = torch.cat(all_text_features)
    
    # 计算指标
    metrics, logits, i2t_ranks, t2i_ranks = compute_metrics(
        image_features.cuda(), 
        text_features.cuda(),
        model.model.logit_scale.exp()
    )
    
    # 保存检索结果
    if args.save_results:
        save_retrieval_results(logits, dataset, args.save_dir)
    
    # 打印结果
    print("\nRetrieval Results:")
    print("\nImage-to-Text:")
    print(f"R@1: {metrics['I2T_R@1']:.2f}%")
    print(f"R@5: {metrics['I2T_R@5']:.2f}%")
    print(f"R@10: {metrics['I2T_R@10']:.2f}%")
    print(f"Median Rank: {metrics['I2T_median_rank']:.1f}")
    
    print("\nText-to-Image:")
    print(f"R@1: {metrics['T2I_R@1']:.2f}%")
    print(f"R@5: {metrics['T2I_R@5']:.2f}%")
    print(f"R@10: {metrics['T2I_R@10']:.2f}%")
    print(f"Median Rank: {metrics['T2I_median_rank']:.1f}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--folder_path', type=str, required=True,
                      help='Path to folder containing images and text files')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_results', action='store_true',
                      help='Save retrieval results to files')
    parser.add_argument('--save_dir', type=str, default='retrieval_results',
                      help='Directory to save retrieval results')
    
    args = parser.parse_args()
    main(args) 