import torch
import torch.nn.functional as F
from models.hf_wrapper import HFCLIPWrapper
from data_loader.arrow_load_stream_for_clip import TextImageArrowStream
from torch.utils.data import DataLoader
import logging
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    return metrics

def main(args):
    # 创建数据集
    dataset = TextImageArrowStream(
        args=args,
        image_size=224,
        resize_ratio=0.8,
        resolution=1024,
        random_flip=False,
        log_fn=logger.info,
        index_file=args.index_file,
        multireso=True,
        batch_size=args.batch_size,
        world_size=1,
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
            
            if args.max_samples and len(all_image_features) * args.batch_size >= args.max_samples:
                break
    
    # 合并特征
    image_features = torch.cat(all_image_features)
    text_features = torch.cat(all_text_features)
    
    # 计算指标
    metrics = compute_metrics(
        image_features.cuda(), 
        text_features.cuda(),
        model.model.logit_scale.exp()
    )
    
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
    parser.add_argument('--index_file', type=str, required=True,
                      help='Path to dataset index file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples to evaluate')
    
    args = parser.parse_args()
    main(args) 