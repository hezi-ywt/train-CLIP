import torch
import torch.nn.functional as F
from models.hf_wrapper import HFCLIPWrapper
from data_loader.arrow_load_stream_for_clip import TextImageArrowStream
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import glob
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def list_checkpoints(checkpoint_dir):
    """列出检查点目录中的所有检查点文件"""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    print("\nAvailable checkpoints:")
    for i, ckpt in enumerate(checkpoints):
        # 获取文件名和大小
        filename = os.path.basename(ckpt)
        size_mb = os.path.getsize(ckpt) / (1024 * 1024)
        print(f"{i+1}. {filename} ({size_mb:.1f}MB)")
    return checkpoints

def load_model(checkpoint_path, device='cuda'):
    """加载模型检查点"""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\nCheckpoint info:")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Global step: {checkpoint['global_step']}")
    
    if 'metrics' in checkpoint:
        print("\nMetrics at checkpoint:")
        for k, v in checkpoint['metrics'].items():
            print(f"{k}: {v}")
    
    model = HFCLIPWrapper.load_from_checkpoint(
        checkpoint_path,
        model_name="/mnt/data/clip-vit-large-patch14",
        strict=False
    )
    model = model.to(device)
    model.eval()
    
    return model

def evaluate_model(model, val_dataloader, device='cuda'):
    """评估模型性能"""
    print("\nEvaluating model...")
    model.eval()
    
    all_image_features = []
    all_text_features = []
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Extracting features"):
            image = batch['image'].to(device)
            text = batch['prompt']
            
            # 处理文本
            text_inputs = model.processor(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(device)
            
            # 提取特征
            image_features = F.normalize(model.model.get_image_features(image), dim=-1)
            text_features = F.normalize(model.model.get_text_features(**text_inputs), dim=-1)
            
            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
    
    # 计算相似度和指标
    image_features = torch.cat(all_image_features)
    text_features = torch.cat(all_text_features)
    
    similarity = image_features @ text_features.t()
    
    # 计算准确率
    i2t_acc = (similarity.argmax(dim=1) == torch.arange(len(similarity))).float().mean()
    t2i_acc = (similarity.argmax(dim=0) == torch.arange(len(similarity))).float().mean()
    
    return {
        'i2t_accuracy': i2t_acc.item(),
        't2i_accuracy': t2i_acc.item(),
        'avg_accuracy': (i2t_acc.item() + t2i_acc.item()) / 2
    }

def extract_features(model, image_path=None, text=None, device='cuda'):
    """提取图像或文本的特征"""
    model.eval()
    with torch.no_grad():
        features = {}
        
        if image_path:
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            image = model.processor(images=image, return_tensors="pt")["pixel_values"].to(device)
            image_features = model.model.get_image_features(image)
            features['image'] = F.normalize(image_features, dim=-1)
        
        if text:
            # 处理文本
            text_inputs = model.processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(device)
            text_features = model.model.get_text_features(**text_inputs)
            features['text'] = F.normalize(text_features, dim=-1)
            
    return features

def compare_models(model1_path, model2_path, sample_batch, device='cuda'):
    """比较两个模型的输出差异"""
    model1 = load_model(model1_path, device)
    model2 = load_model(model2_path, device)
    
    with torch.no_grad():
        # 获取两个模型的特征
        features1 = extract_features(model1, **sample_batch)
        features2 = extract_features(model2, **sample_batch)
        
        # 计算特征差异
        for key in features1:
            diff = ((features1[key] - features2[key])**2).mean().sqrt()
            print(f"{key} feature difference (L2): {diff:.4f}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.list_only:
        list_checkpoints(args.checkpoint_dir)
        return
    
    # 加载模型
    model = load_model(args.checkpoint_path if args.checkpoint_path else None, device)
    
    # 根据命令执行不同操作
    if args.evaluate:
        # 创建验证数据加载器
        val_dataset = TextImageArrowStream(
            args=args,
            image_size=224,
            index_file=args.val_index_file,
            batch_size=args.batch_size
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
        metrics = evaluate_model(model, val_loader, device)
        print("\nEvaluation Results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    
    elif args.extract_features:
        features = extract_features(
            model,
            image_path=args.image_path,
            text=args.text,
            device=device
        )
        print("\nExtracted Features:")
        for k, v in features.items():
            print(f"{k} shape: {v.shape}")
            print(f"{k} norm: {v.norm().item():.4f}")
    
    elif args.compare_with:
        sample_batch = {
            'image_path': args.image_path,
            'text': args.text
        }
        compare_models(args.checkpoint_path, args.compare_with, sample_batch, device)
    
    return model

if __name__ == "__main__":
    parser = ArgumentParser()
    # 基本参数
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--list_only', action='store_true')
    
    # 评估相关参数
    parser.add_argument('--evaluate', action='store_true',
                      help='Evaluate model performance')
    parser.add_argument('--val_index_file', type=str,
                      help='Validation dataset index file')
    parser.add_argument('--batch_size', type=int, default=32)
    
    # 特征提取相关参数
    parser.add_argument('--extract_features', action='store_true',
                      help='Extract features from image/text')
    parser.add_argument('--image_path', type=str,
                      help='Path to image for feature extraction')
    parser.add_argument('--text', type=str,
                      help='Text for feature extraction')
    
    # 模型比较参数
    parser.add_argument('--compare_with', type=str,
                      help='Path to another checkpoint to compare with')
    
    args = parser.parse_args()
    main(args) 