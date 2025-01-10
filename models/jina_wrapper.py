import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoProcessor
import numpy as np
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

class JinaCLIPWrapper(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 minibatch_size: int,
                 learning_rate: float = 5e-5,
                 warmup_steps: int = 2000,
                 max_steps: int = 100000,
                 truncate_dim: int = 512
                 ):
        super().__init__()
        
        # 加载模型和处理器
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        self.minibatch_size = minibatch_size
        self.truncate_dim = truncate_dim
        
        # 保存超参数
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
        self.automatic_optimization = False

    def training_step(self, train_batch, idx):
        optimizer = self.optimizers()
        
        # 从字典中获取数据
        images = train_batch['image']  # [B, C, H, W] tensor
        texts = train_batch['prompt']  # 文本列表
        
        # 使用processor只处理文本
        text_inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # 将文本输入移到正确的设备上
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        # 确保 minibatch_size 合理
        if self.minibatch_size > len(images):
            self.minibatch_size = len(images)
        
        n = max(1, len(images) // self.minibatch_size)
        actual_minibatch_size = len(images) // n
        
        # 分块
        image_mbs = torch.chunk(images, n)
        input_ids_mbs = torch.chunk(text_inputs['input_ids'], n)
        attention_mask_mbs = torch.chunk(text_inputs['attention_mask'], n)
        
        # 计算特征
        with torch.no_grad():
            ims = []
            txt = []
            
            for im, ids, mask in zip(image_mbs, input_ids_mbs, attention_mask_mbs):
                try:
                    im_feat = F.normalize(self.model.get_image_features(pixel_values=im), dim=1)
                    txt_feat = F.normalize(self.model.get_text_features(
                        input_ids=ids,
                        attention_mask=mask
                    ), dim=1)
                    
                    ims.append(im_feat)
                    txt.append(txt_feat)
                except Exception as e:
                    print(f"Error in feature extraction: {e}")
                    print(f"Image shape: {im.shape}")
                    print(f"Input ids shape: {ids.shape}")
                    print(f"Attention mask shape: {mask.shape}")
                    raise
            
            ims = self.all_gather(torch.cat(ims))
            txt = self.all_gather(torch.cat(txt))
            
            image_logits = torch.cat(ims) @ torch.cat(txt).t() * self.model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)
            loss = (F.cross_entropy(image_logits, ground_truth) + 
                   F.cross_entropy(image_logits.t(), ground_truth)).div(2)
            
            acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
            acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()
            self.log_dict({
                'loss': loss / len(ims), 
                'i2t_acc': acc_i / len(images) / len(ims),
                't2i_acc': acc_t / len(images) / len(ims),
                'avg_acc': (acc_i + acc_t) / 2 / len(images) / len(ims)
            }, prog_bar=True)
        
        optimizer.zero_grad()
        
        # 图像损失
        for j, im in enumerate(image_mbs):
            images_tmp = [im.clone() for im in ims]
            images_tmp[self.global_rank][j*self.minibatch_size:(j+1)*self.minibatch_size] = \
                F.normalize(self.model.get_image_features(pixel_values=im), dim=1)
            
            image_logits = torch.cat(images_tmp) @ torch.cat(txt).t() * self.model.logit_scale.exp()
            loss = (F.cross_entropy(image_logits, ground_truth) + 
                   F.cross_entropy(image_logits.t(), ground_truth))/2
            self.manual_backward(loss)
        
        # 文本损失
        for j, (ids, mask) in enumerate(zip(input_ids_mbs, attention_mask_mbs)):
            text_tmp = [t.clone() for t in txt]
            text_tmp[self.global_rank][j*self.minibatch_size:(j+1)*self.minibatch_size] = \
                F.normalize(self.model.get_text_features(
                    input_ids=ids,
                    attention_mask=mask
                ), dim=1)
            
            image_logits = torch.cat(ims) @ torch.cat(text_tmp).t() * self.model.logit_scale.exp()
            loss = (F.cross_entropy(image_logits, ground_truth) + 
                   F.cross_entropy(image_logits.t(), ground_truth))/2
            self.manual_backward(loss)
        
        optimizer.step()
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        
        # 限制logit scale
        self.model.logit_scale.data.clamp_(-np.log(100), np.log(100))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.2
        )
        
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.max_steps,
            cycle_mult=1.0,
            max_lr=self.learning_rate,
            min_lr=1e-6,
            warmup_steps=self.warmup_steps,
            gamma=0.5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        } 