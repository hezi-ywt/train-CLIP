from transformers import CLIPProcessor, CLIPModel
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import math
import copy

class HFCLIPWrapper(pl.LightningModule):
    def __init__(self,
                 model_name: str,  # 例如 "openai/clip-vit-base-patch32"
                 minibatch_size: int
                 ):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.minibatch_size = minibatch_size
        self.model_name = model_name
        
        # 禁用自动优化以便手动控制
        self.automatic_optimization = False

    def training_step(self, train_batch, idx):
        # get optimizers and scheduler
        optimizer = self.optimizers()

        image = train_batch['image']
        text = train_batch['prompt']  # 文本列表

        # 使用processor处理文本
        text_inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(image.device)

        n = math.ceil(len(image) // self.minibatch_size)
        image_mbs = torch.chunk(image, n)
        
        # 处理文本batch
        text_features = []
        for i in range(n):
            start_idx = i * self.minibatch_size
            end_idx = min((i + 1) * self.minibatch_size, len(text))
            text_batch = {k: v[start_idx:end_idx] for k, v in text_inputs.items()}
            text_features.append(self.model.get_text_features(**text_batch))
        
        # 计算原始统计数据
        with torch.no_grad():
            ims = [F.normalize(self.model.get_image_features(im), dim=1) for im in image_mbs]
            txt = [F.normalize(feat, dim=1) for feat in text_features]
            
            # gather from all GPUs
            ims = self.all_gather(torch.cat(ims))
            txt = self.all_gather(torch.cat(txt))

            if len(ims.shape) == 3:
                ims = list(ims)
                txt = list(txt)
            else:
                ims = [ims]
                txt = [txt]

        # image loss
        for j, mb in enumerate(image_mbs):
            images_tmp = copy.deepcopy(ims)
            images_tmp[self.global_rank][j*self.minibatch_size:(j+1)*self.minibatch_size] = F.normalize(self.model.get_image_features(mb), dim=1)
            image_logits = torch.cat(images_tmp) @ torch.cat(txt).t() * self.model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth))/2
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            self.log('train_loss', loss, prog_bar=True)
            self.log('logit_scale', self.model.logit_scale.exp())
            
            # 计算准确率
            acc_i = (image_logits.argmax(dim=1) == ground_truth).float().mean()
            acc_t = (image_logits.argmax(dim=0) == ground_truth).float().mean()
            self.log('train_acc_i', acc_i, prog_bar=True)
            self.log('train_acc_t', acc_t, prog_bar=True)
            
            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=5e-5,  # 使用较小的学习率
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.2
        )
        
        return optimizer

    def forward(self, images, text):
        # 处理文本输入
        text_inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(images.device)
        
        # 获取特征
        image_features = self.model.get_image_features(images)
        text_features = self.model.get_text_features(**text_inputs)
        
        # 归一化
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        
        # 计算相似度
        logits = image_features @ text_features.t() * self.model.logit_scale.exp()
        return logits, logits.t() 

    def validation_step(self, val_batch, idx):
        image = val_batch['image']
        text = val_batch['prompt']

        # 处理文本
        text_inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(image.device)

        # 获取特征
        image_features = F.normalize(self.model.get_image_features(image), dim=1)
        text_features = F.normalize(self.model.get_text_features(**text_inputs), dim=1)

        # 计算相似度
        logits = image_features @ text_features.t() * self.model.logit_scale.exp()
        
        # 计算准确率
        ground_truth = torch.arange(len(logits)).to(logits.device)
        acc_i = (logits.argmax(dim=1) == ground_truth).float().mean()
        acc_t = (logits.argmax(dim=0) == ground_truth).float().mean()
        
        self.log('val_acc_i', acc_i)
        self.log('val_acc_t', acc_t)
        self.log('val_acc', (acc_i + acc_t) / 2)
        
        return {'val_acc': (acc_i + acc_t) / 2} 