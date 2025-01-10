import pickle
import random
from pathlib import Path
import ast
import numpy as np
import re
import json
import time
from functools import partial
from PIL import Image
import random
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset

from IndexKits.index_kits import ArrowIndexV2, MultiResolutionBucketIndexV2, MultiIndexV2
from . import eugebooru_clip as eugebooru

class TextImageArrowStream(Dataset):
    def __init__(self,
                 args="",
                 image_size=224,
                 resize_ratio=0.75,
                 resolution=512,
                 random_flip=None,
                 enable_CN=True,
                 log_fn=print,
                 index_file=None,
                 multireso=False,
                 batch_size=-1,
                 world_size=1,
                 random_shrink_size_cond=False,
                 merge_src_cond=False,
                 uncond_p=0.0,
                 uncond_p_t5=0.0,
                 rank=0,
                 return_raw_image=False,
                 dtype=torch.float32,
                 **kwarges
                 ):
        self.args = args
        self.resolution = resolution
        self.log_fn = lambda x: log_fn(f"    {Path(__file__).stem} | " + x)
        self.return_raw_image = return_raw_image
        self.random_flip = random_flip
        # If true, the Chinese prompt from the `text_zh` column will be taken from the arrow file;
        # otherwise, the English prompt from the `text_en` column will be taken,
        # provided that `text_zh` or `text_en` exists in the arrow file.
        self.enable_CN = enable_CN
        self.index_file = index_file
        self.multireso = multireso
        self.batch_size = batch_size
        self.world_size = world_size
        self.index_manager = self.load_index()
        self.image_size = image_size
        self.resize_ratio = resize_ratio
        # size condition
        self.random_shrink_size_cond = random_shrink_size_cond
        self.merge_src_cond = merge_src_cond

        assert isinstance(resolution, int), f"resolution must be an integer, got {resolution}"
        self.flip_norm = T.Compose(
            [
                T.RandomHorizontalFlip() if self.random_flip else T.Lambda(lambda x: x),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )
        
        self.image_transform = T.Compose([
            T.RandomHorizontalFlip() if self.random_flip else T.Lambda(lambda x: x),
            T.RandomResizedCrop(self.image_size,
                                scale=(self.resize_ratio, 1.),
                                ratio=(1., 1.)),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            
        ])


        self.dataset_hook = {
            "tag_counter": {
                "epoch": {
                    "artist": {},
                    "character": {},
                },
            }
        }
        
        # show info
        if self.merge_src_cond:
            self.log_fn("Enable merging src condition: (oriW, oriH) --> ((WH)**0.5, (WH)**0.5)")

        self.log_fn("Enable image_meta_size condition (original_size, target_size, crop_coords)")
        self.log_fn(f"Image_transforms: {self.flip_norm}")
        
    def fix_img(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img
    
    
    def fix_img(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)
    
    
    def load_index(self):
        multireso = self.multireso
        index_file = self.index_file
        batch_size = self.batch_size
        world_size = self.world_size

        if multireso:
            if isinstance(index_file, (list, tuple)):
                if len(index_file) > 1:
                    raise ValueError(f"When enabling multireso, index_file should be a single file, but got {index_file}")
                index_file = index_file[0]
            index_manager = MultiResolutionBucketIndexV2(index_file, batch_size, world_size)
            self.log_fn(f"Using MultiResolutionBucketIndexV2: {len(index_manager):,}")
        else:
            if isinstance(index_file, str):
                index_file = [index_file]
            if len(index_file) == 1:
                index_manager = ArrowIndexV2(index_file[0])
                self.log_fn(f"Using ArrowIndexV2: {len(index_manager):,}")
            else:
                index_manager = MultiIndexV2(index_file)
                self.log_fn(f"Using MultiIndexV2: {len(index_manager):,}")

        return index_manager

    def shuffle(self, seed, fast=False):
        self.index_manager.shuffle(seed, fast=fast)

    def get_raw_image(self, index, image_key="image"):
        try:
            ret = self.index_manager.get_image(index, image_key)
        except Exception as e:
            self.log_fn(f'get_raw_image | Error: {e}')
            ret = Image.new("RGB", (256, 256), (255, 255, 255))
        return ret

    @staticmethod
    def random_crop_image(image, origin_size, target_size):
        aspect_ratio = float(origin_size[0]) / float(origin_size[1])
        if origin_size[0] < origin_size[1]:
            new_width = target_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = target_size[1]
            new_width = int(new_height * aspect_ratio)

        image = image.resize((new_width, new_height), Image.LANCZOS)

        if new_width > target_size[0]:
            x_start = random.randint(0, new_width - target_size[0])
            y_start = 0
        else:
            x_start = 0
            y_start = random.randint(0, new_height - target_size[1])
        image_crop = image.crop((x_start, y_start, x_start + target_size[0], y_start + target_size[1]))
        crops_coords_top_left = (x_start, y_start)
        return image_crop, crops_coords_top_left

    def get_style(self, index):
        "Here we use a default learned embedder layer for future extension."
        style = 0
        return style

    def get_image_with_hwxy(self, index, image_key="image", return_raw_image=False):

        image = self.get_raw_image(index, image_key=image_key)
        origin_size = image.size

        kwargs = {
            
            # 'origin_size': tuple(origin_size),
            # 'target_size': tuple(target_size),
            # 'crops_coords_top_left': tuple(crops_coords_top_left)
        }
        if return_raw_image:
            return image, {}
        if self.multireso:
            image_tensor = self.image_transform(image)
        else:
            image_tensor = self.image_transform(image_crop)


        

        style = self.get_style(index)
        kwargs['style'] = style

        return image_tensor, kwargs


    def get_tags(
        self,
        ind,
    ):  
        try:
            meta_info = self.index_manager.get_attribute(ind, 'meta_info')

            if len(meta_info) > 1:

                if random.random() < 0.5:
                    try:
                        return eugebooru.get_ata_caption(meta_info, self.dataset_hook)
                    except Exception as e:
                        print(f"Error retrieving tags: {e}, use original text")
                        return self.get_original_text(ind)
                else:
                    return self.get_original_text(ind)

                
            return self.get_original_text(ind)
        except Exception as e:
            return self.get_original_text(ind)




    def get_original_text(self, ind):
        try:
            text = self.index_manager.get_attribute(ind, 'text_zh' if self.enable_CN else 'text_en')
            if text:
                text = text.replace("|||","")
                text = text.replace(", general, sensitive, questionable, explicit","")
                text = text.strip()
            else:   
                meta_info = self.index_manager.get_attribute(ind, 'meta_info')
                if meta_info:
                    if "caption_base" in meta_info:
                        text = meta_info["caption_base"].replace("|||","")
                        text = text.strip()
                    else:
                        text = ""
                else:
                    text = ""
        except Exception as e:
            
            text = ""
        if text == '':
            print(f"get_empty_text | text: {text}")
        return text

    

    def get_original_text_old(self, ind):
        text = self.index_manager.get_attribute(ind, 'text_zh' if self.enable_CN else 'text_en')
        text = str(text).strip()
        return text

    def get_text(self, ind):
        text =  self.get_original_text(ind)
        if text == '':
            text = 'Generate a random image'
        # print(f"get_text | text: {text}")
        return text

    def __getitem__(self, ind):
        # Get text
        # text = self.get_text(ind)
        text = self.get_tags(ind)

        original_pil_image, kwargs = self.get_image_with_hwxy(ind, return_raw_image=self.return_raw_image)
        pixel = original_pil_image  
        # text_token = clip.tokenize(text)[0]
        
        if random.random() < 0.00001:    

            print(f"prompts: {text}")

        return {
            "image": pixel,
            "prompt": text
        }


    def __len__(self):
        return len(self.index_manager)
