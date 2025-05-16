import torch.nn as nn
import torch
from PIL import Image
import torch.nn as nn
import torch
import requests
from PIL import Image
import os
from pathlib import Path
import numpy as np
from PIL import Image
import glob
import pickle
from flamingo_mini import FlamingoConfig, FlamingoModel, FlamingoProcessor




def load_model(model_path):

    tokenizer = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = FlamingoModel.from_pretrained(model_path ) # , torch_dtype=torch.float16
    model.to(device)
    processor = FlamingoProcessor(model.config)
    tokenizer = processor.tokenizer

    def raw_imgtext_to_input(raw_img_list, text_list, device):
        raw_img_list = raw_img_list[0]
        inputs = processor(images=raw_img_list, device=device)
        inputs['pixel_values'] = inputs['pixel_values']#.to(torch.float16)
        inputs['pixel_values'] = inputs['pixel_values'].detach()
        inputs["pixel_values"].requires_grad = True
        return inputs
    
    return model, raw_imgtext_to_input, tokenizer, processor



def glob_images(image_folder):
    image_extensions = ('*.jpg', '*.JPEG', '*.png')
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_folder, extension)))
    return sorted(image_paths)



def load_dataset(raw_imgtext_to_input, config):
    all_images = glob_images(image_folder=config.data_path)
    text = ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # _, raw_imgtext_to_input, _ , _ = load_model(config.model_type)

    dataset = []
    for i in range(len(all_images)):
        image_path = all_images[i]
        image = Image.open(image_path).convert("RGB")
        input = raw_imgtext_to_input([image], [text], device)
        dataset.append((input, image_path))
    
    return dataset


