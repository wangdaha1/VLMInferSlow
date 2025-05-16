import torch.nn as nn
import torch
import requests
from PIL import Image
import os
from pathlib import Path
import logging
import argparse
import time
import pynvml
import numpy as np
from PIL import Image
import glob
import pickle
from attacks import *
from utils import load_model, load_dataset



def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('VLMInferSlow')
    parser.add_argument('--seed', type=int, default=256, help='random seed')
    parser.add_argument('--avg_times', type=int, default=1)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--model_type', type=str, default='Flamingo')

    parser.add_argument('--data_path', type=str, default='./images')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--root_path', type=str, default='./results')
    
    parser.add_argument('--iter', type=int, default=500)
    parser.add_argument('--p', type=str, choices=['2', 'inf'], default='2')
    parser.add_argument('--restrict', type=str, default='yes', choices=['yes', 'no'])
    parser.add_argument('--epsilon', type=float, default=64)
    parser.add_argument('--fd_eta', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5.)
    parser.add_argument('--q', type=int, default=5)
    parser.add_argument('--eos_coef', type=float, default=0.5)
    parser.add_argument('--eos_factor', type=float, default=0.1)
    parser.add_argument('--kl_coef', type=float, default=0.1)
    parser.add_argument('--kl_k', type=int, default=100)
    
    return parser.parse_args()




def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    def log_string(str):
        logger.info(str)
        print(str)

    args = parse_args()
    



    '''CREATE DIR'''
    log_dir = Path(os.path.join(args.root_path, 'log', args.model_type))
    log_dir.mkdir(parents=True, exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("OPT")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    args.save_path = log_dir


    '''Load Model'''
    model, raw_imgtext_to_input, tokenizer, processor = load_model(args.model_path)
    '''Load Image'''    
    dataset = load_dataset(raw_imgtext_to_input, args)

    '''VLMInferSlow ttack'''
    attacker = VLMInferSlowAttack(model, tokenizer, processor, device, logger, config=args)

    print("Note that if you just try one image, the latency and energy of the original image will be inaccurate")
    logger.info("Note that if you just try one image, the latency and energy of the original image will be inaccurate")

    for i in range(len(dataset)):
        x, image_path = dataset[i]
        logger.info(f"processing {i+1}th image: {image_path}")

        res = attacker.run_attack(x)

if __name__ == '__main__':
    args = parse_args()

    main(args)

