# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Compute similarity matrix with LanguageBind')
    parser.add_argument('--languagebind_model', type=str, default='LanguageBind_Video_V1.5_FT')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--cache_dir', type=str, default='./cache_dir')

    parser.add_argument('--feint6k_msrvtt', type=str, default='feint6k_msrvtt.csv')
    parser.add_argument('--feint6k_vatex', type=str, default='feint6k_vatex.csv')
    parser.add_argument('--video_path', type=str, default='videos')

    parser.add_argument('--text_batch_size', type=int, default=16)
    parser.add_argument('--video_batch_size', type=int, default=16)
    return parser.parse_args()


def prepare_feint6k(dataset='msrvtt'):
    if dataset == 'msrvtt':
        csv = pd.read_csv(args.feint6k_msrvtt)
    elif dataset == 'vatex':
        csv = pd.read_csv(args.feint6k_vatex)
    else:
        raise ValueError(f'Unknown dataset {dataset}')

    texts = csv['sentence'].tolist()
    _video_ids = csv['video_id'].tolist()
    video_ids = [_video_ids[i*6] for i in range(len(_video_ids) // 6)]
    video_paths = [
        os.path.join(args.video_path, x+'.mp4')
        for x in video_ids]
    return video_paths, texts


@torch.no_grad()
def main(args):
    device = torch.device(args.device)
    clip_type = {'video': args.languagebind_model}

    model = LanguageBind(clip_type=clip_type, cache_dir=args.cache_dir)
    model = model.to(device)
    model.eval()

    pretrained_ckpt = f'LanguageBind/{args.languagebind_model}'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir=os.path.join(args.cache_dir, 'tokenizer_cache_dir'))
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

    for dataset in ['msrvtt', 'vatex']:
        videos, texts = prepare_feint6k(dataset)

        # get text embeds
        text_embed_list = []
        nbatches = math.ceil(len(texts) / args.text_batch_size)
        for i in tqdm(range(nbatches), desc='text feat'):
            if i + 1 == nbatches:
                _texts = texts[i * args.text_batch_size : len(texts)]
            else:
                _texts = texts[i * args.text_batch_size : (i + 1) * args.text_batch_size]
            inputs = {'language': to_device(tokenizer(_texts, max_length=77, padding='max_length', truncation=True, return_tensors='pt'), device)}
            embeddings = model(inputs)
            text_embed_list.append(embeddings['language'].detach().cpu())
        text_embed = torch.cat(text_embed_list, dim=0)

        # get video embeds
        video_embed_list = []
        nbatches = math.ceil(len(texts) / args.video_batch_size)
        for i in tqdm(range(nbatches), desc='video feat'):
            if i + 1 == nbatches:
                _videos = videos[i * args.video_batch_size : len(videos)]
            else:
                _video = videos[i * args.video_batch_size : (i + 1) * args.video_batch_size]
            inputs = {'video': to_device(modality_transform['video'](_videos), device)}
            embeddings = model(inputs)
            video_embed_list.append(embeddings['video'].detach().cpu())
        video_embed = torch.cat(video_embed_list, dim=0)

        sim_mat = torch.softmax(text_embed @ video_embed.T, dim=-1).detach().cpu().numpy()
        np.save(f'sim_mat_{dataset}.npy', sim_mat)
        print(f'sim mat of LanguageBind from {dataset} saved to sim_mat_{dataset}.npy')


if __name__ == '__main__':
    args = parse_args()
    main(args)
