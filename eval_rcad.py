# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Compute RCAD metrics')
    parser.add_argument('--msrvtt_sim_mat', type=str, default='sim_mat_msrvtt.npy')
    parser.add_argument('--vatex_sim_mat', type=str, default='sim_mat_vatex.npy')
    return parser.parse_args()


def print_metrics(dataset, sim_mat):
    rank = []
    for qid in range(sim_mat.shape[0]//6):
        mat = sim_mat[qid*6:qid*6+6, qid]
        matl = list(-mat)
        rank.append(sorted(matl).index(matl[0])+1)
    rank = np.array(rank)

    results = {
        'R@1': np.mean(rank <= 1)*100,
        'R@3': np.mean(rank <= 3)*100,
        'meanR': np.mean(rank),
        'medianR': np.median(rank)}

    print(f'RCAD on {dataset}: R@1={results["R@1"]:.1f} R@3={results["R@3"]:.1f} meanR={results["meanR"]:.1f} medianR={results["medianR"]:.1f}')

    return results


def main(args):
    # msrvtt
    sim_mat = np.load(args.msrvtt_sim_mat)
    print_metrics('msrvtt', sim_mat)

    # vatex
    sim_mat = np.load(args.vatex_sim_mat)
    print_metrics('vatex', sim_mat)


if __name__ == '__main__':
    args = parse_args()
    main(args)
