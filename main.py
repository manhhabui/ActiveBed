import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from algorithms.Rand.src.Trainer_Rand import Trainer_Rand
from algorithms.LLAL.src.Trainer_LLAL import Trainer_LLAL
from algorithms.RandOD.src.Trainer_RandOD import Trainer_RandOD
from algorithms.LLALOD.src.Trainer_LLALOD import Trainer_LLALOD

def fix_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


algorithms_map = {"Rand": Trainer_Rand, "RandOD": Trainer_RandOD, "LLAL": Trainer_LLAL, "LLALOD": Trainer_LLALOD}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--exp_idx", help="Index of experiment")
    bash_args = parser.parse_args()
    with open(bash_args.config, "r") as inp:
        args = argparse.Namespace(**json.load(inp))

    # fix_random_seed(args.seed_value)
    logging.basicConfig(
        filename="algorithms/" + args.algorithm + "/results/logs/" + args.exp_name + "_" + bash_args.exp_idx + ".log",
        filemode="w",
        level=logging.INFO,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = algorithms_map[args.algorithm](args, device, bash_args.exp_idx)
    trainer.train()
    print("Finished!")