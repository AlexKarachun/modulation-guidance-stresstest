import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from torch.utils.data import Dataset

import yaml
from omegaconf import OmegaConf
from yt_tools.utils import instantiate_from_config



def create_dataloader(dataloader_config_path: str, batch_size: int, skip_rows=0):
    with open(dataloader_config_path) as f:
        dataloader_config = OmegaConf.create(yaml.load(f, Loader=yaml.SafeLoader))
    # Set batch size
    dataloader_config["params"]["batch_size"] = batch_size
    return instantiate_from_config(dataloader_config, skip_rows=skip_rows)



def get_loader_prompts_only(args):
    dataset = COCODataset(args.prompts_path)
    dataset_sampler = InfiniteSampler(
        dataset=dataset, rank=dist.get_rank(),
        shuffle=False, num_replicas=dist.get_world_size()
    )
    data = iter(torch.utils.data.DataLoader(
        dataset=dataset, sampler=dataset_sampler, batch_size=args.train_batch_size
    ))
    return data


class COCODataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.captions = df['caption'].astype(str).tolist()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {'prompts': self.captions[idx]}


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1