import torch.utils.data
import torchaudio
import os
from utils import *
import random
from natsort import natsorted

# 允许程序加载多个相同的库到内存中。这通常可以解决因加载了多个OpenMP运行时库而导致的程序崩溃问题。
os.environ["KMP_DUPLICATE_LIB_OK"] = "True" 

from torch.utils.data.distributed import DistributedSampler


class DemandDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, cut_len=16000 * 2, cv=0):
        self.cut_len = cut_len
        self.clean_dir = os.path.join(data_dir, "clean")
        self.noisy_dir = os.path.join(data_dir, "noisy")
        self.clean_wav_name = os.listdir(self.clean_dir)
        self.clean_wav_name = natsorted(self.clean_wav_name) # 按照自然排序
        self.cv = cv

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx]) # 此处用了clean_wav_name，因为clean和noisy的文件名是一样的

        clean_ds, _ = torchaudio.load(clean_file)
        noisy_ds, _ = torchaudio.load(noisy_file)
        clean_ds = clean_ds.squeeze() # 去掉维度为1的维度
        noisy_ds = noisy_ds.squeeze()
        length = len(clean_ds)
        assert length == len(noisy_ds) # 长度一致
        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len % length])
            noisy_ds_final.append(noisy_ds[: self.cut_len % length])
            clean_ds = torch.cat(clean_ds_final, dim=-1) # 沿着最后一个维度拼接起来
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start : wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start : wav_start + self.cut_len]

        return clean_ds, noisy_ds, length


def load_data(ds_dir, batch_size, n_cpu, cut_len):
    # torchaudio.set_audio_backend("sox_io")  # in linux 将音频后端设置为"sox_io"
    train_dir = os.path.join(ds_dir, "train")
    test_dir = os.path.join(ds_dir, "test")

    train_ds = DemandDataset(train_dir, cut_len,cv=0)
    test_ds = DemandDataset(test_dir, cut_len, cv=1)

    train_dataset = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        pin_memory=False, # 将数据加载到CUDA固定（pinned）内存中
        shuffle=False,
        sampler=DistributedSampler(train_ds),
        drop_last=True, # 如果数据集大小不能被batch_size整除，则设置为True可以删除最后一个不完整的batch
        num_workers=n_cpu,
    )
    test_dataset = torch.utils.data.DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=False,
        sampler=DistributedSampler(test_ds),
        drop_last=False, 
        num_workers=n_cpu,
    )

    return train_dataset, test_dataset
