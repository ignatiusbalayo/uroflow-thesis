import torch
import torch.nn as nn

class UroflowMLP(nn.Module):

    def __init__(self, input_dim= 20):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    
    def forward(self, x):
        return self.model(x)
    
if __name__ == '__main__':
    import os
    from configs.path_configs import DATA_PATH
    from Dataset.devices import Device
    from Dataset.dataset import UroflowDataset_v2
    from Dataset.data_loader import UrflowDataLoader
    from Transforms.base_transform import LFT, MelSpectrogram
    from utils.read_normalization_params import read_norm_data, Transform_keys

    os.system('clear')

    dataset = UroflowDataset_v2(DATA_PATH, Device.UM)
    lft = LFT(no_bins=20)
    mel_1d = MelSpectrogram(sr=dataset.get_device_rate())

    dataloader  = UrflowDataLoader(dataset=dataset, batch_size=4, transform=mel_1d, shuffle=True, permutate=False)
    mlp_model = UroflowMLP()

    x_mean, x_std, y_mean, y_std = read_norm_data(Device.UM, Transform_keys.Mel_1d)

    for x, y in dataloader:
        x = (x - x_mean) / x_std
        y = (y - y_mean) / y_std

        x = torch.tensor(x, dtype=torch.float32)
        out = mlp_model(x)
        print(out, y)
        break        
    
    