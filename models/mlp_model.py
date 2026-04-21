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

    os.system('clear')

    dataset = UroflowDataset_v2(DATA_PATH, Device.UM)
    lft = LFT(no_bins=20)

    dataloader  = UrflowDataLoader(dataset=dataset, batch_size=4, transform=lft, shuffle=True, permutate=False)
    mlp_model = UroflowMLP()

    for x, y in dataloader:
        x = torch.tensor(x, dtype=torch.float32)
        out = mlp_model(x)
        print(f'X shape: {x.shape}', f'Y shape: {y.shape}', f'Output model shape: {out.shape}',f'Mean of X: {x.mean()}', f'Std of X: {x.std()}' ,sep=' | ')
        
    
    