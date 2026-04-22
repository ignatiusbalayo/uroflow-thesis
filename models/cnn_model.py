import torch
import torch.nn as nn

class UroflowCNN(nn.Module):

    def __init__(self, input_channels=20, input_dim=601):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_dim)
            out = self.features(dummy)
            feature_dim = out.shape[1]
        
        self.regressor = nn.Linear(feature_dim, 1)


    def forward(self, x):
        out = self.features(x)
        out = out.squeeze(-1)
        return self.regressor(out)



if __name__ == '__main__':
    import os
    from configs.path_configs import DATA_PATH
    from Dataset.devices import Device
    from Dataset.dataset import UroflowDataset_v2
    from Dataset.data_loader import UrflowDataLoader
    from Transforms.base_transform import MelSpectrogram_gpu
    from utils.read_normalization_params import read_norm_data, Transform_keys

    os.system('clear')

    dataset = UroflowDataset_v2(DATA_PATH, Device.UM)
    mel_2d = MelSpectrogram_gpu(sr=dataset.get_device_rate(), enable_2d=True)

    dataloader  = UrflowDataLoader(dataset=dataset, batch_size=4, transform=mel_2d, shuffle=True, permutate=False)
    x_mean, x_std, y_mean, y_std = read_norm_data(Device.UM, Transform_keys.Mel_2d)
    cnn_model = UroflowCNN()

    for x, y in dataloader:
        x = (x - x_mean) / x_std
        y = (y - y_mean) / y_std
        x = torch.tensor(x, dtype=torch.float32)
        
        out = cnn_model(x)
        print(out, y)
        break




        