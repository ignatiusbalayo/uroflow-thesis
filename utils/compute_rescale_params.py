from Dataset.dataset import UroflowDataset_v2
from Dataset.data_loader import UrflowDataLoader
from configs.path_configs import DATA_PATH
from Transforms.base_transform import LFT, MelSpectrogram
from Dataset.devices import Device




def compute_rescale_configs():
    dataset = UroflowDataset_v2(DATA_PATH, device=Device.UM)
    # transform = LFT(no_bins=20)
    transform = MelSpectrogram(sr=dataset.get_device_rate())

    dataloader = UrflowDataLoader(dataset, len(dataset), transform, shuffle=False, permutate=False)
    x, y = next(iter(dataloader))
    x_mean, x_std = x.mean(), x.std()
    y_mean, y_std = y.mean(), y.std()

    print(f'X_mean: {x_mean}', f'X_std: {x_std}', f'Y mean: {y_mean}', f'Y std: {y_std}', sep=' | ')

    x_rescaled = (x - x_mean) / x_std
    y_rescaled = (y - y_mean) / y_std

    print(f'X_rescaled mean: {x_rescaled.mean()}', f'X_reshaped std: {x_rescaled.std()}', f'y_rescaled mean: {y_rescaled.mean()}', f'y_rescaled std: {y_rescaled.std()}', sep=' | ')



if __name__ == '__main__':
    import os
    os.system('clear')
    compute_rescale_configs()
