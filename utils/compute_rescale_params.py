import joblib
from Dataset.dataset import UroflowDataset_v2
from Dataset.data_loader import UrflowDataLoader
from configs.path_configs import DATA_PATH, NO_BINS
from Transforms.base_transform import LFT, MelSpectrogram
from Dataset.devices import Device


def compute_rescale_configs_params():
    state_dict = {}
    transform_names = ['lft', 'mel_1d', 'mel_2d']
    for device in Device.get_devices():
        dataset = UroflowDataset_v2(data_path= DATA_PATH, device=device)
        transforms = [
            LFT(no_bins=NO_BINS),
            MelSpectrogram(dataset.get_device_rate(), enable_2d=False),
            MelSpectrogram(dataset.get_device_rate(), enable_2d=True)
        ]
        temp = {}
        for i, transform in enumerate(transforms):
            dataloader = UrflowDataLoader(dataset, batch_size=len(dataset), transform=transform, shuffle=False, permutate=False)
            x, y = next(iter(dataloader))
            y_mean, y_std = y.mean(), y.std()
            if i < 2:
                x_mean, x_std = x.mean(), x.std()
            else:
                x_mean, x_std = x.mean((0, 2), keepdims= True), x.std((0, 2), keepdims=True)
            
            key = transform_names[i]

            temp[key] = {'x_mean': None, 'x_std': None, 'y_mean': None, 'y_std': None}
            temp[key]['x_mean'], temp[key]['x_std'] = x_mean, x_std
            temp[key]['y_mean'], temp[key]['y_std']  = y_mean, y_std

        state_dict[device.value] = temp

    joblib.save(state_dict, 'normalization_stats_state.joblib')
    return state_dict

if __name__ == '__main__':
    compute_rescale_configs_params()

            
        


            
            










def compute_rescale_configs():
    dataset = UroflowDataset_v2(DATA_PATH, device=Device.UM)
    # transform = LFT(no_bins=20)
    transform = MelSpectrogram(sr=dataset.get_device_rate(), enable_2d=True)

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
