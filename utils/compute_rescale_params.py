import joblib
import numpy as np
from Dataset.dataset import UroflowDataset_v2
from Dataset.data_loader import UrflowDataLoader
from configs.path_configs import DATA_PATH, NO_BINS, EPS
from Transforms.base_transform import LFT, MelSpectrogram_gpu
from Dataset.devices import Device


def compute_rescale_configs_params():
    state_dict = {}
    transform_names = ['lft', 'mel_1d', 'mel_2d']
    for device in Device.get_devices():
        dataset = UroflowDataset_v2(data_path= DATA_PATH, device=device)
        transforms = [
            LFT(no_bins=NO_BINS),
            MelSpectrogram_gpu(dataset.get_device_rate(), enable_2d=False),
            MelSpectrogram_gpu(dataset.get_device_rate(), enable_2d=True)
        ]
        temp = {}
        for i, transform in enumerate(transforms):
            dataloader = UrflowDataLoader(dataset, batch_size=len(dataset), transform=transform, shuffle=False, permutate=False)
            x, y = next(iter(dataloader))

            y_mean = float(y.mean())
            y_std = float(np.maximum(y.std(), EPS))

            if x.ndim == 2:
                x_mean = x.mean(axis=0, keepdims= True)
                x_std = np.maximum(x.std(axis=0, keepdims=True), EPS) 

            elif x.ndim == 3:
                x_mean  = x.mean(axis=(0, 2), keepdims= True)
                x_std =  np.maximum(x.std(axis=(0, 2), keepdims=True), EPS) 
            
            key = transform_names[i]
            temp[key] = {
                'x_mean': x_mean,
                'x_std': x_std,
                'y_mean': y_mean,
                'y_std': y_std
            }

        state_dict[device.value] = temp

    joblib.dump(state_dict, 'normalization_stats_state.joblib')
    return state_dict

if __name__ == '__main__':
    compute_rescale_configs_params()

        
