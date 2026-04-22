import random
import numpy as np
from abc import abstractmethod
from Dataset.dataset import Dataset
from Transforms.base_transform import Transform

class DataLoader:
    """Composes a batched generator around the dataset"""
    def __init__(self, dataset: Dataset, batch_size:int, shuffle=True, permutate= False):
        self.dataset = dataset
        self.bach_size = batch_size
        self.shuffle = shuffle
        self.permutate = permutate
    
    @abstractmethod
    def __iter__(self, dataset: Dataset):
        raise NotImplementedError('Subclasses must implement this method')
    

class UrflowDataLoader(DataLoader):

    def __init__(self, dataset: Dataset, batch_size:int , transform: Transform, shuffle: bool=False, permutate= False):
        super().__init__(dataset, batch_size, shuffle, permutate)
        self.transfrom = transform


    def __iter__(self):
        """Implements an iter around the Dataset obj"""
        size = len(self.dataset)
        indices = list(range(size))

        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, size, self.bach_size):
            sampled_indices = indices[i : i+ self.bach_size]
            x_buffer = []
            y_buffer = []

            for index in sampled_indices:
                x, y  = self.dataset[index]
                x_transformed = self.transfrom.fit_transform(x)
                x_buffer.append(x_transformed)
                y_buffer.append(y)

            yield self._make_contiguous(x_buffer, y_buffer, self.permutate)

    def _make_contiguous(self, features: list, targets: list, permutate):
        if features[0].ndim >= 2 and len({x.shape[-1] for x in features}) > 1:
            max_len = max(x.shape[-1] for x in features)
            features = [
                np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, max_len - x.shape[-1])])
                for x in features
            ]
        x_batched = np.vstack(features)
        y_batched = np.concatenate(targets)

    
        if permutate:
            perm = np.random.permutation(len(x_batched))
            
            x_batched = x_batched[perm]
            y_batched = y_batched[perm]

        return x_batched, y_batched
    

if __name__ == '__main__':
    import os
    from Dataset.dataset import UroflowDataset_v2
    from configs.path_configs import DATA_PATH
    from Dataset.data_loader import UrflowDataLoader
    from Dataset.devices import Device
    from Transforms.base_transform import LFT, MelSpectrogram
    
    os.system('clear')

    dataset = UroflowDataset_v2(data_path=DATA_PATH, device=Device.UM)
    
    lft = LFT(no_bins=20)
    mel_spec = MelSpectrogram(sr=dataset.get_device_rate(), enable_2d=False)
    dataloader = UrflowDataLoader(dataset, batch_size=4, transform=mel_spec, permutate=True)

    data_iter = iter(dataloader)
    x, y = next(data_iter)

    print(f'Shape of feautres: {x.shape}', f'Shape of Target: {y.shape}', sep=' | ')




