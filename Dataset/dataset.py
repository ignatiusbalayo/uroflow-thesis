import os
from abc import ABC, abstractmethod
from Dataset.devices import Device
from configs.path_configs import MINZE_RATE, MINZE_READING_INDEX
from scipy.io import wavfile
import pandas as pd
import numpy as np

class Dataset(ABC):

    @abstractmethod
    def  __len__(self):
        """Returns len of the dataset"""
        raise NotImplementedError('Child class must implement this method')
    
    @abstractmethod
    def __getitem__(self):
        """Returns single item"""
        raise NotImplementedError("Child class must implement this method")
    

class UroflowDataset_v2(Dataset):
    """V2 implementation of the uroflow dataset"""

    def __init__(self, data_path: str, device: Device):
        self.data_path = data_path
        self.device = device.value
        self.x_files = []
        self.y_files = []

        self._load_data()
        self.device_rate = None

    def _load_data(self):
        """Reads file paths from root path"""

        for v in os.listdir(self.data_path):
            v_dir = os.path.join(self.data_path, v)
            if not v_dir.endswith('.xlsx'):
                for sub_v in os.listdir(v_dir):
                    sub_v_dir = os.path.join(v_dir, sub_v)
                    if sub_v_dir.endswith('.csv'):
                        self.y_files.append(sub_v_dir)
                    elif self.device in sub_v_dir:
                        self.x_files.append(sub_v_dir)
                    else:
                        continue

    def __len__(self):
        """computes length of object"""

        return len(self.y_files)
    
    def __getitem__(self, index):
        x_file, y_file = self.x_files[index], self.y_files[index]
        device_rate, sound_file = wavfile.read(x_file)

        device_rate_ps = int(MINZE_RATE * device_rate)
        y_reading = pd.read_csv(y_file).to_numpy()[:,MINZE_READING_INDEX].astype(np.float32)

        no_readings_recorded = y_reading.shape[0]
        device_sound_sampled_ps = np.array([sound_file[i * device_rate_ps: (i + 1) * device_rate_ps] for i in range(no_readings_recorded)])
        device_sound_sampled_ps = device_sound_sampled_ps.astype(np.float32)

        self.device_rate = device_rate

        return device_sound_sampled_ps, y_reading


if __name__ == '__main__':
    from configs.path_configs import DATA_PATH
    
    for device in Device.get_devices():
        dataset = UroflowDataset_v2(DATA_PATH, device)
        x, y = dataset[0]
        print(f'Device: {device.value}',  f'Feature shape: {x.shape}', f'Target shape: {y.shape}', f'Length: {len(dataset)}',f'Device  Rate: {dataset.device_rate}', sep='\t|\t')


        

