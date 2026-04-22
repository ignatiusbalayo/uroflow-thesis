import joblib
from configs.path_configs import NORMALIZATION_PARAMS_PATH
from Dataset.devices import Device
from enum import Enum

class Transform_keys(Enum):
    LFT = 'lft'
    Mel_1d = 'mel_1d'
    Mel_2d = 'mel_2d'

    @classmethod
    def get_transform_keys(cls):
        return list(cls)


def  read_data_one(device: Device, transform_key: Transform_keys):
    """Reads the data for a given transform """
    state_obj = joblib.load(NORMALIZATION_PARAMS_PATH)
    temp = state_obj[device.value][transform_key.value]
    return temp



if __name__ == '__main__':
    import os
    os.system('clear')

    state_obj = read_data_one(Device.UM, Transform_keys.Mel_2d)
    print(state_obj)

