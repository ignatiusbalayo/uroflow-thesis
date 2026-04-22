import joblib
from configs.path_configs import NORMALIZATION_PARAMS_PATH
from Dataset.devices import Device
from enum import Enum
from utils.compute_rescale_params import compute_rescale_configs_params

class Transform_keys(Enum):
    LFT = 'lft'
    Mel_1d = 'mel_1d'
    Mel_2d = 'mel_2d'

    @classmethod
    def get_transform_keys(cls):
        return list(cls)


def  read_norm_data(device: Device, transform_key: Transform_keys):
    """Reads the data for a given transform """
    try:
        state_obj = joblib.load(NORMALIZATION_PARAMS_PATH)
    except:
        state_obj = compute_rescale_configs_params()
    temp = state_obj[device.value][transform_key.value]
    return temp


if __name__ == '__main__':
    import os
    import joblib

    os.system('clear')

    temp  = read_norm_data(Device.PHONE, Transform_keys.Mel_2d)
    print(f'Normalization data for device {Device.UM.value}', temp, sep='\n')





