if __name__ == '__main__':
    import os
    from Dataset.devices import Device
    from Dataset.dataset import UroflowDataset_v2
    from configs.path_configs import DATA_PATH, MINZE_READING_INDEX, MINZE_RATE
    import numpy as np
    import pandas as pd
    from scipy.io import wavfile

    DONE = []

    os.system('clear')


    shape = None
    for device in Device.get_devices():
        dataset = UroflowDataset_v2(data_path=DATA_PATH, device=device)
        print(f'WORKING ON DEVICE: {device.value}')
        
        for i in range(len(dataset)):
            try:
                x, y = dataset[i]
            except Exception as e:
                print(f'file at index {i} corrupted skipping it....')
                continue
            if i == 0:
                shape = x.shape

            if not x.shape == shape:
                print(f'Value at index {i} has shape {x.shape} instead of {shape}')
            
    
    
    
