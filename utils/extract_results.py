import pandas as pd
import numpy as np
from Dataset.devices import Device

def extract_device_results(state_dict, device: Device):
    MODELS = ['rf', 'svr', 'xgboost']
    metrics = ['r2', 'mae']

    r2_scores = []
    mae_scores = []

    for model in MODELS:
        r2_scores.append(state_dict[device.value][model][metrics[0]])
        mae_scores.append(state_dict[device.value][model][metrics[-1]])
    
    result_df = pd.DataFrame(data=np.array([r2_scores, mae_scores]), columns=MODELS, index=metrics)
    
    return result_df

def summarize_results(state_dict):
    for device in Device.get_devices():
        device_df = extract_device_results(state_dict=state_dict, device=device)
        print(f'Test Results for: {(device.value).upper()}', '='*34, device_df, '\n', sep='\n')



if __name__ == '__main__':
    import os
    import joblib
    from configs.path_configs import CLASSICAL_MODEL_PATH


    os.system('clear')

    state_dict = joblib.load(CLASSICAL_MODEL_PATH)
    summarize_results(state_dict)
    