import numpy as np
import matplotlib.pyplot as plt



def visualize_flow(flow, sampling_freq=100e-3, title='Flow rate characteristics', color='r'):
    time = np.array([i * sampling_freq for i in range(len(flow))])
    plt.plot(time, flow, color=color)
    plt.xlabel(f'Time(s)')
    plt.ylabel(f'Flow rate(m/s)')
    plt.title(title)
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    from configs.path_configs import DATA_PATH
    from Dataset.dataset import UroflowDataset_v2
    from Dataset.devices import Device

    dataset = UroflowDataset_v2(DATA_PATH, Device.UM)
    x, y = dataset[0]
    visualize_flow(y, title='Flow prediction for first data point')