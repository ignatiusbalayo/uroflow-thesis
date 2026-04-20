import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Dataset.dataset import UroflowDataset_v2
from Dataset.data_loader import UrflowDataLoader
from Transforms.base_transform import LFT
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from Dataset.devices import Device
from configs.path_configs import DATA_PATH, SEED
from sklearn.metrics import r2_score, mean_absolute_error

def train_test():
    NO_BINS = 20
    
    MODELS = {
        'rf': RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=5, max_features="sqrt", criterion="squared_error"),
        'svr': SVR(kernel='rbf', C=1, epsilon=0.1, gamma='auto'),
        'xgboost': XGBRegressor(n_estimators=100, learning_rate=0.05, objective='reg:squarederror')
    }

    lft = LFT(no_bins=NO_BINS)
    global_state = {}

    for device in Device.get_devices():
        dataset = UroflowDataset_v2(data_path=DATA_PATH, device=device)
        dataloader = UrflowDataLoader(dataset, batch_size=len(dataset), transform=lft, shuffle=True, permutate=False)

        data_iter = iter(dataloader)
        x, y = next(data_iter)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)

        temp = {}
        for model in MODELS.keys():
            pipleine = Pipeline(
                [
                    ('scaler', StandardScaler()),
                    ('regressor', TransformedTargetRegressor(
                        regressor=MODELS[model],
                        transformer=StandardScaler()
                    ))
                ]
            )

            pipleine.fit(x_train, y_train)
            y_pred = pipleine.predict(x_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_pred, y_test)
          

            state_data = {
                'r2': r2,
                'mae': mae,
                'model_pipline': pipleine
            }
            temp[model] = state_data
        global_state[device.value] = temp
        joblib.dump(global_state, 'classical_model_states.joblib')

    return global_state


if __name__ == '__main__':
    train_test()
