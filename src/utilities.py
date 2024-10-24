import pickle
from pathlib import Path
import yaml
import pandas as pd
import joblib
import yaml

def dataset_select(): # change dataset in config.dataset_config
    dir = Path(__file__).parent.parent
    with open(dir / 'config' / 'dataset_config.yaml','r') as file:
        dataset = yaml.safe_load(file)
    return dataset

def model_select(): # every run in evaluate, prediction, visualize, will select the model to run, change model in config.model_select
    dir = Path(__file__).parent.parent
    with open(dir / 'config' / 'model_select.yaml','r') as file:
        model = yaml.safe_load(file)
    return model['model']

def find_path_from_models(name): # Side function to create path in .virtualenvironment.models
    dir = Path(__file__).parent
    model_path = dir / '..' / "models" / f'{name}'
    return model_path.resolve()

def check_model(name):
    if "random" in name.lower() and "forest" in name.lower():
        model = "random_forest"
    if "logistic" in name.lower():
        model = "logistic"

    try:
        check_models = joblib_load(model)
    except:
        print(f'Không có mô hình {model}')
        return False
    return True

def check_raw():
    try:
        check_raw = read_raw("diabetes.csv")
    except:
        print(f'Chưa có dataset')
        return False
    return True

def check_processed():
    try:
        check_test = read_processed("test.csv")
        check_train = read_processed("train.csv")
    except:
        print(f'Processed không có hoặc không đầy đủ')
        return False
    return True

def read_raw(datasetname):
    current_dir = Path(__file__).parent
    model_path = current_dir / '..' / 'data' / 'raw' / f'{datasetname}'
    df = pd.read_csv(model_path.resolve())
    return df

def read_processed(datasetname):
    current_dir = Path(__file__).parent
    model_path = current_dir / '..' / 'data' / 'processed' / f'{datasetname}'
    df = pd.read_csv(model_path.resolve())
    return df

def joblib_dump(model,name):
    model_path = find_path_from_models(name)
    joblib.dump(model,model_path)

def joblib_load(name):
    model_path = find_path_from_models(name)
    loaded = joblib.load(model_path)
    return loaded
