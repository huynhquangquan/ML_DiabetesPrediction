import pickle
from pathlib import Path
import yaml
import pandas as pd
import joblib


def find_path_from_models(name):
    dir = Path(__file__).parent
    model_path = dir / '..' / "models" / f'{name}'
    return model_path.resolve()

def read_raw(datasetname):
    current_dir = Path(__file__).parent
    model_path = current_dir / '../data/raw' / f'{datasetname}'
    df = pd.read_csv(model_path.resolve())
    return df

def read_processed(datasetname):
    current_dir = Path(__file__).parent
    model_path = current_dir / '../data/processed' / f'{datasetname}'
    df = pd.read_csv(model_path.resolve())
    return df

def joblib_dump(model,name):
    model_path = find_path_from_models(name)
    joblib.dump(model,model_path)

def joblib_load(name):
    model_path = find_path_from_models(name)
    loaded = joblib.load(model_path)
    return loaded