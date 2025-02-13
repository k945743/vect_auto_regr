import argparse
from autoreg import VectAutoReg
from config import Config
import numpy as np
import pandas as pd
import yaml
import os

def _generate(autoreg, trajectory_length, sample_count, idx_len, dest_folder):
    for idx in range(sample_count):
        series = autoreg(trajectory_length)
        data_columns = [f'x_{i}' for i in range(1, series.shape[0]+1)]
        df = pd.DataFrame(series.T, columns=data_columns)
        df['time'] = np.arange(0, len(df))
        df = df[['time']+data_columns]
        seq_id = str(idx).zfill(idx_len)
        file_path = os.path.join(dest_folder, f"seq_{seq_id}.parquet")
        df.to_parquet(file_path)

def generate(config):
    coeffs = [config.COEFFS[key] for key in config.COEFFS.keys()]
    init_values = np.array(config.INIT) if config.INIT is not None else None
    autoreg = VectAutoReg(coeffs, init = init_values)

    trajectory_length = config.trajectory_length
    sample_count = config.sample_count
    idx_len = len(str(sample_count))
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
    dest_folder = os.path.join(config.dest_folder, "VAR_"+timestamp)
    os.makedirs(dest_folder, exist_ok=True)

    _generate(autoreg, trajectory_length, sample_count, idx_len, dest_folder)

    autoreg._generate_companion()
    is_stable = autoreg._compute_stability()

    return dest_folder, config, is_stable

def _log(dest_folder, config, comp):
    log_folder = os.path.join(dest_folder, "log")
    os.makedirs(log_folder, exist_ok=True)
    log_file_path = os.path.join(log_folder, "log.txt")
    with open(log_file_path, "w") as f:
        f.write(f"is_stable: {str(comp)}\n")

    config_file_path = os.path.join(log_folder, "config.yaml")
    with open(config_file_path, "w") as f:
        yaml.dump(config.dict(), f)

    
    

if __name__ == "__main__":
    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the yaml config file")

    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    config = Config(**config)
    dest_folder, config, comp  = generate(config)

    _log(dest_folder, config, comp)