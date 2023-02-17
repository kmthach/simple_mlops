import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import read_params


def split_and_save_data(config_path):
    config = read_params(config_path)
    
    data_path = config['load_data']['raw_dataset_csv']
    test_size = config['split_data']['test_size']
    random_state = config['base']['random_state']
    
    data = pd.read_csv(
        data_path, 
        sep=',', 
        encoding='utf-8')
    train, test = train_test_split(
        data,
        test_size = test_size,
        random_state = random_state
    )
    
    train_path = config['split_data']['train_path']
    test_path = config['split_data']['test_path']
    
    train.to_csv(
        train_path,
        sep =',',
        index = False,
        encoding = 'utf-8'
    )
    
    test.to_csv(
        test_path,
        sep =',',
        index = False,
        encoding = 'utf-8'
    )
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        '--config',
        default= 'params.yaml'
    )
    
    parsed_args = args.parse_args()
    split_and_save_data(parsed_args.config)
    