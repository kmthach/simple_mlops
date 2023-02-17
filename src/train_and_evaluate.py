import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from get_data import read_params
import argparse
import joblib
import json

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_evaluate(config_path):
    config = read_params(config_path)
    
    train_path = config['split_data']['train_path']
    test_path = config['split_data']['test_path']
    target_col = config['base']['target_col']
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    train_X = train.drop(target_col, axis = 1)
    test_X = test.drop(target_col, axis = 1)
    
    train_y = train[target_col]
    test_y = test[target_col]
    
    alpha = config['estimators']['ElasticNet']['params']['alpha']
    l1_ratio = config['estimators']['ElasticNet']['params']['l1_ratio']
    random_state = config['base']['random_state']
    
    model = ElasticNet(
        alpha = alpha,
        l1_ratio = l1_ratio,
        random_state= random_state
    )
    
    model.fit(train_X, train_y)
    
    predicts = model.predict(test_X)
    rmse, mae, r2 = eval_metrics(test_y, predicts)
    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    
    score_path = config['report']['score']
    params_path = config['report']['params']
    
    with open(score_path, 'w') as f:
        score = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        json.dump(score, f, indent = 4)
        
    with open(params_path, 'w') as f:
        params = {
            'alpha': alpha,
            'l1_ratio': l1_ratio,
        }
        json.dump(params, f, indent = 4)
        
    model_dir = config['model_dir']
    os.makedirs(model_dir, exist_ok = True)
    model_path = os.path.join(model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default = 'params.yaml')
    parsed_args = args.parse_args()
    train_and_evaluate(parsed_args.config)