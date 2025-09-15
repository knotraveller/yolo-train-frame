
import logging
import sys
import pickle
import os

import argparse
import numpy as np
import torch
import torch.nn as nn
import datautils


from ultralytics import YOLO


params = {
    'resume': False,
    'noload': False,
    'model': 'yolo12n.pt',
    'device': -1,
    'data_root': './datasets/data',
    'log_level': 'DEBUG',

    'train': True,

    'epochs': 10,
    'patience': 100,
    'batch': 0.60,
    'save': True,
    'save_period': -1,
    'cache': False,
    'workers': 8, 
    'exist_ok': False,
    'classes': None,
    'resume': True,
    'profile': False,
    'resume': False,

}
def init_model():
    cuda_num = torch.cuda.device_count()
    LOG.debug(f'CUDA devices: {cuda_num}')

    model_path = params["model"]
    if not os.path.exists(model_path):
        LOG.info(f'Model not found: {model_path}')
        
    if not os.path.exists(model_path) or params['noload']:
        model = YOLO(f"{params['model']}")
        LOG.info(f'Created new model: {params["model"]}')
    else:
        model = YOLO(model_path)
        LOG.info(f'Loaded existing model: {params["model"]}')

    return model

def train(model):
    data_config_path = os.path.join(params['data_root'], 'data.yaml')
    LOG.info(f'Loading data config: {data_config_path}')
    LOG.info(f'Starting training')
    results = model.train(
        data = data_config_path,
        epochs=params['epochs'], 
        device=params['device'],
        exist_ok=params['exist_ok'],
        batch=params['batch'],
        resume=params['resume'],
        project='runs',
        name='train',
        )
    LOG.info('Training complete')
    return results

def export(model):
    path = model.export(format='onnx')
    LOG.info(f'Model exported to {path}')
    return path
    

def val(model):
    LOG.info('Validating model')
    metrics = model.val(project='runs', name='val', save_json=True)
    LOG.info(f'Valid mAPs50-95: {metrics.box.maps}')
    LOG.info(f'Valid speed: {metrics.speed}')
    LOG.info('Validation complete')
    return metrics

def predict(model):
    LOG.info('Predicting')
    source = os.path.join(params['data_root'], 'test', 'images')
    results = model.predict(source=source, save=True, project='runs', name='predict')
    LOG.info('Prediction complete')
    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Task: train', action='store_true')
    parser.add_argument('--resume', help='Resume training', action='store_true')
    parser.add_argument('--noload', help='Do not load saved model', action='store_true')
    parser.add_argument('--val', help='Task: valuate', action='store_true')
    parser.add_argument('--predict', help='Task: predict', action='store_true')
    
    parser.add_argument('-d', '--data_root', help='data folder in the datasets.',
                        type=str, required=False, default='./data')
    parser.add_argument('-m', '--model', help='Model type', 
                        type=str, required=False, default='yolo12n.pt')
    parser.add_argument('-l', '--log_level', help='Log level',
                        type=str, required=False, default='DEBUG') 
    parser.add_argument('-e', '--epochs', help='Number of epochs',
                        type=int, required=False, default=10)
    
    args = parser.parse_args()

    LOG = datautils.init_logging('yolo', logging.DEBUG)
    LOG.info('----------STARTED----------')
    LOG.debug(f'Args: {args}')

    params['resume'] = args.resume
    params['data_root'] = os.path.join('./datasets', args.data_root)
    params['model'] = args.model
    params['noload'] = args.noload
    params['epochs'] = args.epochs
    

    model = init_model()

    if args.train:
        train(model)
        export(model)

    if args.val:
        val(model)

    if args.predict:
        predict(model)
