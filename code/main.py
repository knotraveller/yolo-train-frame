
import logging
import sys
import pickle
import os

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as datautils
import datautils


DEVICE = 'cuda'

from ultralytics import YOLO

def train(data_root, result_dir, model_path, model, noload):
    
    pass

def test(data_root, result_dir, model_path, model, noload):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Task: train', action='store_true')
    parser.add_argument('--test', help='Task: test', action='store_true')
    parser.add_argument('-d', '--data_root', help='Root folder for datasets.',
                        type=str, required=False, default='./data')
    parser.add_argument('-r', '--result_dir', help='folder for results.',
                        type=str, required=False, default='./results')
    parser.add_argument('-mp', '--model_path', help='Path to saved model',
                        type=str, required=False, default='./models')
    parser.add_argument('-m', '--model', help='Model type', 
                        type=str, required=False, default='yolov12n')
    parser.add_argument('-l', '--log_level', help='Log level',
                        type=str, required=False, default='DEBUG') 
    parser.add_argument('--noload', help='Do not load saved model', action='store_true')
    args = parser.parse_args()

    LOG = datautils.init_logging(args.model, getattr(logging, args.log_level.upper()))
    LOG.info('----------STARTED----------')
    LOG.debug(f'Args: {args}')

    if args.train:
        train(args.data_root, args.result_dir, args.model_path, args.model, args.noload)

    if args.test:
        test(args.data_root, args.result_dir, args.model_path, args.model, args.noload)
