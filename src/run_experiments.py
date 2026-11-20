import time
import os
import sys
import glob
import shutil
import argparse
import datetime
import numpy as np
import torch
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

import Lorenz_parts


def main():

    # default parameters
    parser = argparse.ArgumentParser(description='Run ESN - Lorenz63 experiments')
    parser.add_argument('--train_data', type=str, default=os.path.join(PROJECT_DIR, 'data', 'train_data.pkl'), 
                        help='Path to train data')
    parser.add_argument('--test_data', type=str, default=os.path.join(PROJECT_DIR, 'data', 'test_data.pkl'), 
                        help='Path to test data')
    parser.add_argument('--hist_data', type=str, default=os.path.join(PROJECT_DIR, 'data', 'hist_data.pkl'), 
                        help='Path to histograms data')
    parser.add_argument('--hist_length', type=int, default=10_000_000, 
                        help='Length of histogram trajectory')
    parser.add_argument('--epochs', type=int, default=1, 
                        help='Number of training epochs')
    parser.add_argument('--train_batches_per_epoch', type=int, default=5000, 
                        help='Num. train batches per epoch')
    parser.add_argument('--test_batches_per_epoch', type=int, default=100, 
                        help='Num. test batches per epoch')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size')
    parser.add_argument('--warm_length', type=int, default=10, 
                        help='Length of warm trajectory')
    parser.add_argument('--x_length', type=int, default=100, 
                        help='Length of input trajectory')
    parser.add_argument('--data_dim', type=int, default=3, 
                        help='Data dimension')
    parser.add_argument('--standard', type=bool, default=True, 
                        help='Standard scaler')
    parser.add_argument('--mean', type=list, default=[0,0,23.67], 
                        help='Mean values of Lorenz system')
    parser.add_argument('--std', type=list, default=[7.95,9.04,8.58], 
                        help='Standard deviation of Lorenz system')
    parser.add_argument('--N', type=int, default=100, 
                        help='Length of input trajectory')
    parser.add_argument('--degree', type=int, default=50, 
                        help='Num. connections per neuron in reservoir')
    parser.add_argument('--radius', type=float, default=0.5, 
                        help='Reservoir spectral radius')
    parser.add_argument('--w', type=float, default=0.5, 
                        help='Initialization range for Win = unif[-w,w]')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-5, 
                        help='Weight decay')
    parser.add_argument('--device', type=str, default=('cpu' if torch.cuda.is_available() else 'cpu'), help='Device')
    args = parser.parse_args()

    
    # create experiment folder
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_dir = os.path.join(PROJECT_DIR, 'experiments', f'{timestamp}_{os.path.basename(SCRIPT_DIR)}')
    os.makedirs(exp_dir, exist_ok=True)


    # backup scripts
    scripts_dir = os.path.join(exp_dir, 'scripts')
    os.makedirs(scripts_dir, exist_ok=True)
    for file in glob.glob(os.path.join(SCRIPT_DIR, '*.py')):
        shutil.copy(file, scripts_dir)

    

    pdf = PdfPages(os.path.join(exp_dir, 'ESN_Lorenz63_experiments.pdf'))
    
    exp_numb = 0
    fig = Lorenz_parts.start_experiment(exp_dir, args, exp_numb)
    pdf.savefig(fig)

    pdf.close()
    






if __name__ == '__main__':
    main()
