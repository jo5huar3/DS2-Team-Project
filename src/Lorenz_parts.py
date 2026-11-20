import random
import math 
import time
import os

import numpy as np
import pandas as pd
import pickle as pk

import scipy.sparse as sparse
import scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

def cur_time(start_time):
    return round(time.time() - start_time, 2)


def loader(args, train=True):

    warm_length = args.warm_length
    x_length = args.x_length
    batch_size = args.batch_size
    train_batches_per_epoch = args.train_batches_per_epoch
    test_batches_per_epoch = args.test_batches_per_epoch
    

    if train:
        with open(args.train_data, 'rb') as file:
            data = pk.load(file)
        batches = train_batches_per_epoch
        indices = np.arange(0, data.shape[0] - (warm_length+x_length+1) + 1)
        np.random.shuffle(indices)

    else:
        with open(args.test_data, 'rb') as file:
            data = pk.load(file)
        batches = test_batches_per_epoch
        indices = np.arange(0, data.shape[0], (warm_length+x_length+1))[:-1]


    if args.standard:
        data = (data - args.mean) / args.std


    for b in range(batches):

        warm = []
        x = []
        y = []
        
        for ind in indices[b * batch_size : (b+1) * batch_size]:

            warm.append(data[ind : ind + warm_length])
            x.append(data[ind + warm_length : ind + warm_length + x_length])
            y.append(data[ind + warm_length + 1 : ind + warm_length + x_length + 1])

        warm = torch.from_numpy(np.stack(warm)).float()
        x = torch.from_numpy(np.stack(x)).float()
        y = torch.from_numpy(np.stack(y)).float()

        yield warm, x, y 


def get_linear_warmup_cosine_decay_scheduler(optimizer, total_steps, warmup):
    
    warmup_steps = total_steps * warmup

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        
        else:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 0.99)))
        
    
    return LambdaLR(optimizer, lr_lambda)


def ks_distance(a, b):
    # Sort the combined samples
    data = np.sort(np.concatenate([a, b]))
    # Compute ECDFs
    ecdf_a = np.searchsorted(np.sort(a), data, side='right') / len(a)
    ecdf_b = np.searchsorted(np.sort(b), data, side='right') / len(b)
    # Compute max absolute difference
    D = np.max(np.abs(ecdf_a - ecdf_b))
    return D, data, ecdf_a, ecdf_b



class ESN(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.N = args.N
        self.degree = args.degree
        self.radius = args.radius
        self.data_dim = args.data_dim
        self.w = args.w
        
        self.Win = self.generate_win()
        self.A = self.generate_A()
        
        self.Wout = nn.Sequential(
                                    nn.Linear(self.N, int((self.N-self.data_dim)*2/3)+self.data_dim),
                                    nn.GELU(),
                                    nn.Linear(int((self.N-self.data_dim)*2/3)+self.data_dim, 
                                              int((self.N-self.data_dim)*1/3)+self.data_dim),
                                    nn.GELU(),
                                    nn.Linear(int((self.N-self.data_dim)*1/3)+self.data_dim, self.data_dim),
                                )
        '''
        self.Wout = nn.Sequential(
                                    nn.Linear(self.N, self.data_dim)
                                )
        '''

    def generate_win(self):
        
        Win = nn.Linear(self.data_dim, self.N, bias=False)
        with torch.no_grad():
            Win.weight.copy_(torch.FloatTensor(self.N, self.data_dim).uniform_(-self.w, self.w))
            Win.weight.requires_grad = False


        return Win

    def generate_A(self):

        res_vals = sparse.rand(self.N, self.N, density=self.degree/float(self.N)).todense()
        eigvals = np.linalg.eigvals(res_vals)
        max_abs_eigval = np.max(np.abs(eigvals))
        res_vals = (res_vals/max_abs_eigval) * self.radius

        if (res_vals == np.inf).sum() > 0:
            return self.generate_A()

        reservoir = nn.Linear(self.N, self.N, bias=False)
        with torch.no_grad():
            reservoir.weight.copy_(torch.FloatTensor(res_vals.T))
            reservoir.weight.requires_grad = False

        return reservoir
    
    def warming_states(self, warm):

        batch = warm.shape[0]
        length = warm.shape[1]

        warm_state = torch.zeros(batch, self.N, device=warm.device)
        warm_state = torch.tanh(self.Win(warm[:,0]))
        for i in range(1, length):
            warm_state = torch.tanh(self.A(warm_state) + self.Win(warm[:,i]))

        return warm_state

    def forward(self, warm, x):

        warm_state = self.warming_states(warm)

        batch = x.shape[0]
        length = x.shape[1]

        states = torch.zeros(batch, length, self.N, device=x.device)
        predict = torch.zeros(batch, length, self.data_dim, device=x.device)
        states[:,0] = torch.tanh(self.A(warm_state) + self.Win(x[:,0]))
        
        for i in range(1, length):
            states[:,i] = torch.tanh(self.A(states[:,i-1]) + self.Win(x[:,i]))

        predict = self.Wout(states) 
        
        return predict

    def predict_long_trajectory(self, warm, start_point, length, start_time):

        warm_state = self.warming_states(warm.unsqueeze(0)).squeeze(0)

        predict = torch.zeros(length, self.data_dim, device=start_point.device)
        
        predict[0] = start_point
        state = torch.tanh(self.A(warm_state) + self.Win(predict[0]))
        predict[1] = self.Wout(state) 
        
        for i in range(2, length):
            state = torch.tanh(self.A(state) + self.Win(predict[i-1]))
            predict[i] = self.Wout(state) 

            if (i % (length//10) == 0):
                print(f"Prediction {i / length} completed, time: {cur_time(start_time)}, ", flush=True)
        
        return predict



def train(model, 
          optimizer, 
          scheduler,
          criterion, 
          train_loss, 
          args):

    model.train()

    for warm, x, y in loader(args, train=True):

        warm = warm.to(args.device)
        x = x.to(args.device)
        y = y.to(args.device)

        optimizer.zero_grad()

        loss = criterion(model(warm, x), y)
        loss.backward()

        optimizer.step()
        scheduler.step()

        train_loss.append(loss.item())

    return train_loss
   

def test(model, 
         criterion, 
         test_loss, 
         args):
        
    model.eval()

    loss_sum = 0

    with torch.no_grad():
        for warm, x, y in loader(args, train=False):

            warm = warm.to(args.device)
            x = x.to(args.device)
            y = y.to(args.device)

            loss = criterion(model(warm, x), y)

            loss_sum += loss.item()

    test_loss.append(loss_sum / args.test_batches_per_epoch)

    return test_loss


def final_test(model, 
               train_loss, 
               test_loss, 
               args,
               start_time):

    # prepare model and other variables for ploting metric for the final test on the best checkpoint
    model.eval()
    
    warm_length = args.warm_length
    hist_length = args.hist_length
    
    with open(args.hist_data, 'rb') as file:
        true_data = pk.load(file)


    with torch.no_grad():

        warm = true_data[:warm_length]
        start_point = true_data[warm_length]

        if args.standard:
            warm = (warm - args.mean) / args.std
            start_point = (start_point - args.mean) / args.std

        warm = torch.FloatTensor(warm).to(args.device)
        start_point = torch.FloatTensor(start_point).to(args.device)
        pred_data = model.predict_long_trajectory(warm, start_point, hist_length, start_time).cpu().numpy()

        if args.standard:
            pred_data = pred_data * args.std + args.mean
    


    # plot histograms and metrics

    description = f'''  True: X Y Z
    m1: {stats.moment(true_data, moment=1, center=0)}
    m2: {stats.moment(true_data, moment=2)}

    Predicted: X Y Z
    m1: {stats.moment(pred_data, moment=1, center=0)}
    m2: {stats.moment(pred_data, moment=2)}

    Model parameters:
    N: {args.N}
    degree: {args.degree}
    radius: {args.radius}
    w: {args.w}

    Optimizer parameters:
    lr: {args.lr}
    wd: {args.wd}

    Training parameters:
    epochs: {args.epochs}
    train_batches_per_epoch: {args.train_batches_per_epoch}
    batch_size: {args.batch_size}
    warm_length: {args.warm_length}
    x_length: {args.x_length}
    '''
    

    traj_window = 200
    ts = np.arange(traj_window)

    fig = plt.figure(constrained_layout=True, figsize=(36,12))
    fig.legend(title=description, bbox_to_anchor=(0.15, 0.45))
    subfigs = fig.subfigures(1, 3)  
    

    #fig = plt.figure(figsize=(12, 12), layout='constrained')
    gs = GridSpec(6, 2, figure=subfigs[0])
    ax1 = fig.add_subplot(gs[0:3,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,1])
    ax4 = fig.add_subplot(gs[2,1])
    ax5 = fig.add_subplot(gs[3,1])
    ax6 = fig.add_subplot(gs[4,1])
    ax7 = fig.add_subplot(gs[5,1])

    
    ax1.semilogy(train_loss, label='train')
    ax1.semilogy(np.linspace(1,len(train_loss),len(test_loss)), test_loss, label='test')
    ax1.set_xlabel('Updates')
    ax1.set_ylabel('MSE')
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(ts, pred_data[ : traj_window, 0], lw=0.5)
    ax2.plot(ts, true_data[warm_length : warm_length+traj_window, 0], lw=0.5)
    ax2.set_xlabel(f'First {traj_window} steps')
    ax2.set_ylabel('X')
    ax2.grid(True)

    ax3.plot(ts, pred_data[ : traj_window, 1], lw=0.5)
    ax3.plot(ts, true_data[warm_length : warm_length+traj_window, 1], lw=0.5)
    ax3.set_xlabel(f'First {traj_window} steps')
    ax3.set_ylabel('Y')
    ax3.grid(True)

    ax4.plot(ts, pred_data[ : traj_window, 2], lw=0.5)
    ax4.plot(ts, true_data[warm_length : warm_length+traj_window, 2], lw=0.5)
    ax4.set_xlabel(f'First {traj_window} steps')
    ax4.set_ylabel('Z')
    ax4.grid(True)

    ax5.plot(ts, pred_data[-traj_window : , 0], lw=0.5)
    ax5.set_xlabel(f'Last {traj_window} steps')
    ax5.set_ylabel('X')
    ax5.grid(True)

    ax6.plot(ts, pred_data[-traj_window : , 1], lw=0.5)
    ax6.set_xlabel(f'Last {traj_window} steps')
    ax6.set_ylabel('Y')
    ax6.grid(True)

    ax7.plot(ts, pred_data[-traj_window : , 2], lw=0.5)
    ax7.set_xlabel(f'Last {traj_window} steps')
    ax7.set_ylabel('Z')
    ax7.grid(True)



    #fig = plt.figure(figsize=(12, 12), layout='constrained')
    gs = GridSpec(3, 3, figure=subfigs[1])
    quantile = [0.01, 0.99]
    pdf_lim = [0.015,0.01,0.018]

    for i in range(pred_data.shape[1]):
        ax1 = fig.add_subplot(gs[i,0])
        ax2 = fig.add_subplot(gs[i,1])
        ax3 = fig.add_subplot(gs[i,2])

        var_p = pred_data[:,i]
        var_t = true_data[:,i]

        t_min = var_t.min() - 1
        t_max = var_t.max() + 1
        bins = np.linspace(t_min, t_max, 200)

        ax1.hist(var_p, bins=bins, alpha=0.8, density=True, label='Esn')
        ax1.hist(var_t, bins=bins, alpha=0.5, density=True, label='Lorenz')
        ax1.set_xlabel(f'Variable {i}')
        ax1.set_ylabel('pdf')
        ax1.legend()


        bins = np.linspace(t_min, t_max, 2000)

        quant = np.quantile(var_t, quantile[0])
        ax2.hist(var_p, bins=bins, alpha=0.8, density=True, log=True, label='Esn')
        ax2.hist(var_t, bins=bins, alpha=0.5, density=True, log=True, label='Lorenz')
        ax2.set_xlim(t_min,quant)
        ax2.set_ylim(0, pdf_lim[i])
        ax2.set_xlabel(f'Left var {i} tail')
        ax2.set_ylabel('pdf')
        ax2.legend()

        quant = np.quantile(var_t, quantile[1])
        ax3.hist(var_p, bins=bins, alpha=0.8, density=True, log=True, label='Esn')
        ax3.hist(var_t, bins=bins, alpha=0.5, density=True, log=True, label='Lorenz')
        ax3.set_xlim(quant,t_max)
        ax3.set_ylim(0, pdf_lim[i])
        ax3.set_xlabel(f'Right var {i} tail')
        ax3.set_ylabel('pdf')
        ax3.legend()    



    gs = GridSpec(3, 3, figure=subfigs[2])
    quantile = [0.01, 0.99]

    for i in range(pred_data.shape[1]):
        ax1 = fig.add_subplot(gs[i,0])
        ax2 = fig.add_subplot(gs[i,1])
        ax3 = fig.add_subplot(gs[i,2])

        var_p = pred_data[:,i]
        var_t = true_data[:,i]

        t_min = var_p.min() - 1
        t_max = var_p.max() + 1


        D, x, cdf_t, cdf_p = ks_distance(var_t, var_p)
        ax1.plot(x, cdf_p, label=f'Esn\nK_dist = {np.round(D*100, 2)}%')
        ax1.plot(x, cdf_t, label=f'Lorenz')
        ax1.set_xlabel(f'Variable {i}')
        ax1.set_ylabel('cdf')
        ax1.legend()

        quant = np.quantile(var_t, quantile[0])
        D, x, cdf_t, cdf_p = ks_distance(var_t[var_t < quant], var_p[var_p < quant])
        ax2.plot(x, cdf_p, label=f'Esn\nK_dist = {np.round(D*100, 2)}%')
        ax2.plot(x, cdf_t, label=f'Lorenz')
        ax2.set_xlabel(f'Left var {i} tail')
        ax2.set_ylabel('cdf')
        ax2.legend()

        quant = np.quantile(var_t, quantile[1])
        D, x, cdf_t, cdf_p = ks_distance(var_t[quant < var_t], var_p[quant < var_p])
        ax3.plot(x, cdf_p, label=f'Esn\nK_dist = {np.round(D*100, 2)}%')
        ax3.plot(x, cdf_t, label=f'Lorenz')
        ax3.set_xlabel(f'Right var {i} tail')
        ax3.set_ylabel('cdf')
        ax3.legend()  




    return fig


def start_experiment(exp_dir, 
                     args, 
                     exp_numb):
    start_time = time.time()
     
    np.random.seed(0)
    torch.manual_seed(0)

    model = ESN(args).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=args.lr,
                                  weight_decay=args.wd)
    scheduler = get_linear_warmup_cosine_decay_scheduler(optimizer,
                                                         total_steps=args.epochs*args.train_batches_per_epoch,
                                                         warmup=0.05)
    criterion = nn.MSELoss()

    model_best_checkp = os.path.join(exp_dir, f'model_p{exp_numb}.pth')

    print(f'Experiment {exp_numb} -----------------------------------------', flush=True)
    print(f'Total num. epochs: {args.epochs}', flush=True)
    print(f'Num. train batches per epoch: {args.train_batches_per_epoch}', flush=True)
    print(f'Num. test batches per epoch: {args.test_batches_per_epoch}', flush=True)

    train_loss = []
    test_loss = []
    best_val = float('inf')

    for epoch in range(1, args.epochs+1):

        print(f"Epoch {epoch} train phase, time: {cur_time(start_time)}", flush=True)

        train_loss = train(model, optimizer, scheduler, criterion, train_loss, args)

        print(f"Epoch {epoch} test phase, time: {cur_time(start_time)}", flush=True)

        test_loss = test(model, criterion, test_loss, args)

        print(f"Epoch {epoch} finished, time: {cur_time(start_time)}, train loss={np.mean(train_loss[-10:])}, test loss={test_loss[-1]}", flush=True)
        
        # overwrite checkpoint on improvement
        if test_loss[-1] < best_val:

            best_val = test_loss[-1]
            torch.save(model.state_dict(), model_best_checkp)
            print(f"Saved model at epoch {epoch}, time {cur_time(start_time)}", flush=True)


    # save final model
    model_final_checkp = os.path.join(exp_dir, f'model_p{exp_numb}_final.pth')
    torch.save(model.state_dict(), model_final_checkp)
    print(f"Saved final model, time {cur_time(start_time)}", flush=True)


    # final test + metrics
    print(f"Final test phase, time: {cur_time(start_time)}", flush=True)

    model.load_state_dict(torch.load(model_best_checkp))
    fig = final_test(model, train_loss, test_loss, args, start_time)

    print(f"Final test finished, time: {cur_time(start_time)}", flush=True)

    return fig