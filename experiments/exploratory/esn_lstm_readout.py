# esn_lstm_readout.py

import torch
import torch.nn as nn

class ESNRegressorLSTM(nn.Module):
    """
    ESN reservoir (fixed) + trainable LSTM readout.
    Input  : (B, L, input_size)
    Output : (B, output_size)  -- predicts the next state after the window
    """
    def __init__(self,
                 input_size=3,
                 reservoir_size=300,
                 output_size=3,
                 leak=0.9,
                 spectral_radius=1.0,
                 input_scale=0.6,
                 bias_scale=0.0,
                 sparsity=0.03,
                 activation='tanh',
                 seed=42,
                 # LSTM readout params:
                 lstm_hidden=64,
                 lstm_layers=1,
                 lstm_dropout=0.0,   # dropout between LSTM layers if layers>1
                 bidirectional=False):
        super().__init__()

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.leak = leak
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.bias_scale = bias_scale
        self.sparsity = sparsity
        self.seed = seed

        # activation
        if activation == 'tanh':
            self.act = torch.tanh
        elif activation == 'relu':
            self.act = torch.relu
        else:
            raise ValueError("activation must be 'tanh' or 'relu'")

        # -------- fixed random reservoir init (CPU for reproducibility) --------
        g = torch.Generator(device='cpu')
        g.manual_seed(seed)

        # Input weights (N_res, N_in) in [-1, 1]
        W_in = (torch.rand((reservoir_size, input_size), generator=g) * 2.0 - 1.0)
        W_in *= input_scale

        # Bias
        b = (torch.rand((reservoir_size,), generator=g) * 2.0 - 1.0) * bias_scale

        # Sparse recurrent W
        W = torch.zeros(reservoir_size, reservoir_size)
        nnz = max(1, int(sparsity * reservoir_size * reservoir_size))
        row_idx = torch.randint(reservoir_size, (nnz,), generator=g)
        col_idx = torch.randint(reservoir_size, (nnz,), generator=g)
        vals    = (torch.rand((nnz,), generator=g) * 2.0 - 1.0)
        W[row_idx, col_idx] = vals

        # Scale to spectral radius via CPU eig
        with torch.no_grad():
            Wd = W.to(dtype=torch.float64, device='cpu')
            eigvals = torch.linalg.eigvals(Wd).abs().real
            max_ev = float(torch.max(eigvals)) if torch.any(eigvals > 0) else 1.0
            if max_ev == 0.0:
                max_ev = 1.0
            W = (W * (spectral_radius / max_ev)).to(dtype=torch.float32)

        # Freeze as buffers (not trainable)
        self.register_buffer('W_in', W_in)
        self.register_buffer('W', W)
        self.register_buffer('b', b)

        # -------- LSTM readout --------
        self.lstm = nn.LSTM(
            input_size=reservoir_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=(lstm_dropout if lstm_layers > 1 else 0.0),
            bidirectional=bidirectional,
            batch_first=True,
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(lstm_out_dim, output_size)

    def _reservoir_sequence(self, X):
        """
        Run the fixed reservoir across the input sequence and return
        the full sequence of reservoir states.
        X: (B, L, input_size)
        returns R_seq: (B, L, reservoir_size)
        """
        B, L, F = X.shape
        assert F == self.input_size
        W, W_in, b = self.W, self.W_in, self.b
        alpha = self.leak
        act = self.act

        r = X.new_zeros(B, self.reservoir_size)
        R_seq = []

        for t in range(L):
            x_t = X[:, t, :]                                  # (B, F)
            drive = x_t @ W_in.t() + r @ W.t() + b            # (B, N_res)
            r = (1 - alpha) * r + alpha * act(drive)
            R_seq.append(r)

        return torch.stack(R_seq, dim=1)  # (B, L, N_res)

    def forward(self, X):
        """
        X: (B, L, input_size)
        """
        R_seq = self._reservoir_sequence(X)           # (B, L, N_res)
        lstm_out, _ = self.lstm(R_seq)                # (B, L, H) or (B, L, 2H)
        y_hat = self.head(lstm_out[:, -1, :])         # last time step
        return y_hat
