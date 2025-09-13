from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

class MLPModule(nn.Module):
    def __init__(self, in_dim=32, hidden=128, out_dim=3):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, X): return self.seq(X)

net = NeuralNetClassifier(
    MLPModule,
    max_epochs=20,
    lr=1e-3,
    iterator_train__shuffle=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

param_grid = {
    "lr": [1e-2, 3e-3, 1e-3],
    "module__hidden": [64, 128, 256],
    "max_epochs": [10, 20],
}

gs = GridSearchCV(net, param_grid=param_grid, cv=5, n_jobs=-1, refit=True)
gs.fit(X.numpy(), y.numpy())
print(gs.best_params_, gs.best_score_)
best_model = gs.best_estimator_
