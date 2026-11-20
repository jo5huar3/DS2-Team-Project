
"""
Train an ESN on Lorenz-63 and generate histogram plots.
Outputs: PNG figures under ./plots and metrics.json
"""
import os, json, numpy as np, matplotlib.pyplot as plt

def lorenz63(state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    return np.array([sigma*(y-x), x*(rho-z)-y, x*y - beta*z])

def rk4_step(f, state, dt):
    k1=f(state); k2=f(state+0.5*dt*k1); k3=f(state+0.5*dt*k2); k4=f(state+dt*k3)
    return state+(dt/6.0)*(k1+2*k2+2*k3+k4)

def generate_l63(n_steps=4000, dt=0.01, x0=(1,1,1)):
    X=np.zeros((n_steps,3)); s=np.array(x0,float)
    for _ in range(300): s=rk4_step(lorenz63,s,dt)
    for t in range(n_steps): s=rk4_step(lorenz63,s,dt); X[t]=s
    return X

class Standardizer:
    def fit(self,data): self.m=data.mean(0); self.s=data.std(0); self.s[self.s==0]=1; return self
    def transform(self,data): return (data-self.m)/self.s
    def inverse_transform(self,data): return data*self.s+self.m

class ESN:
    def __init__(self,input_dim=3,reservoir_size=300,output_dim=3,spectral_radius=0.9,density=0.05,leak_rate=0.3,reg=1e-4,seed=7):
        rng=np.random.default_rng(seed)
        self.Win=rng.normal(0,1,(reservoir_size,input_dim+1))
        W=rng.normal(0,1,(reservoir_size,reservoir_size))
        mask=rng.random((reservoir_size,reservoir_size))<density
        W*=mask
        v=rng.normal(size=reservoir_size); v/=np.linalg.norm(v)
        for _ in range(10): v=W@v; v/=np.linalg.norm(v)
        sr=np.linalg.norm(W@v); self.W=W/sr*spectral_radius
        self.reservoir_size=reservoir_size; self.reg=reg; self.leak=leak_rate
    def fit(self,U,Y,washout=200):
        x=np.zeros(self.reservoir_size); Xs=[]; Ys=[]
        for t in range(len(U)):
            pre=self.Win@np.concatenate(([1],U[t]))+self.W@x
            x=(1-self.leak)*x+self.leak*np.tanh(pre)
            if t>=washout: Xs.append(np.concatenate(([1],x,U[t]))); Ys.append(Y[t])
        Xs=np.stack(Xs); Ys=np.stack(Ys)
        XtX=Xs.T@Xs+self.reg*np.eye(Xs.shape[1])
        self.Wout=np.linalg.solve(XtX,Xs.T@Ys)
    def predict(self,U_seed,steps):
        x=np.zeros(self.reservoir_size)
        for t in range(len(U_seed)):
            pre=self.Win@np.concatenate(([1],U_seed[t]))+self.W@x
            x=(1-self.leak)*x+self.leak*np.tanh(pre)
        u=U_seed[-1]; outs=[]
        for _ in range(steps):
            s=np.concatenate(([1],x,u)); y=s@self.Wout; outs.append(y)
            pre=self.Win@np.concatenate(([1],y))+self.W@x
            x=(1-self.leak)*x+self.leak*np.tanh(pre); u=y
        return np.stack(outs)

def nrmse(a,b): return np.sqrt(np.mean((a-b)**2))/(np.std(a)+1e-12)

def main():
    os.makedirs("plots",exist_ok=True)
    X=generate_l63(); Tt=2800
    train,test=X[:Tt],X[Tt:]
    sc=Standardizer().fit(train)
    tr,te=sc.transform(train),sc.transform(test)
    U,Y=tr[:-1],tr[1:]
    esn=ESN(); esn.fit(U,Y)
    pred_n=esn.predict(tr[-100:],len(te))
    pred=sc.inverse_transform(pred_n); true=test[:len(pred)]
    resid=pred-true
    coords=['x','y','z']
    for i,c in enumerate(coords):
        plt.figure(); plt.hist(true[:,i],bins=40,alpha=.5,label=f"True {c}")
        plt.hist(pred[:,i],bins=40,alpha=.5,label=f"Pred {c}")
        plt.title(f"True vs Pred {c}"); plt.legend(); plt.tight_layout()
        plt.savefig(f"plots/true_vs_pred_{c}.png"); plt.close()
        plt.figure(); plt.hist(resid[:,i],bins=40,alpha=.8,label=f"Residual {c}")
        plt.title(f"Residuals {c}"); plt.legend(); plt.tight_layout()
        plt.savefig(f"plots/residual_{c}.png"); plt.close()
    metrics={c:nrmse(true[:,i],pred[:,i]) for i,c in enumerate(coords)}
    json.dump(metrics,open("metrics.json","w"),indent=2)
    print(metrics)

if __name__=="__main__": main()
