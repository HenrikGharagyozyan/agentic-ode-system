import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Settings
torch.set_num_threads(4)
device = torch.device('cpu')
torch.manual_seed(0)
np.random.seed(0)

# Load data
mat = scipy.io.loadmat('/home/henrik/Projects/agentic-ode-system/data/SNLS80mV.mat')
u_raw = mat['V1'].flatten()
y_raw = mat['V2'].flatten()
start = 42650
end = 46746   # exclusive
u = u_raw[start:end]
y = y_raw[start:end]
# Center
u = u - u.mean()
y = y - y.mean()

# Split
N_train = 3072
u_train, y_train = u[:N_train], y[:N_train]
u_test, y_test = u[N_train:], y[N_train:]

# Finite difference acceleration (training only)
dt = 1.0 / 610.35
# central difference, ignore first and last sample
acc_meas = (y_train[2:] - 2*y_train[1:-1] + y_train[:-2]) / (dt**2)
# align states: y_k, v_k for k = 1 .. N_train-2
y_mid = y_train[1:-1]
v_mid = (y_train[2:] - y_train[:-2]) / (2*dt)

# Normalise inputs (using training statistics only)
def norm_np(x, mu=None, sigma=None):
    if mu is None: mu = x.mean()
    if sigma is None: sigma = x.std() + 1e-8
    return (x - mu) / sigma, mu, sigma

y_n, y_mu, y_sig = norm_np(y_mid)
v_n, v_mu, v_sig = norm_np(v_mid)
u_n, u_mu, u_sig = norm_np(u_train[1:-1])   # same indices as y_mid

# Convert to torch float32
y_t = torch.from_numpy(y_n.astype(np.float32)).to(device)
v_t = torch.from_numpy(v_n.astype(np.float32)).to(device)
u_t = torch.from_numpy(u_n.astype(np.float32)).to(device)
acc_t = torch.from_numpy(acc_meas.astype(np.float32)).to(device)

# Physics-constrained network
class PhysicsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)   # weights = [-a, -b, -c, g]
        self.res = nn.Sequential(
            nn.Linear(4, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.1)
        for p in self.res.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.01)
            else:
                nn.init.zeros_(p)

    def forward(self, y, v, u):
        y3 = y**3
        phi = torch.stack([v, y, y3, u], dim=-1)   # (...,4)
        acc = self.linear(phi)
        acc = acc + self.res(phi)
        return acc.squeeze(-1)

model = PhysicsNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_steps = 2000
for step in range(num_steps):
    model.train()
    optimizer.zero_grad()
    pred = model(y_t, v_t, u_t)
    loss = criterion(pred, acc_t)
    loss.backward()
    optimizer.step()

    if step % 200 == 0 or step == num_steps-1:
        model.eval()
        with torch.no_grad():
            # initial state from first sample of the whole excerpt
            y0 = torch.tensor((y[0] - y_mu) / y_sig, dtype=torch.float32, device=device)
            v0 = torch.tensor(((y[1] - y[0]) / dt - v_mu) / v_sig, dtype=torch.float32, device=device)
            state = torch.stack([y0, v0])
            preds = [y0.item() * y_sig + y_mu]   # de-normalised y0
            for k in range(len(u)-1):
                uk = (u[k] - u_mu) / u_sig
                yk, vk = state
                def f(state_vec, u_in):
                    y_, v_ = state_vec
                    acc_ = model(y_.view(1), v_.view(1), torch.tensor(u_in, dtype=torch.float32, device=device).view(1))
                    return torch.stack([v_, acc_.squeeze()])
                k1 = f(state, uk)
                k2 = f(state + 0.5*dt*k1, uk)
                k3 = f(state + 0.5*dt*k2, uk)
                k4 = f(state + dt*k3, uk)
                state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
                preds.append(state[0].item() * y_sig + y_mu)
            preds = np.array(preds, dtype=np.float32)
            # Compute RMSE on training and test segments (as defined)
            train_rmse = np.sqrt(np.mean((preds[0:N_train] - y_train)**2))
            test_rmse = np.sqrt(np.mean((preds[N_train:] - y_test)**2))
            print(f'step {step:4d} loss {loss.item():.6f}  train_RMSE {train_rmse:.6f}  test_RMSE {test_rmse:.6f}')

# Final free-run test (as required by the problem statement)
model.eval()
with torch.no_grad():
    y0 = torch.tensor((y[0] - y_mu) / y_sig, dtype=torch.float32, device=device)
    v0 = torch.tensor(((y[1] - y[0]) / dt - v_mu) / v_sig, dtype=torch.float32, device=device)
    state = torch.stack([y0, v0])
    y_pred = [y0.item() * y_sig + y_mu]
    for k in range(len(u)-1):
        uk = (u[k] - u_mu) / u_sig
        yk, vk = state
        def f(state_vec, u_in):
            y_, v_ = state_vec
            acc_ = model(y_.view(1), v_.view(1), torch.tensor(u_in, dtype=torch.float32, device=device).view(1))
            return torch.stack([v_, acc_.squeeze()])
        k1 = f(state, uk)
        k2 = f(state + 0.5*dt*k1, uk)
        k3 = f(state + 0.5*dt*k2, uk)
        k4 = f(state + dt*k3, uk)
        state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        y_pred.append(state[0].item() * y_sig + y_mu)
    y_pred = np.array(y_pred, dtype=np.float32)

train_rmse_final = np.sqrt(np.mean((y_pred[0:N_train] - y_train)**2))
test_rmse_final = np.sqrt(np.mean((y_pred[N_train:] - y_test)**2))

print(f'FREE_RUN_TRAIN_RMSE: {train_rmse_final:.6f}')
print(f'FREE_RUN_TEST_RMSE:  {test_rmse_final:.6f}')