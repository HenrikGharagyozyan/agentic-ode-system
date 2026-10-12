import numpy as np
import torch
import scipy.io
import matplotlib.pyplot as plt
import os
import sys

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Parameters
fs = 610.35
dt = 1.0 / fs
start_idx = 42650
excerpt_len = 4096
train_len = 3072
test_len = 1024

# Load data
mat_path = '/home/henrik/Projects/agentic-ode-system/data/SNLS80mV.mat'
print(f'Loading data from {mat_path}')
mat = scipy.io.loadmat(mat_path)
u_raw = mat['V1'].squeeze().astype(np.float32)
y_raw = mat['V2'].squeeze().astype(np.float32)
print(f'Raw data shape: u {u_raw.shape}, y {y_raw.shape}')

# Excerpt
u = u_raw[start_idx:start_idx + excerpt_len]
y = y_raw[start_idx:start_idx + excerpt_len]
print(f'Excerpt shape: u {u.shape}, y {y.shape}')

# Centering using training mean
train_u_mean = u[:train_len].mean()
train_y_mean = y[:train_len].mean()
u = u - train_u_mean
y = y - train_y_mean
print(f'Train means subtracted: u_mean={train_u_mean:.6f}, y_mean={train_y_mean:.6f}')

# Finite-difference velocity and acceleration
v = np.empty_like(y)
v[0] = (y[1] - y[0]) / dt
v[-1] = (y[-1] - y[-2]) / dt
v[1:-1] = (y[2:] - y[:-2]) / (2 * dt)

a_fd = np.empty_like(y)
a_fd[0] = (v[1] - v[0]) / dt
a_fd[-1] = (v[-1] - v[-2]) / dt
a_fd[1:-1] = (v[2:] - v[:-2]) / (2 * dt)
print(f'Computed finite-difference acceleration, range [{a_fd.min():.3f}, {a_fd.max():.3f}]')

# Prepare training tensors
y_train = y[:train_len]
v_train = v[:train_len]
u_train = u[:train_len]
a_train = a_fd[:train_len]

# Normalize inputs using training statistics
def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)

y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
v_mean, v_std = v_train.mean(), y_train.std() + 1e-8  # note: using y_train.std()? Actually should be v_train.std()
# Fix: compute correctly
v_mean, v_std = v_train.mean(), v_train.std() + 1e-8
u_mean, u_std = u_train.mean(), u_train.std() + 1e-8

def norm(x, m, s): return (x - m) / s

y_n = norm(y_train, y_mean, y_std)
v_n = norm(v_train, v_mean, v_std)
u_n = norm(u_train, u_mean, u_std)

X_train = torch.from_numpy(np.stack([y_n, v_n, u_n], axis=1))  # (Ntrain,3)
a_target = torch.from_numpy(a_train)                           # (Ntrain,)

print(f'Training set: X_train {X_train.shape}, a_target {a_target.shape}')

# Define MLP
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = Net()
print(f'Model architecture:\n{model}')

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.5, patience=5, verbose=False)
best_val = float('inf')
wait = 0
max_steps = 2000
batch_size = 256
Ntrain = train_len

print('Starting training...')
for step in range(max_steps):
    # Random batch
    idx = torch.randint(0, Ntrain, (batch_size,))
    xb = X_train[idx]
    ab = a_target[idx]
    pred = model(xb)
    loss = torch.nn.functional.mse_loss(pred, ab)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation on last 256 training samples
    with torch.no_grad():
        val_idx = torch.arange(Ntrain-256, Ntrain)
        xv = X_train[val_idx]
        av = a_target[val_idx]
        val_loss = torch.nn.functional.mse_loss(model(xv), av).item()
    scheduler.step(val_loss)

    if val_loss < best_val - 1e-6:
        best_val = val_loss
        wait = 0
        torch.save(model.state_dict(), 'best_model.pth')
        if step % 50 == 0:
            print(f'Step {step}: loss {loss.item():.6f}, val {val_loss:.6f}')
    else:
        wait += 1
        if wait >= 50:
            print(f'Early stop at step {step}')
            break

print('Training finished.')

# Load best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Free-run simulation using RK4
def accel(y_val, v_val, u_val):
    y_n = (y_val - y_mean) / y_std
    v_n = (v_val - v_mean) / v_std
    u_n = (u_val - u_mean) / u_std
    with torch.no_grad():
        return model(torch.tensor([y_n, v_n, u_n], dtype=torch.float32)).item()

N = excerpt_len
y_pred = np.empty(N, dtype=np.float32)
v_pred = np.empty(N, dtype=np.float32)

# Initial conditions
y_pred[0] = y[0]
v_pred[0] = (y[1] - y[0]) / dt

print('Running free-run simulation (RK4)...')
for n in range(N-1):
    yn = y_pred[n]
    vn = v_pred[n]
    un = u[n]

    k1_y = vn
    k1_v = accel(yn, vn, un)

    k2_y = vn + 0.5*dt*k1_v
    k2_v = accel(yn + 0.5*dt*k1_y, vn + 0.5*dt*k1_v, un)

    k3_y = vn + 0.5*dt*k2_v
    k3_v = accel(yn + 0.5*dt*k2_y, vn + 0.5*dt*k2_v, un)

    k4_y = vn + dt*k3_v
    k4_v = accel(yn + dt*k3_y, vn + dt*k3_v, un)

    y_pred[n+1] = yn + dt/6.0 * (k1_y + 2*k2_y + 2*k3_y + k4_y)
    v_pred[n+1] = vn + dt/6.0 * (k1_v + 2*k2_v + 2*k3_v + k4_v)

# Compute RMSE
train_rmse = np.sqrt(np.mean((y_pred[:train_len] - y[:train_len])**2))
test_rmse = np.sqrt(np.mean((y_pred[train_len:] - y[train_len:])**2))

print(f'FREE_RUN_TRAIN_RMSE: {train_rmse:.6f}')
print(f'FREE_RUN_TEST_RMSE:  {test_rmse:.6f}')

# Optional plot
try:
    plt.figure(figsize=(10,4))
    plt.plot(y[train_len:], label='Measured')
    plt.plot(y_pred[train_len:], label='Predicted')
    plt.xlabel('Sample index (test)')
    plt.ylabel('Voltage (V)')
    plt.title('Free-run prediction on test segment')
    plt.legend()
    plt.tight_layout()
    plt.savefig('test_plot.png')
    plt.close()
    print('Saved test_plot.png')
except Exception as e:
    print(f'Plotting failed: {e}')