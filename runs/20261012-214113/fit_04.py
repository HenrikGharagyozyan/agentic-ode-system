import scipy.io
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 1. Load data
mat = scipy.io.loadmat('/home/henrik/Projects/agentic-ode-system/data/SNLS80mV.mat')
u_raw = mat['V1'].squeeze().astype(np.float32)
y_raw = mat['V2'].squeeze().astype(np.float32)

# 2. Excerpt and centering
start = 42650
end = start + 4096  # 46746
u = u_raw[start:end]
y = y_raw[start:end]
# subtract means computed on the training part (first 3072 samples)
u -= u[:3072].mean()
y -= y[:3072].mean()

print('Loading data from /home/henrik/Projects/agentic-ode-system/data/SNLS80mV.mat')
print(f'Raw data shape: u {u_raw.shape}, y {y_raw.shape}')
print(f'Excerpt shape: u {u.shape}, y {y.shape}')
print(f'Train means subtracted: u_mean={u[:3072].mean():.6f}, y_mean={y[:3072].mean():.6f}')

# 3. Finite-difference acceleration (target)
dt = 1.0 / 610.35
# velocity
v = np.empty_like(y)
v[0] = (y[1] - y[0]) / dt
v[-1] = (y[-1] - y[-2]) / dt
v[1:-1] = (y[2:] - y[:-2]) / (2 * dt)
# acceleration
a_fd = np.empty_like(y)
a_fd[0] = (v[1] - v[0]) / dt
a_fd[-1] = (v[-1] - v[-2]) / dt
a_fd[1:-1] = (v[2:] - v[:-2]) / (2 * dt)
a_fd = a_fd.astype(np.float32)
print(f'Computed finite-difference acceleration, range [{a_fd.min():.3f}, {a_fd.max():.3f}]')

# 4. Prepare training tensors (inputs & normalized targets)
Ntrain = 3072
y_train = y[:Ntrain]
v_train = v[:Ntrain]
u_train = u[:Ntrain]
a_train = a_fd[:Ntrain]

# input normalisation (training statistics)
y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
v_mean, v_std = v_train.mean(), v_train.std() + 1e-8
u_mean, u_std = u_train.mean(), u_train.std() + 1e-8

def norm(x, m, s): return (x - m) / s
y_n = norm(y_train, y_mean, y_std)
v_n = norm(v_train, v_mean, v_std)
u_n = norm(u_train, u_mean, u_std)

X_train = torch.from_numpy(np.stack([y_n, v_n, u_n], axis=1))  # (Ntrain,3)

# target normalisation
a_mean, a_std = a_train.mean(), a_train.std() + 1e-8
a_norm = (a_train - a_mean) / a_std
a_target = torch.from_numpy(a_norm)  # (Ntrain,)

print(f'Training set: X_train {X_train.shape}, a_target {a_target.shape}')

# 5. Define the MLP
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = Net()
print('Model architecture:')
print(model)

# 6. Training loop
torch.set_num_threads(4)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5)  # no verbose flag
best_val = float('inf')
wait = 0
max_steps = 4000
batch_size = 512

print('Starting training...')
for step in range(max_steps):
    idx = torch.randint(0, Ntrain, (batch_size,))
    xb = X_train[idx]
    ab = a_target[idx]
    pred = model(xb)
    loss = nn.functional.mse_loss(pred, ab)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # validation on last 256 training samples
    with torch.no_grad():
        val_idx = torch.arange(Ntrain-256, Ntrain)
        xv = X_train[val_idx]
        av = a_target[val_idx]
        val_loss = nn.functional.mse_loss(model(xv), av).item()
    scheduler.step(val_loss)

    if val_loss < best_val - 1e-6:
        best_val = val_loss
        wait = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        wait += 1
        if wait >= 80:
            print(f'Early stop at step {step}')
            break

    if step % 50 == 0:
        print(f'Step {step}: loss {loss.item():.6f}, val {val_loss:.6f}')

print('Training finished.')

# 7. Load best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 8. Free-run simulation (RK4) – using un-normalized acceleration
def accel_raw(y_val, v_val, u_val):
    # normalise inputs with the same statistics used in training
    y_n = (y_val - y_mean) / y_std
    v_n = (v_val - v_mean) / v_std
    u_n = (u_val - u_mean) / u_std
    with torch.no_grad():
        a_norm = model(torch.tensor([y_n, v_n, u_n], dtype=torch.float32)).item()
    # un-normalize to get real acceleration
    return a_norm * a_std + a_mean

N = len(y)
y_pred = np.empty(N, dtype=np.float32)
v_pred = np.empty(N, dtype=np.float32)

# initial conditions
y_pred[0] = y[0]
v_pred[0] = (y[1] - y[0]) / dt

print('Running free-run simulation (RK4)...')
for n in range(N-1):
    yn = y_pred[n]
    vn = v_pred[n]
    un = u[n]

    # RK4
    k1_y = vn
    k1_v = accel_raw(yn, vn, un)

    k2_y = vn + 0.5*dt*k1_v
    k2_v = accel_raw(yn + 0.5*dt*k1_y, vn + 0.5*dt*k1_v, un)

    k3_y = vn + 0.5*dt*k2_v
    k3_v = accel_raw(yn + 0.5*dt*k2_y, vn + 0.5*dt*k2_v, un)

    k4_y = vn + dt*k3_v
    k4_v = accel_raw(yn + dt*k3_y, vn + dt*k3_v, un)

    y_pred[n+1] = yn + dt/6.0 * (k1_y + 2*k2_y + 2*k3_y + k4_y)
    v_pred[n+1] = vn + dt/6.0 * (k1_v + 2*k2_v + 2*k3_v + k4_v)

# 9. Compute RMSEs and print
train_rmse = np.sqrt(np.mean((y_pred[:Ntrain] - y[:Ntrain])**2))
test_rmse = np.sqrt(np.mean((y_pred[Ntrain:] - y[Ntrain:])**2))
print(f'FREE_RUN_TRAIN_RMSE: {train_rmse:.6f}')
print(f'FREE_RUN_TEST_RMSE:  {test_rmse:.6f}')

# 10. Optional plot
plt.figure(figsize=(10,4))
plt.plot(y[Ntrain:], label='Measured', alpha=0.7)
plt.plot(y_pred[Ntrain:], label='Predicted', alpha=0.7)
plt.legend()
plt.xlabel('Sample index (test)')
plt.ylabel('Voltage (V)')
plt.title('Free-run prediction on test segment')
plt.tight_layout()
plt.savefig('test_plot.png', dpi=150)
plt.close()
print('Saved test_plot.png')