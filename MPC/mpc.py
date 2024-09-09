import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Clear and initialize
# (Python doesn't need explicit clearing)

# Create system, initialize parameters
steps = 100  # simulation steps
ts = 0.01  # sampling period

ad = np.array([
    [1.00027180509492, 0.00625101658363726, -0.000298104527325984, -0.000592137149727941, -0.000195218555764740],
    [-0.00625101658365004, 0.879670596217866, 0.0123995907573806, 0.00942892684037583, -0.00775386215642799],
    [-0.000298104527325549, -0.0123995907573839, 0.999169855139624, -0.0148759276100900, 0.000129671924415677],
    [0.000592137149728420, 0.00942892684037156, 0.0148759276100894, 0.998913472148301, 0.0286900249744246],
    [-0.000195218555764543, 0.00775386215643324, 0.000129671924425366, -0.0286900249744255, 0.999703452784522]
])

bd = np.array([-0.023307871208778, -0.314731276263951, -0.008803109981206, 0.016810972019614, 0.005019051193557]).reshape(-1, 1)
cs = np.array([0.023307871208772, -0.314731276263952, 0.008803109981209, 0.016810972019614, -0.005019051193548])
ds = 0

sys1 = signal.StateSpace(ad, bd, cs, ds, dt=ts)
xs0 = np.zeros(5)

# MPC key parameters
P = 10  # prediction horizon
M = 5  # control horizon
q = 1  # Q matrix weight
r = 10  # R matrix weight
h = 0.5  # H matrix weight
alpha = 0.2  # desired trajectory smoothness (range 0~1), smaller means faster response
target = 1  # target value

# Matrix initialization
A = np.zeros((P, M))  # dynamic matrix
Q = np.eye(P) * q  # Q matrix
R = np.eye(M) * r  # R matrix
H = np.ones((P, 1)) * h  # H matrix
S = np.eye(P)  # shift matrix
S = np.roll(S, -1, axis=1)
S[-1, -1] = 1
DU = np.zeros((M, 1))
W = np.zeros((P, 1))  # desired trajectory
Y0 = np.zeros((P, 1))  # predicted output trajectory
Y_cor = np.zeros((P, 1))  # predicted output trajectory correction

# 1. Model
# 1.1 Get step response model
t, stepresponse = signal.dstep(sys1, t=np.arange(0, P*ts, ts))
stepresponse = stepresponse[0].flatten()

# 1.2 Create dynamic matrix A, size P*M
A[:, 0] = stepresponse[:P]
for i in range(0, P):
    for j in range(1, M):
        if i >= j:
            A[i, j] = A[i-1, j-1]

# 2. Prediction
xs1 = ad @ xs0
y = [cs @ xs0]
u = [0]
ref = []

for k in range(1, 3*steps):
    xs1 = ad @ xs0 + bd.flatten() * u[-1]
    y.append(cs @ xs0 + ds * u[-1])
    xs0 = xs1

    if k < steps:
        target = 1
    elif k - steps < steps:
        target = -1
    else:
        target = 1
    ref.append(target)

    # Reference trajectory
    W = np.array([alpha**i * y[-1] + (1 - alpha**i) * target for i in range(1, P+1)]).reshape(-1, 1)

    # Error compensation, trajectory correction
    Y_cor = Y0 + H * (y[-1] - Y0[0, 0])

    # Shift
    Y0 = S @ Y_cor

    # Calculate incremental control
    Y0 = Y0 + A @ DU

    # Solve for optimal value
    DU = np.linalg.inv(A.T @ Q @ A + R) @ A.T @ Q @ (W - Y0)
    u.append(u[-1] + DU[0, 0])

# Plot results
plt.figure(figsize=(10, 8))

plt.subplot(211)
plt.plot(y, linewidth=2, label='Output')
plt.plot(ref, linewidth=2, label='Reference')
plt.title('System Output')
plt.xlabel('t')
plt.ylabel('y')
plt.ylim(-1.5, 1.5)
plt.legend()
plt.grid(True)

plt.subplot(212)
plt.plot(u, linewidth=2)
plt.title('Control Input')
plt.xlabel('t')
plt.ylabel('u')
plt.grid(True)

plt.tight_layout()
plt.show()
