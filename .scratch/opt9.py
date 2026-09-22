import numpy as np
from scipy.optimize import brentq, minimize

def kh_of(G, theta, beta, d, e):
    c = 1.0 - 4*d - 4*e
    xi = G*np.cos(theta); eta = G*np.sin(theta)
    cx, cy = np.cos(xi), np.cos(eta)
    num = -2.0*((1-beta)*(cx+cy-2.0) + beta*(cx*cy-1.0))
    den = c + 2*d*(cx+cy) + 4*e*cx*cy
    return np.sqrt(np.maximum(num/den, 0.0))

def vratio_grid(beta, d, e, Gs, thetas):
    G, T = np.meshgrid(Gs, thetas, indexing='ij')
    return kh_of(G, T, beta, d, e)/G

def worst(p, Gmax=2*np.pi/8, nG=41, nT=25):
    beta, d, e = p
    Gs = np.linspace(1e-3, Gmax, nG); Ts = np.linspace(0.0, np.pi/4, nT)
    v = vratio_grid(beta, d, e, Gs, Ts)
    return np.max(np.abs(v-1.0))

# 5-point reference: beta=0, d=0, e=0
print("5pt  worst over band to 8ppw :", worst((0.0,0.0,0.0)))
print("4th-order compact (b=1/3,d=1/12,e=0):", worst((1/3,1/12,0.0)))
print("JSS  (a=.5461 -> b=.4539, c=.6248, d=(1-c)/4, e=0):", worst((1-0.5461, (1-0.6248)/4, 0.0)))

best=None
rng=np.random.default_rng(0)
for _ in range(400):
    x0 = np.array([rng.uniform(0.0,0.8), rng.uniform(0.0,0.2), rng.uniform(0.0,0.1)])
    r = minimize(worst, x0, method='Nelder-Mead',
                 options=dict(xatol=1e-12, fatol=1e-16, maxiter=20000, maxfev=20000))
    if r.success and (best is None or r.fun < best.fun): best = r
print("optimum:", best.x, best.fun)
b,d,e = best.x
print("c =", 1-4*d-4*e)
