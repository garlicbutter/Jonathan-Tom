import numpy as np

# parameters
L1, L2 = 0.4, 0.3  # meters
m1, m2, m_end = 8, 5, 4  # kg
K_e = 10**5 # N/m

# F = Kx + Bx' + Mx''
K_imp = [[62500.0, 0.0],[0.0, 62500.0]]
B_imp = [[3500.0, 0.0],[0.0, 3500.0]]
M_imp = [[100.0, 0.0],[0.0, 100.0]]
