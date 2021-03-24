import numpy as np
from robosuite.controllers.Impedance_param import kcm_impedance
from scipy.linalg import expm


def ImpedanceEq(x0, th0, x0_d, th0_d, F_int, F0, xm_pose, xmd_pose, dt):
    """
    Impedance Eq: F_int-F0=K(x0-xm)+C(x0_d-xm_d)-Mxm_dd

    Solving the impedance equation for x(k+1)=Ax(k)+Bu(k) where
    x(k+1)=[Xm,thm,Xm_d,thm_d]

    Parameters:
        x0,x0_d,th0,th0_d - desired goal position/orientation and velocity
        F_int - measured force/moments in [N/Nm]
        F0 - desired applied force/moments
        xm_pose - impedance model (updated in a loop) initialized at the initial pose of robot
        A_d, B_d - A and B matrices of x(k+1)=Ax(k)+Bu(k)
    Output:
        X_nex = x(k+1) = [Xm,thm,Xm_d,thm_d]
    """
    K, C, M = kcm_impedance()

    # state space formulation
    # X=[xm;thm;xm_d;thm_d] U=[F_int;M_int;x0;th0;x0d;th0d]
    A_1 = np.concatenate((np.zeros([6, 6], dtype=int), np.identity(6)), axis=1)
    A_2 = np.concatenate((np.dot(-np.linalg.pinv(M), K), np.dot(-np.linalg.pinv(M), C)), axis=1)
    A_temp = np.concatenate((A_1, A_2), axis=0)

    B_1 = np.zeros([6, 18], dtype=int)
    B_2 = np.concatenate((np.linalg.pinv(M), np.dot(np.linalg.pinv(M), K),
                          np.dot(np.linalg.pinv(M), C)), axis=1)
    B_temp = np.concatenate((B_1, B_2), axis=0)

    # discrete state space A, B matrices
    A_d = expm(A_temp * dt)
    B_d = np.dot(np.dot(np.linalg.pinv(A_temp), (A_d - np.identity(A_d.shape[0]))), B_temp)

    # defining goal vector of position/ velocity inside the hole
    X0 = np.concatenate((x0, th0), axis=0).reshape(6, 1)  # const
    X0d = np.concatenate((x0_d, th0_d), axis=0).reshape(6, 1)  # const

    # impedance model xm is initialized to initial position of the EEF and modified by force feedback
    xm = xm_pose[:3].reshape(3, 1)
    thm = xm_pose[3:].reshape(3, 1)
    xm_d = xmd_pose[:3].reshape(3, 1)
    thm_d = xmd_pose[3:].reshape(3, 1)

    # State Space vectors
    X = np.concatenate((xm, thm, xm_d, thm_d), axis=0)  # 12x1 column vector
    zero_arr = np.array([0.0, 0.0, 0.0]).reshape(3, 1)
    # U = np.concatenate((zero_arr, zero_arr, x0, th0, zero_arr, zero_arr), axis=0)
    U = np.concatenate((F0 - F_int, x0, th0, x0_d, th0_d), axis=0).reshape(18, 1)

    # discrete state solution X(k+1)=Ad*X(k)+Bd*U(k)
    X_nex = np.dot(A_d, X) + np.dot(B_d, U)
    # X_nex = np.round(X_nex, 10)

    return X_nex.reshape(12,)
