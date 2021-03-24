import numpy as np
import math


def kcm_impedance():
    # kk = 60
    # mm = 2
    # cc = 0.707 * 2 * math.sqrt(kk * mm)
    # #
    # K = np.identity(6) * kk
    # M = np.identity(6) * mm
    # C = np.identity(6) * cc



    g_pos = 8.51427155e+01 * 10
    g_orient = 2.58588562e+00 * 8
    m = 0.2

    #C = np.zeros([6,6])
    C = np.array([[g_pos, 0, 0, 0, 0, 0],
                  [0, g_pos, 0, 0, 0, 0],
                  [0, 0, g_pos, 0, 0, 0],
                  [0, 0, 0, g_orient, 0, 0],
                  [0, 0, 0, 0, g_orient, 0],
                  [0, 0, 0, 0, 0, g_orient]])

    K = np.array([[g_pos, 0, 0, 0, 0, 0],
                  [0, g_pos, 0, 0, 0, 0],
                  [0, 0, g_pos, 0, 0, 0],
                  [0, 0, 0, g_orient, 0, 0],
                  [0, 0, 0, 0, g_orient, 0],
                  [0, 0, 0, 0, 0, g_orient]])

    M = np.identity(6) * m

    # param = np.array([8.51427155e+01 * 10, 2.58588562e+00 * 8, 7.38883162 * 10, 2.58588562e+00 * 4
    #                      , 4.29988213e+01, 8.96179886e+01, -6.97503967e+01, 2.68183804e+01
    #                      , 5.73403931e+01, -5.17920609e+01, -3.78155098e+01, 5.21090088e+01
    #                      , -6.07303009e+01, -8.36500931e+01, -4.22023392e+01, 1.03413078e+02
    #                      , -1.03323402e+02, 4.82370415e+01, -3.03124313e+01, -6.77349243e+01
    #                      , 8.47555695e+01, -2.32001991e+01, 1.76980610e+01, -1.02977247e+01
    #                      , 8.93070602e+01, 5.74303627e-01, -3.56585464e+01, 1.14796577e+02
    #                      , -2.04232140e+01, 8.16841722e-02, -2.35834045e+01, 1.68239021e+01
    #                      , 5.55183411e+01, -5.98762894e+01, 1.61110473e+00, -5.04463120e+01
    #                      , 4.49372559e+01, -7.15331173e+00, 5.56242981e+01, -2.45204353e+01
    #                      , -6.03688354e+01, 3.72756338e+00, 1.43604660e+01, -7.06865463e+01
    #                      , -2.03724842e+01, -2.37109184e+01, 3.16011982e+01, -6.35860138e+01
    #                      , 3.41194687e+01, 8.08382874e+01, 5.46934485e-01, -4.85310211e+01
    #                      , 7.69228745e+01, -4.80418444e+00, 1.34810419e+01, -8.26619720e+01
    #                      , -9.00178146e+01, 3.23810272e+01, 1.57333031e+01, 3.20296364e+01
    #                      , 8.52591217e-01, 1.18239235e+02, 2.64988861e+01, 2.80290675e+00
    #                      , -1.93128033e+01, 2.88705807e+01, -3.40526428e+01, -7.43630295e+01
    #                      , 1.21558304e+01, 2.02143326e+01, 9.02202415e+00, -2.00483189e+01
    #                      , -2.33711262e+01, -2.63077927e+01, 2.67623672e+01, -8.31557541e+01], dtype=np.float32)
    #
    # K_imp_loc = np.identity(3)
    # C_imp_pos = np.identity(3)
    #
    # K_imp_ori = np.identity(3)
    # C_imp_ori = np.identity(3)
    #
    # K_imp_loc_ori = np.identity(3)
    # C_imp_loc_ori = np.identity(3)
    #
    # K_imp_ori_loc = np.identity(3)
    # C_imp_ori_loc = np.identity(3)
    #
    # [K_imp_loc[0, 0], K_imp_loc[0, 1], K_imp_loc[0, 2], K_imp_loc[1, 0], K_imp_loc[1, 1], K_imp_loc[1, 2],
    #  K_imp_loc[2, 0], K_imp_loc[2, 1], K_imp_loc[2, 2],
    #  C_imp_pos[0, 0], C_imp_pos[0, 1], C_imp_pos[0, 2], C_imp_pos[1, 0], C_imp_pos[1, 1], C_imp_pos[1, 2],
    #  C_imp_pos[2, 0], C_imp_pos[2, 1], C_imp_pos[2, 2],
    #  K_imp_ori[0, 0], K_imp_ori[0, 1], K_imp_ori[0, 2], K_imp_ori[1, 0], K_imp_ori[1, 1], K_imp_ori[1, 2],
    #  K_imp_ori[2, 0], K_imp_ori[2, 1], K_imp_ori[2, 2],
    #  C_imp_ori[0, 0], C_imp_ori[0, 1], C_imp_ori[0, 2], C_imp_ori[1, 0], C_imp_ori[1, 1], C_imp_ori[1, 2],
    #  C_imp_ori[2, 0], C_imp_ori[2, 1], C_imp_ori[2, 2],
    #  K_imp_loc_ori[0, 0], K_imp_loc_ori[0, 1], K_imp_loc_ori[0, 2], K_imp_loc_ori[1, 0], K_imp_loc_ori[1, 1],
    #  K_imp_loc_ori[1, 2], K_imp_loc_ori[2, 0], K_imp_loc_ori[2, 1], K_imp_loc_ori[2, 2],
    #  C_imp_loc_ori[0, 0], C_imp_loc_ori[0, 1], C_imp_loc_ori[0, 2], C_imp_loc_ori[1, 0], C_imp_loc_ori[1, 1],
    #  C_imp_loc_ori[1, 2], C_imp_loc_ori[2, 0], C_imp_loc_ori[2, 1], C_imp_loc_ori[2, 2],
    #  K_imp_ori_loc[0, 0], K_imp_ori_loc[0, 1], K_imp_ori_loc[0, 2], K_imp_ori_loc[1, 0], K_imp_ori_loc[1, 1],
    #  K_imp_ori_loc[1, 2], K_imp_ori_loc[2, 0], K_imp_ori_loc[2, 1], K_imp_ori_loc[2, 2],
    #  C_imp_ori_loc[0, 0], C_imp_ori_loc[0, 1], C_imp_ori_loc[0, 2], C_imp_ori_loc[1, 0], C_imp_ori_loc[1, 1],
    #  C_imp_ori_loc[1, 2], C_imp_ori_loc[2, 0], C_imp_ori_loc[2, 1], C_imp_ori_loc[2, 2],
    #  ] = param[4:]
    #
    # K = np.zeros((6, 6))
    # C = np.zeros((6, 6))
    #
    # K[:3, :3] = K_imp_loc
    # K[:3, 3:6] = K_imp_loc_ori
    # K[3:6, :3] = K_imp_ori_loc
    # K[3:6, 3:6] = K_imp_ori
    #
    # C[:3, :3] = C_imp_pos
    # C[:3, 3:6] = C_imp_loc_ori
    # C[3:6, :3] = C_imp_ori_loc
    # C[3:6, 3:6] = C_imp_ori
    #
    # m = 0.2
    # M = np.identity(6)*m

    return K, C, M
