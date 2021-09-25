import numpy as np
from numpy.core.fromnumeric import reshape
from testing_module import things_to_test, run_test
import matplotlib.pyplot as plt

def main():
    kp_test = things_to_test('controller_stiffness',
                            testing_min = np.array([100, 100, 50, 50, 50, 100]),
                            testing_max = np.array([1500, 1500, 50, 150, 150, 150]),
                            amount_of_tests = 3)

    kd_test = things_to_test('damping_ratio',
                            testing_min = np.array([1, 1, 1, 10, 10, 10]),
                            testing_max = np.array([3, 3, 1, 10, 10, 10]),
                            amount_of_tests = 3)

    perception_error_test = 0

    success_list = []
    run_time_list = []
    inserting_eeff_xy_max_list = []
    inserting_eeff_z_max_list = []

    for kp in iter(kp_test):
        for kd in iter(kd_test):
            result = run_test(kp, kd, perception_error_test)
            success_list.append(result['success'])
            run_time_list.append(result['run_time'])
            inserting_eeff_xy_max_list.append(result['inserting_eeff_xy_max'])
            inserting_eeff_z_max_list.append(result['inserting_eeff_z_max'])
    
    # draw some figures with the test results
    data_to_plot = run_time_list
    which_kp_kd = 0 # choose one from 0~5

    # drawing
    x = np.array(list(iter(kp_test)))[:,which_kp_kd]
    y = np.array(list(iter(kd_test)))[:,which_kp_kd]
    xv, yv = np.meshgrid(x, y)
    fig = plt.figure(figsize=(10,6))
    ax = fig.gca(projection='rectilinear')
    plt.xlabel(r'$K_p$')
    plt.ylabel(r'$K_d$')
    sc = ax.scatter(xv,yv,c=data_to_plot,cmap='gnuplot')
    cb = plt.colorbar(sc)
    cb.set_label('run_time')
    plt.show() 

if __name__ == '__main__':
    main()
