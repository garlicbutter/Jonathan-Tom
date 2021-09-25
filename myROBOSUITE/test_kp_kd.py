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
    which_kp_kd = 0 # choose an index from 0~5

    success_list = []
    run_time_list = []
    inserting_eeff_xy_max_list = []
    inserting_eeff_z_max_list = []
    kp_plot = []
    kd_plot = []
    test_num = 1

    for kp in iter(kp_test):
        for kd in iter(kd_test):
            result = run_test(kp, kd, perception_error_test)
            print('episode number: {a:3d}, run_time: {b:2.4f}, kp: {c:2.4f}, kd: {d:2.4f}'.format(a=test_num,b=result['run_time'],c=kp[which_kp_kd],d=kd[which_kp_kd]))
            test_num += 1
            if result['success']:
                run_time_list.append(result['run_time'])
                inserting_eeff_xy_max_list.append(result['inserting_eeff_xy_max'])
                inserting_eeff_z_max_list.append(result['inserting_eeff_z_max'])
                kp_plot.append(kp[which_kp_kd])
                kd_plot.append(kd[which_kp_kd])

        
    # draw some figures with the test results
    data_to_plot = run_time_list

    # drawing
    fig = plt.figure(figsize=(10,6))
    ax = fig.gca(projection='rectilinear')
    plt.xlabel(r'$K_p$')
    plt.ylabel(r'$K_d$')
    sc = ax.scatter(kp_plot,kd_plot,c=data_to_plot,cmap='gnuplot')
    cb = plt.colorbar(sc)
    cb.set_label('run time')
    plt.show() 

if __name__ == '__main__':
    main()
