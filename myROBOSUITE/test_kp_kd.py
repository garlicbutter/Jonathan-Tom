from math import fabs
import numpy as np
from numpy.core.fromnumeric import reshape
from testing_module import things_to_test, run_test
import matplotlib.pyplot as plt
import pandas as pd  


def main():
    # parameters to test
    kp_test = things_to_test('controller_stiffness',
                            testing_min = np.array([100, 100, 50, 50, 50, 100]),
                            testing_max = np.array([1500, 1500, 50, 150, 150, 150]),
                            amount_of_tests = 3)

    kd_test = things_to_test('damping_ratio',
                            testing_min = np.array([1, 1, 1, 10, 10, 10]),
                            testing_max = np.array([3, 3, 1, 10, 10, 10]),
                            amount_of_tests = 3)

    perception_error_test = 0
    filename = './results/kp_kd_result.csv'

    # testing code (no need to change anything below)
    data = []
    test_num = 1
    for kp in iter(kp_test):
        for kd in iter(kd_test):
            result = run_test(kp, kd, perception_error_test)
            data.append([kp,
                        kd,
                        perception_error_test,
                        result['run_time'],
                        result['inserting_eeff_xy_max'],
                        result['inserting_eeff_z_max']])

            print('episode number: {a:3d}, run_time: {b:2.2f}'.format(a=test_num,b=result['run_time']))
            print('          kp: ',end="")
            print(kp)
            print('          kd: ',end="")
            print(kd)
            test_num += 1

    df = pd.DataFrame(data, columns=['kp', 'kd', 'perception error','run time', 'xy error', 'z error'])
    import os.path
    df.to_csv(filename,mode='a',header= not os.path.exists(filename),index=False)
    print('Saved result to ' + filename)

def draw_kp_kd():
    pass
    # # draw some figures with the test results
    # data_to_plot = run_time_list
    # # drawing
    # fig = plt.figure(figsize=(10,6))
    # ax = fig.gca(projection='rectilinear')
    # plt.xlabel(r'$K_p$')
    # plt.ylabel(r'$K_d$')
    # sc = ax.scatter(kp_plot,kd_plot,c=data_to_plot,cmap='gnuplot')
    # cb = plt.colorbar(sc)
    # cb.set_label('run time')
    # plt.show()

if __name__ == '__main__':
    main()
