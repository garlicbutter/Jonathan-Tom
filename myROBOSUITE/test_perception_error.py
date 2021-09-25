import numpy as np
from testing_module import things_to_test, run_test
import matplotlib.pyplot as plt

def main():
    kp_test = np.array([1500, 1500, 15, 150, 150, 150])
    kd_test = np.array([1.5, 1.5, 1, 1, 1, 1])
    perception_error_test = things_to_test('perception_error',
                                            testing_min = np.array([0.0]),
                                            testing_max = np.array([0.001]),
                                            amount_of_tests = 5)

    success_list = []
    run_time_list = []
    inserting_eeff_xy_max_list = []
    inserting_eeff_z_max_list = []
    
    for perception_error in iter(perception_error_test):
        result = run_test(kp_test, kd_test,perception_error)
        success_list.append(result['success'])
        run_time_list.append(result['run_time'])
        inserting_eeff_xy_max_list.append(result['inserting_eeff_xy_max'])
        inserting_eeff_z_max_list.append(result['inserting_eeff_z_max'])    

    # draw some figures with the test results
    data_to_plot = run_time_list

    # drawing
    x = np.array(list(iter(perception_error_test))).flatten()
    y = data_to_plot

    fig = plt.figure(figsize=(10,6))
    plt.plot(x,y)
    plt.xlabel(r'$Perception error$')
    plt.ylabel(r'$Run time$')
    plt.show() 

if __name__ == '__main__':
    main()
