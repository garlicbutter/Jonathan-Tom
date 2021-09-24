import numpy as np
from testing_module import things_to_test, run_test

def main():
    kp_test = np.array([100, 100, 50, 50, 50, 100])
    kd_test = np.array([3, 3, 1, 10, 10, 10])
    perception_error_test = things_to_test('perception_error',
                                            testing_min = np.array([0.0001]),
                                            testing_max = np.array([0.001]),
                                            amount_of_tests = 101)

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
    print(success_list)


if __name__ == '__main__':
    main()
