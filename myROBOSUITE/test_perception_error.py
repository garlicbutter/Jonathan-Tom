import numpy as np
from testing_module import things_to_test, run_test

def main():
    kp_test = np.array([100, 100, 50, 50, 50, 100])
    kd_test = np.array([3, 3, 1, 10, 10, 10])
    perception_error_test = things_to_test('perception_error',
                                            testing_min = np.array([0.0001]),
                                            testing_max = np.array([0.001]),
                                            amount_of_tests = 101)
    test_results = np.array([])

    for perception_error in iter(perception_error_test):
        result = run_test(kp_test, kd_test,perception_error)
        test_results = np.append(test_results, result)
    
    # draw some figures with the test results
    pass


if __name__ == '__main__':
    main()
