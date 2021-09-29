import numpy as np
from testing_module import things_to_test, run_test
import matplotlib.pyplot as plt
import pandas as pd 
#######################################################
# go to the bottom section of the code to run this file
#######################################################
def run_episodes(filename):
    kp_test = np.array([700, 700, 50, 50, 50, 100])
    kd_test = np.array([1.5, 1.5, 1, 10, 10, 10])
    perception_error_test = things_to_test('perception_error',
                                            testing_min = np.array([0.0]),
                                            testing_max = np.array([0.001]),
                                            amount_of_tests = 20)

    # testing code (no need to change anything below)
    data = []
    test_num = 1
    for perception_error in iter(perception_error_test):
        result = run_test(kp_test, kd_test, perception_error)
        data.append([kp_test,
                    kd_test,
                    perception_error,
                    result['run_time'],
                    result['inserting_eeff_xy_max'],
                    result['inserting_eeff_z_max']])

        print('episode number: {a:3d}, run_time: {b:2.2f}'.format(a=test_num,b=result['run_time']))
        print('          kp: ',end="")
        print(kp_test)
        print('          kd: ',end="")
        print(kd_test)
        test_num += 1

    df = pd.DataFrame(data, columns=['kp', 'kd', 'perception error','run time', 'xy error', 'z error'])
    import os.path
    df.to_csv(filename,mode='a',header= not os.path.exists(filename),index=False)
    print('Saved result to ' + filename)

def draw_perception_error(filename):
   # read file
    df = pd.read_csv(filename)  
    # draw some figures with the test results.
    # Options:
        # run time
        # xy error
        # z error
    data_to_plot = 'z error'
    # drawing
    x = df['perception error'].apply(lambda x: float(x.replace("[","").replace("]",""))*1000)
    y = df[data_to_plot]

    fig = plt.figure(figsize=(10,6))
    plt.plot(x,y)
    plt.xlabel('Perception error [mm]')
    plt.ylabel(data_to_plot)
    plt.grid()
    plt.show() 

if __name__ == '__main__':
    filename = './results/percp09291458.csv' # the file to read/write

    # runs the simulation and save result file
    run_episodes(filename)

    # draw the results from the file
    draw_perception_error(filename)