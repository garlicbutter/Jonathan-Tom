import numpy as np
from testing_module import things_to_test, run_test
import matplotlib.pyplot as plt
import pandas as pd  
#######################################################
# go to the bottom section of the code to run this file
#######################################################
def run_episodes(filename):
    # parameters to test
    kp_test = things_to_test('controller_stiffness',
                            testing_min = np.array([100, 100, 500, 100, 100, 50]),
                            testing_max = np.array([2000, 2000, 500, 100, 100, 50]),
                            amount_of_tests = 15)

    kd_test = things_to_test('damping_ratio',
                            testing_min = np.array([1, 1, 2, 5, 5, 5]),
                            testing_max = np.array([10, 10, 2, 5, 5, 5]),
                            amount_of_tests = 15)

    perception_error_test = 0

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
                        result['inserting_eeff_z_max'],
                        result['actuation_torque']])

            print('episode number: {a:3d}, run_time: {b:2.2f}'.format(a=test_num,b=result['run_time']))
            print('          kp: ',end="")
            print(kp)
            print('          kd: ',end="")
            print(kd)
            test_num += 1

    df = pd.DataFrame(data, columns=['kp', 'kd', 'perception error','run time', 'xy error', 'z error','torque'])
    import os.path
    df.to_csv(filename,mode='a',header= not os.path.exists(filename),index=False)
    print('Saved result to ' + filename)

def draw_kp_kd(filename):
    # read file
    df = pd.read_csv(filename)  
    # draw some figures with the test results.
    # Options:
        # run time
        # xy error
        # z error
    data_to_plot = 'run time'
    # drawing
    fig = plt.figure(figsize=(10,6))
    ax = fig.gca(projection='rectilinear')
    which_direction = 2 # 0~5 so you can choose which kp, kd to plot
    df['kp_list'] = df['kp'].apply(lambda x: [float(val) for val in x.replace("[","").replace("]","").split()])
    df['kd_list'] = df['kd'].apply(lambda x: [float(val) for val in x.replace("[","").replace("]","").split()])
    df['kp_plot'] = df['kp_list'].apply(lambda x: x[which_direction])
    df['kd_plot'] = df['kd_list'].apply(lambda x: x[which_direction])
    plt.title('Performance of different impedance parameters')
    plt.xlabel(r'$Stiffness$'+' on direction '+str(which_direction+1))
    plt.ylabel(r'$Damping$'+' on direction '+str(which_direction+1))
    size = df['run time'].apply(lambda x: 30.0 if float(x)>0.1 else 0.0)
    sc = ax.scatter(df['kp_plot'],df['kd_plot'],c=df[data_to_plot],cmap='jet',s=size)
    cb = plt.colorbar(sc)
    cb.set_label(data_to_plot)
    plt.show()

if __name__ == '__main__':
    filename = './results/kp3kd3.csv' # the file to read/write

    # runs the simulation and save result file
    # run_episodes(filename)

    # draw the results from the file
    draw_kp_kd(filename)