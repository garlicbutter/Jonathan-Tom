import matplotlib.pyplot as plt
import numpy as np
from testing_module import things_to_test

kp_test = things_to_test('controller_stiffness',
                        testing_min = np.array([100, 100, 50, 50, 50, 100]),
                        testing_max = np.array([1500, 1500, 50, 150, 150, 150]),
                        amount_of_tests = 5)

kd_test = things_to_test('damping_ratio',
                        testing_min = np.array([1, 1, 1, 10, 10, 10]),
                        testing_max = np.array([3, 3, 1, 10, 10, 10]),
                        amount_of_tests = 5)

x = np.array(list(iter(kp_test)))[:,0]
y = np.array(list(iter(kd_test)))[:,0]
# x = np.fromiter(x_list, dtype=float)
# y = np.fromiter(y_list, dtype=float)
xv, yv = np.meshgrid(x, y)
data  = [10,20,30,40,50,
        10,20,30,40,50,
        10,20,30,40,50,
        10,20,30,40,50,
        10,20,30,40,50]

fig = plt.figure(figsize=(10,6))
ax = fig.gca(projection='rectilinear')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
sc = ax.scatter(xv,yv,c=data,cmap='gnuplot')
cb = plt.colorbar(sc)
cb.set_label('data')
plt.show() 
