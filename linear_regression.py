from statistics import mean
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg') #Python MacOSX backend ImportError fix
from matplotlib import pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# xs = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
# ys = np.array([3, 2, 5, 7, 5, 7, 9, 5], dtype=np.float64)

def create_dataset(n, variance, step=1.5, correlation=False):
	val = 0
	ys = []
	for i in range(n):
		y = val+random.randrange(-variance, variance)
		ys.append(y)
		if correlation and correlation == 'pos':
			val+=step
		elif correlation and correlation == 'neg':
			val-=step
	xs = [i for i in range(len(ys))]

	return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
# plt.scatter(xs, ys)
# plt.show()

def calc_slope_intercept(xs, ys):
	m = ((mean(xs)*mean(ys)) - (mean(xs*ys)))/((mean(xs)*mean(xs)) - mean(xs*xs))
	b = mean(ys) - m*mean(xs)
	return m, b

def squared_error(ys, regression_line):
	e_square = sum((regression_line - ys)**2)
	return e_square

def coeff_of_determination(ys, regression_line):
	y_mean_line = [mean(ys) for y in ys]
	squared_error_regression = squared_error(ys, regression_line)
	squared_error_y_mean = squared_error(ys, y_mean_line)
	return 1 - (squared_error_regression/squared_error_y_mean)


xs, ys = create_dataset(50, 20, correlation=False)

m, b = calc_slope_intercept(xs, ys)

# print(m, b)

regression_line = [(m*x)+b for x in xs]

predict_x = 4.5
predict_y = (m*predict_x)+b

r_squared = coeff_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g', s=100)
plt.plot(xs, regression_line)
plt.show()

