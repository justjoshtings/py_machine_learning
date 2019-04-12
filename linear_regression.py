from statistics import mean
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg') #Python MacOS backend ImportError fix
from matplotlib import pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

def create_dataset(n, variance, step=1.5, correlation=False):
	'''
	Fn to create a random dataset for testing algorithm purposes.
	Returns numpy arrays of X and Y values.
	'''
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

def calc_slope_intercept(xs, ys):
	'''
	Fn to calculate the slope and intercepts of our dataset.
	Returns the slope and intercept.
	'''
	m = ((mean(xs)*mean(ys)) - (mean(xs*ys)))/((mean(xs)*mean(xs)) - mean(xs*xs))
	b = mean(ys) - m*mean(xs)
	return m, b

def squared_error(ys, regression_line):
	'''
	Fn to calculate the squared error of our dataset.
	Return the squared error value.
	'''
	return sum((regression_line - ys)**2)

def coeff_of_determination(ys, regression_line):
	'''
	Fn to calculate the Coefficienct of Determination or more commonly known as R value for our dataset.
	Return R value.
	'''
	y_mean_line = [mean(ys) for y in ys]
	squared_error_regression = squared_error(ys, regression_line)
	squared_error_y_mean = squared_error(ys, y_mean_line)
	return 1 - (squared_error_regression/squared_error_y_mean)

'''
Initialize and run sample dataset and linear regression algorithm.
Run a sample prediction.
Plot dataset, best fit line, and prediction.
'''
xs, ys = create_dataset(50, 20, correlation='pos')

m, b = calc_slope_intercept(xs, ys)

regression_line = [(m*x)+b for x in xs]

predict_x = 7
predict_y = (m*predict_x)+b

r_squared = coeff_of_determination(ys, regression_line)
print('The R value is: {:.2f}'.format(r_squared))

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g', s=100)
plt.plot(xs, regression_line)
plt.show()

