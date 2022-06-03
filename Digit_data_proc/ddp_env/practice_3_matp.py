import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import kurtosis
from scipy.stats import skew
import math
from math import sqrt, cos, log1p, pi, sin
from random import uniform
#step1
# a, b = 5, 7
# x_uniform = np.random.uniform(a, b, 100000)
# print("Среднее значение : ", np.mean(x_uniform))
#
# print("Дисперсия : ", np.var(x_uniform))
#
# print("Коэффициент асимметрии : ", skew(x_uniform))
#
# print("Коэффициент эксцесса : ", kurtosis(x_uniform))

# plt.hist(x_uniform, 10, edgecolor='k')
# plt.figure()
# plt.title('Линейное распределение', fontsize=12)
# count, bins, ignored = plt.hist(x_uniform, 10, edgecolor='k', density=True)
# plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
# plt.show()
# m = 2 ** 31 - 1
# a = 16807
# c = 1
# seed = 12345
# N = 10000
# lower, upper = 5, 7
#
# line_dist = np.zeros(N)
# line_dist[0] = ((seed * a + 1) % m) / m
# for i in range(1, N):
#     line_dist[i] = (a * line_dist[i - 1] + c) % m
#
# line_dist = list(map(lambda x: (upper - lower) * (x / m) + lower, line_dist))
# plt.hist(line_dist, edgecolor='k')
# plt.show()


mu, sigma = 20, 21
# x_n = np.random.normal(mu, sigma, 100000)
# count, bins, ignored = plt.hist(x_n, 30, edgecolor='k', density=True)
# plt.plot(bins, 1/(sigma * np.sqrt(2*np.pi)) * np.exp(- (bins - mu)**2 / (2 * sigma**2)), linewidth=2, color='r')
# plt.show()
# print("Среднее значение : ", np.mean(x_n))
#
# print("Дисперсия : ", np.var(x_n))
#
# print("Коэффициент асимметрии : ", skew(x_n))
#
# print("Коэффициент эксцесса : ", kurtosis(x_n))
# u1 = uniform(0, 1)
# u2 = uniform(0, 1)
# box_muller_dist = np.zeros(1000)
# for i in box_muller_dist:
#     u1 = uniform(0, 1)
#     print(u1)
#     print(sqrt(log1p(u1) * -2))
#     u2 = uniform(0, 1)
#
#     z0 = sqrt(-2 * log1p(u1)) * cos(2 * pi * u2)
#     box_muller_dist[i] = z0 * sigma + mu


# z1 = sqrt(-2 * log1p(0)) * sin(2 * pi * 1)
# z = z0 * sigma + mu
# plt.hist(box_muller_dist, edgecolor='k')
# plt.show()

np.random.seed(521)
U1 = np.random.uniform(size=1000)
U2 = np.random.uniform(size=1000)
R = np.sqrt(-2 * np.log(U1))
Theta = 2 * np.pi * U2
Z0 = (R * np.cos(Theta) * sigma) + mu
Z1 = (R * np.sin(Theta) * sigma) + mu
# X = (R * np.cos(Theta))
# Y = (R * np.sin(Theta))
fig, (ax1, ax2) = plt.subplots(1, 2)
temp = ax1.hist(Z0)
ax1.set_title("Z0")
temp = ax2.hist(Z1)
ax2.set_title("Z1")
plt.show()

# x_r = np.random.rayleigh(21, 100000)
# count, bins, ignored = plt.hist(x_r, 20, edgecolor='k', density=True)
# plt.show()
# print("Среднее значение : ", np.mean(x_r))
#
# print("Дисперсия : ", np.var(x_r))
#
# print("Коэффициент асимметрии : ", skew(x_r))
#
# print("Коэффициент эксцесса : ", kurtosis(x_r))

# x_p = np.random.poisson(21, 100000)
# count, bins, ignored = plt.hist(x_p, 20, edgecolor='k', density=True)
# plt.show()
# print("Среднее значение : ", np.mean(x_p))
#
# print("Дисперсия : ", np.var(x_p))
#
# print("Коэффициент асимметрии : ", skew(x_p))
#
# print("Коэффициент эксцесса : ", kurtosis(x_p))
