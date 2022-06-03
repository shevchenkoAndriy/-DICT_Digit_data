import numpy as np
import matplotlib.pyplot as plt
#
# # Генерация массива случайных чисел
# # с использованием np.random.random
#
# random_nparray = np.random.random(size=10)
#
# print(random_nparray)

# Пример реализации алгоритма моделирования
# методом обратной функции
# экспоненциального распределения
# LAMBDA = 1.5
# noise = [-1/LAMBDA * np.log(random_list[i]) for i in range(SAMPLE_SIZE)]
# График реализации шума
# plt.plot(noise)
# plt.show()
# plt.plot(noise, 'o')
# plt.show()
# Гистограмма и теоретическая плотность распределения шума
# count, bins, ignored = plt.hist(noise, 15, edgecolor='k', density=True,
# label='Гистограмма шума')
# plt.plot(bins, LAMBDA * np.exp(-LAMBDA * bins), linewidth=2, color='r',
# label='Теоретическая плотность распределения шума')
# plt.legend()
# plt.show()
SIZE = 256
random_nparray = np.random.random(SIZE)
noise = np.cbrt((361 * random_nparray ** 2) + 304 * random_nparray + 64)
plt.plot(noise)
plt.show()
plt.plot(noise, 'bo')
plt.show()
plt.hist(noise)
plt.show()
count, bins, ignored = plt.hist(noise, 15, edgecolor='k', density=True,
label='Гистограмма шума')
LAMBDA = 1.5

plt.plot(3/38 * np.sqrt(x))
# plt.legend()
plt.show()
