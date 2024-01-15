# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 18:47:53 2024

@author: rikci
"""

from numpy import loadtxt
import matplotlib.pyplot as plt

# load array
x_axis = loadtxt('x_axis.csv', delimiter=',')
y_axis = loadtxt('y_axis.csv', delimiter=',')
# print the array
print(y_axis)

plt.figure(dpi=600)
plt.plot(x_axis, y_axis, 'bo')
plt.ylabel("\mathbb{E} \langle\psi|U X_2 U^{\dag}|\psi\rangle")
plt.xlabel("N_shadows")
plt.grid(axis = 'y', linestyle = '--')
plt.xticks(rotation=70)  
#plt.savefig('D:/Fisica/TESI/Final results/filename.png')
plt.show()