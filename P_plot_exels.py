import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.figure('1')
x = [2, 3, 4, 5, 7, 10, 12, 15, 18]
org = [207.92, 199.96, 200.37, 196.78, 195.52, 189.79, 187.59, 186.88, 189.20]
pro = [199.20, 198.30, 195.74, 199.52, 195.86, 192.78, 189.64, 187.51, 190.04]
plt.title('Tershold = 1')
plt.plot(x, org, "b")
plt.plot(x, pro, 'r')
plt.xlabel('section before merge')
plt.ylabel('MSE')
plt.savefig('1.png')

plt.figure('3')
x = [2, 3, 4, 5, 10, 15]
org = [208.62, 199.54, 201.11, 197.91, 190.91, 186.10]
pro = [199.84, 197.76, 196.63, 200.31, 192.76, 186.70]
plt.title('Tershold = 3')
plt.plot(x, org, "b")
plt.plot(x, pro, 'r')
plt.xlabel('section before merge')
plt.ylabel('MSE')
plt.savefig('3.png')

plt.figure('7')
x = [2, 3, 4, 5, 10, 15]
org = [207.93, 199.80, 200.95, 197.04, 190.25, 186.36]
pro = [200.42, 198.27, 196.65, 200.08, 191.69, 187.04]
plt.title('Tershold = 7')
plt.plot(x, org, "b")
plt.plot(x, pro, 'r')
plt.xlabel('section before merge')
plt.ylabel('MSE')
plt.savefig('7.png')
plt.figure('10')

x = [2, 3, 4, 5, 10, 15]
org = [207.76, 200.15, 200.96, 196.99, 190.37, 186.20]
pro = [197.95, 198.50, 196.4, 200.04, 193.39, 186.63]
plt.title('Tershold = 10')
plt.plot(x, org, "b")
plt.plot(x, pro, 'r')
plt.xlabel('section before merge')
plt.ylabel('MSE')
plt.savefig('10.png')
