import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import randrange
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
y = np.array([1,2,3,4,5,6,7])
d = np.array([1,2,0])
b = np.array([[1,1,1],[2,2,2]])
c = np.array([[True,True,True],[True,True,False]])
e = {'a':1,'b':2,'c':3}
f = ['con1','con2','max1']
for s in f:
    if s.startswith('con'):
        print(s)
