from cmath import pi
from matplotlib.pyplot import show
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

from readFPGA import twos_complement, int_to_twos_complement

number = 0.033  

num_hex = int_to_twos_complement(number)

print(num_hex)

num_hex2 = '0384'
num = twos_complement(num_hex,16)
print(num)