from readFPGA import read_FPGA_input, twos_complement_to_hex, twos_complement
import binascii 
import numpy as np
from datetime import datetime
import pandas as pd
# theta = 270 * np.pi / 180
# SCM_x00 = twos_complement_to_hex(-np.sin(theta))

# #SCM_x00 = SCM_x00.encode('unicode_escape')
# #SCM_x00 = r'{}'.format(SCM_x00)
# #output = "7FFF".encode('unicode_escape')
# text = binascii.unhexlify(SCM_x00)

# to get timestamp
# now = datetime.now()
# date_time = now.strftime("%m%d%Y_%H%M%S")

# theta = 45 * np.pi / 180 # First no rotation

# print(twos_complement_to_hex(-1))
# print(twos_complement_to_hex(1))
# print(twos_complement_to_hex(-0.5))
# print(twos_complement_to_hex(0.5))

# df1 = pd.read_csv("D:\CANVAS_work\Canvas-Algorithm\Canvas_FPGA\HW-output\FPGA-60220713_high-high_5deg_03khz_0ADCLoopback_int.txt", sep="\t")
# print(df1)

# df2 = pd.read_csv("test_1.txt", sep="\t")
# print(df2)

# diff_3 = abs(df1.R3 - df2.R3)
# diff_2 = abs(df1.R2 - df2.R2)
# diff_1 = abs(df1.R1 - df2.R1)


# df = pd.read_csv("D:\CANVAS_work\Canvas-Algorithm\Canvas_FPGA\Inputs\mid_amp_03khz.txt", names=['lines'])
# for i in range(0,1000):
#     print(twos_complement(str(df.lines[i]),16))

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import math

from readFPGA import read_FPGA_input, read_INT_input, quick_compare, flatten, twos_complement
from readFPGA import read_FPGA_fft_debug, read_FPGA_input_lines

# a = twos_complement_to_hex(-1)
# # b = bytes(a,'utf-8')
# print(b)

def twos_complement_dum(value):
    if value < 0:
        c_value = round(value)
        hex_value = hex(c_value)
        hex_value = hex_value.zfill(4)
        # Convert the absolute value to binary and remove the prefix '0b'
        binary_value = bin(abs(c_value))[2:]
        # Pad the binary value with zeros to make it 16 bits long
        binary_value = binary_value.zfill(16)
        # Invert all the bits in the binary value
        inverted_binary_value = ''.join(['1' if b == '0' else '0' for b in binary_value])
        # Convert the inverted binary value to an integer and add 1
        inverted_decimal_value = int(inverted_binary_value, 2) + 1
        # Convert the decimal value to hexadecimal format
        hex_value = hex(inverted_decimal_value)[2:]
        # Pad the hexadecimal value with zeros to make it 4 digits long
        hex_value = hex_value.zfill(4)
        # Add a minus sign to the hexadecimal value
        hex_value = hex_value

        # c_value = round(value * 32768)
        # # If the value is non-negative, just convert it to hexadecimal format
        # hex_value = hex(c_value)[2:]
        # # Pad the hexadecimal value with zeros to make it 4 digits long
        # hex_value = hex_value.zfill(4)
    else:
        c_value = round(value)
        # If the value is non-negative, just convert it to hexadecimal format
        hex_value = hex(c_value)[2:]
        # Pad the hexadecimal value with zeros to make it 4 digits long
        hex_value = hex_value.zfill(4)
    return hex_value

out_folder = 'HW-output'
date_time = '134902'
iterate=24        
df_v = pd.read_csv('Verification-24_06202023_134902.txt', sep="\t")
if not all(num<3 for num in df_v.R1_df):
    print("OVER THE LIMIT!!" * 10)
elif not all(num<3 for num in df_v.R2_df):
    print("OVER THE LIMIT!!" * 10)
elif not all(num<3 for num in df_v.R3_df):
    print("OVER THE LIMIT!!" * 10)
else:
    print("You're safe for now")


twos_complement_dum(5)

# def rotateSCM(fname):
#     x,y = read_FPGA_input_lines(fname, 16, 6, 0, 1)
#     z,u = read_FPGA_input_lines(fname, 16, 6, 2, 3)
#     v,w = read_FPGA_input_lines(fname, 16, 6, 4, 5)

#     Rm = np.array([[1,0,0],[0,1,0],[0,0,1]])

#     for i in range(len(x)-1):
#         xyz = np.array([x[i],y[i],z[i]])
#         uvw = np.matmul(xyz,Rm)
#         print(xyz, uvw, u[i],v[i],w[i])
    
# rotateSCM('FPGA/adc_in_rotate_out.txt')

# print("done")


twos_complement_to_hex(0.1)

matrix_first = np.array( [9, 30, 58] )
                        
# Checking with theta
theta1 = -50 * np.pi / 180
theta2 = 24 * np.pi / 180
theta3 = -153 * np.pi / 180

x_rot = np.array( [ [1,             0,              0], 
                                    [0, np.cos(theta1), -np.sin(theta1)], 
                                    [0, np.sin(theta1),  np.cos(theta1)] ] )

y_rot = np.array( [ [ np.cos(theta2),  0, np.sin(theta2)], 
                                    [             0,  1,             0], 
                                    [-np.sin(theta2),  0, np.cos(theta2)] ] )

z_rot       = np.array( [ [np.cos(theta3), -np.sin(theta3), 0], 
                                    [np.sin(theta3),  np.cos(theta3), 0], 
                                    [            0,              0, 1] ] )

z_rot_FPGA       = np.array( [ [np.cos(theta1), np.sin(theta1), 0], 
                                    [-np.sin(theta1),  np.cos(theta1), 0], 
                                    [            0,              0, 1] ] )

z_rot_360       = np.array( [ [np.cos(theta2), -np.sin(theta2), 0], 
                                    [np.sin(theta2),  np.cos(theta2), 0], 
                                    [            0,              0, 1] ] )

product1 = np.matmul(x_rot,y_rot)
product2 = np.matmul(product1,z_rot)
product3 = np.matmul(matrix_first,product2)

final_matrix1 = np.matmul(product1,y_rot)

multi_matrix = np.matmul(x_rot,y_rot)
final_matrix2 = np.matmul(matrix_first,multi_matrix)

# Checking with 360 - theta
theta3 = 180 - 25
theta4 = 180 - 43

x_rot2 = np.array( [ [1,             0,              0], 
                                    [0, np.cos(theta3), -np.sin(theta3)], 
                                    [0, np.sin(theta3),  np.cos(theta3)] ] )

y_rot2 = np.array( [ [ np.cos(theta4),  0, np.sin(theta4)], 
                                    [             0,  1,             0], 
                                    [-np.sin(theta4),  0, np.cos(theta4)] ] )

product3 = np.matmul(matrix_first,x_rot2)
final_matrix3 = np.matmul(product3,y_rot2)

multi_matrix2 = np.matmul(x_rot2,y_rot2)
final_matrix4 = np.matmul(matrix_first,multi_matrix2)

# Checking how multiplying data individually with matrix behaves
# It should only give a -ve answer, instead it gives different value

# it's not the opposite

# It might be 180 - theta

# print(product1)

print("done")

# a = twos_complement_to_hex(0)
# b = bytes(a,'utf-8')
# print(b)
# b = np.array([[int(a[0]),int(a[1])],[int(a[2]),int(a[3])]])
# print(b)
# print(b.tobytes)
# c = b.tobytes
SCM_x00 = binascii.unhexlify(twos_complement_to_hex(np.cos(theta)))
SCM_x01 = binascii.unhexlify(twos_complement_to_hex(-np.sin(theta)))
SCM_x02 = binascii.unhexlify(twos_complement_to_hex(0)) 
SCM_y10 = binascii.unhexlify(twos_complement_to_hex(np.sin(theta)))
SCM_y11 = binascii.unhexlify(twos_complement_to_hex(np.cos(theta)))
SCM_y12 = binascii.unhexlify(twos_complement_to_hex(0)) 
SCM_z20 = binascii.unhexlify(twos_complement_to_hex(0)) 
SCM_z21 = binascii.unhexlify(twos_complement_to_hex(0)) 
SCM_z22 = binascii.unhexlify(twos_complement_to_hex(1)) 
SCM_xoff = binascii.unhexlify(twos_complement_to_hex(0))
SCM_yoff = binascii.unhexlify(twos_complement_to_hex(0))
SCM_zoff = binascii.unhexlify(twos_complement_to_hex(0))

print("FPGA_matrix_sent")
print(SCM_x00)
print(SCM_x01)
print(SCM_x02)
print(SCM_y10)
print(SCM_y11)
print(SCM_y12)
print(SCM_z20)
print(SCM_z21)
print(SCM_z22)
print("\n")

print("done")
print(twos_complement_to_hex(-1))
print(twos_complement("8000",16))

print(twos_complement_to_hex(1))
print(twos_complement("7FFF",16))
print("done")



theta = 270 * np.pi / 180  # TEST 1: # Rotation matrix from x to y (90 deg about z, counter-clockwise)
SCM_x00 = binascii.unhexlify(twos_complement_to_hex(np.cos(theta)))
SCM_x01 = binascii.unhexlify(twos_complement_to_hex(-np.sin(theta)))
SCM_x02 = binascii.unhexlify(twos_complement_to_hex(0)) 
SCM_y10 = binascii.unhexlify(twos_complement_to_hex(np.sin(theta)))
SCM_y11 = binascii.unhexlify(twos_complement_to_hex(np.cos(theta)))
SCM_y12 = binascii.unhexlify(twos_complement_to_hex(0)) 
SCM_z20 = binascii.unhexlify(twos_complement_to_hex(0)) 
SCM_z21 = binascii.unhexlify(twos_complement_to_hex(0)) 
SCM_z22 = binascii.unhexlify(twos_complement_to_hex(1)) 
SCM_xoff = binascii.unhexlify(twos_complement_to_hex(0))
SCM_yoff = binascii.unhexlify(twos_complement_to_hex(0))
SCM_zoff = binascii.unhexlify(twos_complement_to_hex(0))

print("FPGA_matrix_sent")
print(SCM_x00)
print(SCM_x01)
print(SCM_x02)
print(SCM_y10)
print(SCM_y11)
print(SCM_y12)
print(SCM_z20)
print(SCM_z21)
print(SCM_z22)
print("\n")

theta         = 270 * np.pi / 180
x_to_y_matrix = np.array( [ [np.cos(theta), -np.sin(theta), 0], 
                            [np.sin(theta),  np.cos(theta), 0], 
                            [            0,              0, 1] ] )

SCM_x00 = binascii.unhexlify(twos_complement_to_hex(x_to_y_matrix[0][0]))
SCM_x01 = binascii.unhexlify(twos_complement_to_hex(x_to_y_matrix[0][1]))
SCM_x02 = binascii.unhexlify(twos_complement_to_hex(x_to_y_matrix[0][2]))
SCM_y10 = binascii.unhexlify(twos_complement_to_hex(x_to_y_matrix[1][0]))
SCM_y11 = binascii.unhexlify(twos_complement_to_hex(x_to_y_matrix[1][1]))
SCM_y12 = binascii.unhexlify(twos_complement_to_hex(x_to_y_matrix[1][2]))
SCM_z20 = binascii.unhexlify(twos_complement_to_hex(x_to_y_matrix[2][0]))
SCM_z21 = binascii.unhexlify(twos_complement_to_hex(x_to_y_matrix[2][1]))
SCM_z22 = binascii.unhexlify(twos_complement_to_hex(x_to_y_matrix[2][2]))

print("Model_result_matrix_sent")
print(SCM_x00)
print(SCM_x01)
print(SCM_x02)
print(SCM_y10)
print(SCM_y11)
print(SCM_y12)
print(SCM_z20)
print(SCM_z21)
print(SCM_z22)

print("done")