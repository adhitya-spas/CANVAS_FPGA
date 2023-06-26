from cmath import pi
from matplotlib.pyplot import show
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# functions from this folder
from readFPGA import read_FPGA_input, read_INT_input, quick_compare, flatten, twos_complement
from readFPGA import read_FPGA_fft_debug, read_FPGA_input_lines
from inputstimulus import test_signal, input_chirp, white_noise
from win import get_win
from fftcanvas import canvas_fft
from fftpwr import fft_spec_power, fft_xspec_power
from rebinacc import rebin_likefpga, acc_likefpga
from cfbinavg import rebin_canvas, fix_neg1
from log2compress import spec_compress, xspec_compress
from saveas import saveascsv

# remove output files in path
files = glob.glob('output/*')
for f in files:
    os.remove(f)

# some set up parameters
hi_amp  = 27345             # max amplitude of VHDL sims
mid_amp = 2**11
low_amp = 2**6
fs = 131072.                # sampling freq. in Hz

signal_freq0 = 24e3         # signal freq. 1 in Hz
signal_freq1 = 24e3         # signal freq. 2 in Hz
amp0 = hi_amp               # amplitudes (in ADC units)
amp1 = hi_amp               # amplitudes (in ADC units)
shift0 = 0                  # phase shift in radians
shift1 = 83 * np.pi/180     # phase shift in radians
sample_len = 0.5            # seconds
nFFT = 1024                 # length of FFT
n_acc = 8                   # number of FFTs to accummulate

# STEP 1 -------------------- GENERATE INPUT ----------------------------- 
# get one or two test singals
#channels0_td = test_signal(fs, sample_len, signal_freq0, amp0, shift=shift0, channel_num=0, show_plots=False, save_output=None)
#channels1_td = test_signal(fs, sample_len, signal_freq1, amp1, shift=shift1, channel_num=1, show_plots=False, save_output='hex')
inputs = 'Inputs/'
freq = '_03khz'
phase = "5deg"
file0 = inputs + 'high-high_'+'0deg'+freq+'.txt'
file1 = inputs +  'high-high_' + phase + freq + '.txt' 
channels0_td = read_FPGA_input(file0,signed=True,show_plots=False)
channels1_td = read_FPGA_input(file1,signed=True,show_plots=False)

#--------------------------------------David's code--------------------------------

#################
## Input signal
#################

Npts   = len(channels0_td) #1000

f      = 1e3
time   = np.arange(0,Npts) / Npts * 10
amp    = 5

#################
## Input case 1 (rotate x to y)
#################
input1 = channels0_td #input1 = np.sin(2 * np.pi * time) * amp
input2 = channels1_td #input2 = np.zeros(Npts)
input3 = channels0_td #input3 = np.zeros(Npts)

## Diagnostic 
plt.plot(time, input1)
plt.plot(time, input2)
plt.plot(time, input3)
plt.show()

## Rotation matrix from x to y (90 deg about z, counter-clockwise)
theta         = 270 * np.pi / 180
x_to_y_matrix = np.array( [ [np.cos(theta), -np.sin(theta), 0], 
                            [np.sin(theta),  np.cos(theta), 0], 
                            [            0,              0, 1] ] )

## Input array [N x 3]
input_array = np.zeros([len(input1),3])
input_array[:,0] = input1
input_array[:,1] = input2
input_array[:,2] = input3

## Apply rotatation to each point
output = np.matmul(input_array, x_to_y_matrix)

## Diagnostic 
plt.plot(time, output[:,0])
plt.plot(time, output[:,1])
plt.plot(time, output[:,2])
plt.show()


## Print expected output values 
print('')
print('===================')
print('Test 1 (rotation about z')
print('===================')
print('')
print('Rotation Matrix:')
print(x_to_y_matrix)
print('')
print('Input Amps')
print(np.max(input_array[:,0]))
print(np.max(input_array[:,1]))
print(np.max(input_array[:,2]))
print('')
print('Output Amps')
print(np.max(output[:,0]))
print(np.max(output[:,1]))
print(np.max(output[:,2]))
print ('')
print ('Also: check phase of Y output (should match phase of X input')
print ('')

## Writing into text file for verification - Test 1
file = open('test_1.txt','w')
file.write("I/P 1"+"\t\t\t"+"I/P 2"+"\t\t\t"+"I/P 3"+"\t\t\t"+"O/P 1"+"\t\t\t"+"O/P 2"+"\t\t\t"+"O/P 3"+"\n")
for a in range(0,len(output[:,0])):
    file.write(str(input_array[a,0])+"\t\t\t"+str(input_array[a,1])+"\t\t\t"+str(input_array[a,2])+"\t\t\t"+str(output[a,0])+"\t\t\t"+str(output[a,1])+"\t\t\t"+str(output[a,2])+"\n")
#    file.write(str(input_array[a,0])+"\t\t\t"+str(input_array[a,1])+"\t\t\t"+str(input_array[a,2])+"\t\t\t"+str(output[a,0])+"\t\t\t"+str(output[a,1])+"\t\t\t"+str(output[a,2])+"\n")
file.close()

#################
## Input case 2 (rotate y to z)
#################

## Rotation matrix from y to z (90 deg about x) (90 deg about z, counter-clockwise)
theta         = 90 * np.pi / 180
y_to_z_matrix = np.array( [ [1,             0,              0], 
                            [0, np.cos(theta), -np.sin(theta)], 
                            [0, np.sin(theta),  np.cos(theta)] ] )

## Input array [N x 3]
input_array = np.zeros([len(input1),3])
input_array[:,0] = input2
input_array[:,1] = input1
input_array[:,2] = input3

## Apply rotatation to each point
output2 = np.matmul(input_array, y_to_z_matrix)

## Diagnostic 
plt.plot(time, output2[:,0])
plt.plot(time, output2[:,1])
plt.plot(time, output2[:,2])
plt.show()


## Print expected output values 
print('')
print('===================')
print('Test 2 (rotation about x')
print('===================')
print('')
print('Rotation Matrix:')
print(y_to_z_matrix)
print('')
print('Input Amps')
print(np.max(input_array[:,0]))
print(np.max(input_array[:,1]))
print(np.max(input_array[:,2]))
print('')
print('Ouput Amps')
print(np.max(output2[:,0]))
print(np.max(output2[:,1]))
print(np.max(output2[:,2]))
print ('')
print ('Also: check phase of Y output (should be 180 deg out of phase with  X input')
print ('')

## Writing into text file for verification - Test 2
file = open('test_2.txt','w')
file.write("I/P 1"+"\t\t\t"+"I/P 2"+"\t\t\t"+"I/P 3"+"\t\t\t"+"O/P 1"+"\t\t\t"+"O/P 2"+"\t\t\t"+"O/P 3"+"\n")
for a in range(0,len(output2[:,0])):
    file.write(str(input_array[a,0])+"\t\t\t"+str(input_array[a,1])+"\t\t\t"+str(input_array[a,2])+"\t\t\t"+str(output2[a,0])+"\t\t\t"+str(output2[a,1])+"\t\t\t"+str(output2[a,2])+"\n")
file.close()

#################
## Input case 3 (start with vector in xy plane, 35 deg up from x, rotate about x, then z, then y)
#################

## Input array [N x 3]
input_array = np.zeros([len(input1),3])
input_array[:,0] = input1 # * np.cos(35 / 180 * np.pi)
input_array[:,1] = input1 # * np.sin(35 / 180 * np.pi)
input_array[:,2] = input3


## Rotation matrix about x (35 deg about x)
theta          = (180 + 35) * np.pi / 180
about_x_matrix = np.array( [ [1,             0,              0], 
                            [0, np.cos(theta), -np.sin(theta)], 
                            [0, np.sin(theta),  np.cos(theta)] ] )


## Apply rotatation to each point
output3 = np.matmul(input_array, about_x_matrix)

# ## Diagnostic 
# plt.plot(time, input_array[:,0])
# plt.plot(time, input_array[:,1])
# plt.plot(time, input_array[:,2])
# plt.show ()

# ## Diagnostic 
# plt.plot(time, output3[:,0])
# plt.plot(time, output3[:,1])
# plt.plot(time, output3[:,2])


## Amplitude remaining in xy plane, sqrt(x_out^2 + y_out^2), should be cos(19.20747972534415) * (x^2 + y^2 + z^2)
xamp_i = np.max(input_array[:,0])
yamp_i = np.max(input_array[:,1])
zamp_i = np.max(input_array[:,2])

xamp_o = np.max(output3[:,0])
yamp_o = np.max(output3[:,1])
zamp_o = np.max(output3[:,2])

# print( np.sqrt(xamp_i**2 + yamp_i**2 + zamp_i**2) )
# print( np.sqrt(xamp_o**2 + yamp_o**2 + zamp_o**2) )
# print( '' )

# ## Determine angle of full vector above the xy plane ( ~19.2 deg)
# print(  np.arctan( zamp_o / np.sqrt(xamp_o**2 + yamp_o**2 )) * 180 / np.pi    )
# print( '' )

# ## Check that outputs are right 
# print( np.sqrt(xamp_o**2 + yamp_o**2 ) )
# print( np.sqrt(xamp_o**2 + yamp_o**2 + zamp_o**2) * np.cos(19.20747972534415 * np.pi / 180) )
# print ('')

# ## Print expected outputs 
# print( xamp_o )
# print( yamp_o )
# print( zamp_o )
# print ('')

## Now rotate 15 degrees about z (should not change fraction of vector in xy plane)

## Rotation matrix from x to y (90 deg about z)
theta         = 15 * np.pi / 180
about_z       = np.array( [ [np.cos(theta), -np.sin(theta), 0], 
                            [np.sin(theta),  np.cos(theta), 0], 
                            [            0,              0, 1] ] )

## Apply rotatation to each point
output4 = np.matmul(output3, about_z)

## amplitudes
xamp_i = np.max(output3[:,0])
yamp_i = np.max(output3[:,1])
zamp_i = np.max(output3[:,2])

xamp_o = np.max(output4[:,0])
yamp_o = np.max(output4[:,1])
zamp_o = np.max(output4[:,2])

# ## Check that outputs are right 
# print( np.sqrt(xamp_o**2 + yamp_o**2 ) )
# print( np.sqrt(xamp_o**2 + yamp_o**2 + zamp_o**2) * np.cos(19.20747972534415 * np.pi / 180) )
# print ('')


# ## Print expected output values 
# print( xamp_o )
# print( yamp_o )
# print( zamp_o )
# print ('')


## Now rotate 47 degrees about y 

## Rotation matrix from x to y (90 deg about z)
theta         = 47 * np.pi / 180
about_y       = np.array( [ [ np.cos(theta),  0, np.sin(theta)], 
                            [             0,  1,             0], 
                            [-np.sin(theta),  0, np.cos(theta)] ] )

## Apply rotatation to each point
output5 = np.matmul(output4, about_y)

## amplitudes
xamp_o = np.max(output5[:,0])
yamp_o = np.max(output5[:,1])
zamp_o = np.max(output5[:,2])

# ## Determine angle of full vector above the xy plane ( ~19.2 deg)
# Angle = np.arctan( zamp_o / np.sqrt(xamp_o**2 + yamp_o**2 )) 
# print('Angle out of x-y ', Angle * 180 / np.pi)

# ## Print expected output values 
# print( xamp_o )
# print( yamp_o )
# print( zamp_o )
# print ('')

##################
## Finally, combine about-x and about-z rotations.  Result amps. should match result of sequental rotations
##################

full_3D_rot = np.matmul(about_x_matrix, about_z)
full_3D_rot = np.matmul(full_3D_rot, about_y)

## Input array [N x 3]
input_array = np.zeros([len(input1),3])
input_array[:,0] = input1 # * np.cos(35 / 180 * np.pi)
input_array[:,1] = input1 # * np.sin(35 / 180 * np.pi)
input_array[:,2] = input3

## Apply rotatation to each point
output5 = np.matmul(input_array, full_3D_rot)


## Print expected output values 
print('')
print('===================')
print('Test 3 (x,y,z rotation')
print('===================')
print('')
print('Rotation Matrix:')
print(full_3D_rot)
print('')
print('Input Amps')
print(np.max(input_array[:,0]))
print(np.max(input_array[:,1]))
print(np.max(input_array[:,2]))
print('')
print('Ouput Amps')
print(np.max(output5[:,0]))
print(np.max(output5[:,1]))
print(np.max(output5[:,2]))
print('')
print ('Also: check phase of outputs relative to output x.  Output X and Z should be in phase.') 
print ('Also: check phase of outputs relative to output x.  Output X and Y should be 180 deg out of phase.') 
print ('')


## Diagnostic 
plt.plot(time, input_array[:,0])
plt.plot(time, input_array[:,1])
plt.plot(time, input_array[:,2])
plt.show ()

## Diagnostic 
plt.plot(time, output5[:,0], label = 'x')
plt.plot(time, output5[:,1], label = 'y')
plt.plot(time, output5[:,2], label = 'z')


## Add a key 
leg = plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)

## Writing into text file for verification - Test 3
file = open('test_3.txt','w')
file.write("I/P 1"+"\t\t\t"+"I/P 2"+"\t\t\t"+"I/P 3"+"\t\t\t"+"O/P 1"+"\t\t\t"+"O/P 2"+"\t\t\t"+"O/P 3"+"\n")
for a in range(0,len(output5[:,0])):
    file.write(str(input_array[a,0])+"\t\t\t"+str(input_array[a,1])+"\t\t\t"+str(input_array[a,2])+"\t\t\t"+str(output5[a,0])+"\t\t\t"+str(output5[a,1])+"\t\t\t"+str(output5[a,2])+"\n")
file.close()

#--------------------------------End of David's code-------------------------------
print('Test Completed')


