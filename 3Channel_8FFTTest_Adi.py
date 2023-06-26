from doctest import testmod
from encodings import utf_8
from multiprocessing.connection import wait
import os
import sys
from urllib import response
sys.path.append("..\Canvas-Algorithm") #import functions from parent directory
import serial #import serial library
import time
import numpy as np
from numpy import random
#custom function imports
from saveas import save_output_txt
from serialfcns import readFPGA, ser_write, response_check
from inputstimulus import test_signal
from readFPGA import read_FPGA_input, twos_complement_to_hex
import matplotlib.pyplot as plt
import binascii
from datetime import datetime


# some set up parameters - used in signal generation
fs = 131072.                # sampling freq. in Hz
signal_freq0 = 60000         # signal freq. 1 in Hz
signal_freq1 = 60000         # signal freq. 1 in Hz
amp0 = 2**15-1                # amplitudes (in ADC units)
amp1 = 2**15-1                # amplitudes (in ADC units)
shift0 = 0                  # phase shift in radians
shift1 = 0                  # phase shift in radians
sample_len = 0.5             # seconds
xspec_test = True

#misc PIC commands
ack = b'\x06\x0A'
lf = b'\x0A'
delim = b'\x2C'
complete = b'Ready.\n'
initiated = b'\nInitiating.\n'

#define pic packet headers
SetConfig = b'\x01'
Data = b'\x02'
ResetPIC = b'\x03' #Just this, need to wait ~2 seconds after sending command
StartFPGA = b'\x04'
StopFPGA = b'\x05'
SetLength = b'\x06' #takes payload of uint32

#define pic SetConfig payloads
Ingress_Write = b'\x00'
Ingress_Read = b'\x01'
Ch_0_Pkt_Gen = b'\x02'
ADC_And_Rotation = b'\x03'
FFT_Results = b'\x04'
FFT_Power = b'\x05'
Spec_to_X_Spec_IF = b'\x06'
Spectra_Results = b'\x07'
Power_RAM_port_A = b'\x08'
Power_RAM_port_B = b'\x09'
Real_RAM_port_A = b'\x0A'
Real_RAM_port_B = b'\x0B'
X_Spec_Real_Results = b'\x0C'
Imaginary_RAM_port_A = b'\x0D'
Imaginary_RAM_port_B = b'\x0E'
X_Spec_Imaginary_Results = b'\x0F'

#GSE Commands to FPGA
Sync_Pat = b'\x1A\xCF\xFC\x1D'
Test_Enable = b'\x7E\x57\xBA\x11'
SW_Reset = b'\x7E\x57\xDE\xAD'
Config = b'\x7E\x57\xCF\x16'
OpcodeSetMatrix = b'\x7E\x57\x00\x00'


#Generate input signal from file or aribitrarily
fromFile = True
num = 1

if fromFile:
    inputs = 'Inputs/'
    
    amp = "high-high_"
    phase = "5deg"
    f = "_03khz"
    #file0 = inputs+amp+"0deg"+f+'.txt'
    #file0 = 'Inputs/Increment_counter_input.txt'
    #file1 = 'Inputs/Increment_counter_input.txt'
    #file1 = inputs+amp+phase+f+'.txt' 
    #file2 = inputs+amp+phase+f+'.txt' 
    phase0 = "03khz"
    file0 = inputs+"hi_amp_"+phase0+'.txt'
    phase1 = "10khz"
    file1 = inputs+"hi_amp_"+phase1+'.txt'
    phase2 = "24khz"
    file2 = inputs+"hi_amp_"+phase2+'.txt'
    channels0_td = read_FPGA_input(file0,signed=True,show_plots=False)
    channels1_td = read_FPGA_input(file1,signed=True,show_plots=False)
    channels2_td = read_FPGA_input(file2,signed=True,show_plots=False)
else:
    channels0_td = test_signal(fs, sample_len, signal_freq0, amp0, shift=shift0, channel_num=0, show_plots=False, save_output='both')
    channels1_td = test_signal(fs, sample_len, signal_freq1, amp1, shift=shift1, channel_num=1, show_plots=False, save_output='both')
    channels2_td = test_signal(fs, sample_len, signal_freq1, amp1, shift=shift1, channel_num=1, show_plots=False, save_output='both')
if len(channels0_td) > 20480:
    num_samples = 20480
else:
    num_samples = int(len(channels0_td))
print(num_samples)

test0 = channels0_td[0:num_samples]
test1 = channels1_td[0:num_samples]
test2 = channels2_td[0:num_samples]

#initialize serial ports
pic_ser = serial.Serial("COM5",115200)
pic_ser1 = serial.Serial("COM4",115200)
FPGA_ser = serial.Serial("COM6",115200)


#main loop
# spec_core = b'\x01'
# iterate = 0
# #while iterate < 2:
# #set test mode
# if xspec_test:
#     if iterate == 0:
#         testmode = ADC_And_Rotation #change to ADC_And_Rotation for feedback test
#         readcon = 'none'
#         mode = 'ADC_Rot'
#     else: 
#         #testmode = X_Spec_Imaginary_Results
#         testmode = X_Spec_Imaginary_Results #For testing, change this testmode
#         readcon = 'none' #valid options are 'all' or 'none'. All dumps all data to a file, none proceeds with normal mode
#         mode = 'xspec_imaginary'
# else:
#     iterate+=1
#     testmode = Spectra_Results
#     readcon = 'none'
#     mode = ''

spec_core = b'\x01'
iterate = 0
#set test mode
testmode = ADC_And_Rotation #change to ADC_And_Rotation for feedback test
readcon = 'none'
mode = 'ADC_Rot'

while iterate < 4:

    #reset FPGA
    ser_write(FPGA_ser,Sync_Pat+SW_Reset,False)
    print('FPGA Reset')

    #reset PIC and flush FPGA Serial Port
    ser_write(pic_ser,ResetPIC+lf,True)
    ser_write(pic_ser1,ResetPIC+lf,True)

    FPGA_ser.close()
    time.sleep(0.5)
    FPGA_ser.open()

    #response_check(pic_ser,ack)
    #print('Reset Received')
    response_check(pic_ser,initiated)
    print('PIC0 Reset')
    response_check(pic_ser1,initiated)
    print('PIC1 Reset')

    #Set number of samples to be buffered on PIC0 (2 Channels - A1 and A2)
    to_Send = num_samples.to_bytes(4,'big',signed=False)
    ser_write(pic_ser,SetLength+to_Send+lf)
    response_check(pic_ser,ack) #Wait for acknowledge
    print('Data Length Set')

    #buffer data
    t0=time.perf_counter()
    var = 0
    for i in range(len(test0)):
        val0 = test0[i].to_bytes(2,byteorder='big',signed=True)
        val1 = test1[i].to_bytes(2,byteorder='big',signed=True)
        ser_write(pic_ser,Data + val0 + delim + val1 + lf)
        if var%1000 == 0:
            print('buffering ', var)
        var = var+1
        #response_check(pic_ser,ack)
    response_check(pic_ser,complete) #check for complete from PIC
    del_t = time.perf_counter() - t0
    print('PIC0 Data buffered after %f seconds', del_t)

    #Set number of samples to be buffered on PIC1 (1 Channel - A3)
    to_Send = num_samples.to_bytes(4,'big',signed=False)
    ser_write(pic_ser1,SetLength+to_Send+lf)
    response_check(pic_ser1,ack) #Wait for acknowledge
    print('Data Length Set')

    #buffer data
    t0=time.perf_counter()
    var = 0
    for i in range(len(test0)):
        val0 = test2[i].to_bytes(2,byteorder='big',signed=True)
        val1 = test2[i].to_bytes(2,byteorder='big',signed=True)
        ser_write(pic_ser1,Data + val0 + delim + val1 + lf)
        if var%1000 == 0:
            print('buffering ', var)
        var = var+1
        #response_check(pic_ser,ack)
    response_check(pic_ser1,complete) #check for complete from PIC
    del_t = time.perf_counter() - t0
    print('PIC1 Data buffered after %f seconds', del_t)


    # #configure and start FPGA
    # ser_write(FPGA_ser,Sync_Pat+Config+spec_core+testmode,False)
    # print('FPGA Configured')
    # ser_write(FPGA_ser,Sync_Pat+Test_Enable,False)
    # print('FPGA Started')

    # The Rotation Matrix
    # | SCM_x00  SCM_x01  SCM_x02 |         SCM_xoff 
    # | SCM_y10  SCM_y11  SCM_y12 |   and   SCM_yoff    are the upper and lower triangle defaults
    # | SCM_z20  SCM_z21  SCM_z22 |         SCM_zoff

    # For TEST 1:
    # ## Rotation matrix from x to y (90 deg about z, counter-clockwise)
    # theta         = 270 * np.pi / 180
    # x_to_y_matrix = np.array( [ [np.cos(theta), -np.sin(theta), 0], 
    #                             [np.sin(theta),  np.cos(theta), 0], 
    #                             [            0,              0, 1] ] )

################
# Input case 0 (no rotation)
################
    if iterate==0:
        theta = 0 * np.pi / 180 # First no rotation
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

        ser_write(FPGA_ser,Sync_Pat+OpcodeSetMatrix+SCM_x00+SCM_x01+SCM_x02+SCM_y10+SCM_y11+SCM_y12+SCM_z20+SCM_z21+SCM_z22+SCM_xoff+SCM_yoff+SCM_zoff,False)
    
################
# Input case 1 (rotate x to y)
################
    elif iterate==1:
        theta = (270) * np.pi / 180  # TEST 1: # Rotation matrix from x to y (90 deg about z, counter-clockwise)
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
        
        ser_write(FPGA_ser,Sync_Pat+OpcodeSetMatrix+SCM_x00+SCM_x01+SCM_x02+SCM_y10+SCM_y11+SCM_y12+SCM_z20+SCM_z21+SCM_z22+SCM_xoff+SCM_yoff+SCM_zoff,False)
    
################
# Input case 2 (rotate y to z)
################
    elif iterate==2:
        theta = 90 * np.pi / 180 # TEST 2: # Rotation matrix from y to z (90 deg about x) (90 deg about z, counter-clockwise)
        SCM_x00 = binascii.unhexlify(twos_complement_to_hex(1))
        SCM_x01 = binascii.unhexlify(twos_complement_to_hex(0))
        SCM_x02 = binascii.unhexlify(twos_complement_to_hex(0)) 
        SCM_y10 = binascii.unhexlify(twos_complement_to_hex(0))
        SCM_y11 = binascii.unhexlify(twos_complement_to_hex(np.cos(theta)))
        SCM_y12 = binascii.unhexlify(twos_complement_to_hex(-np.sin(theta))) 
        SCM_z20 = binascii.unhexlify(twos_complement_to_hex(0)) 
        SCM_z21 = binascii.unhexlify(twos_complement_to_hex(np.sin(theta))) 
        SCM_z22 = binascii.unhexlify(twos_complement_to_hex(np.cos(theta))) 
        SCM_xoff = binascii.unhexlify(twos_complement_to_hex(0))
        SCM_yoff = binascii.unhexlify(twos_complement_to_hex(0))
        SCM_zoff = binascii.unhexlify(twos_complement_to_hex(0))
        
        ser_write(FPGA_ser,Sync_Pat+OpcodeSetMatrix+SCM_x00+SCM_x01+SCM_x02+SCM_y10+SCM_y11+SCM_y12+SCM_z20+SCM_z21+SCM_z22+SCM_xoff+SCM_yoff+SCM_zoff,False)
   
################
# Input case 3 (start with vector in xy plane, 35 deg up from x, rotate about x, then z, then y)
################
    elif iterate==3:
        theta = (180 + 35) * np.pi / 180 # Rotation matrix about x (35 deg about x)
        about_x_matrix = np.array( [ [1,             0,              0], 
                            [0, np.cos(theta), -np.sin(theta)], 
                            [0, np.sin(theta),  np.cos(theta)] ] )
        # SCM_x00 = binascii.unhexlify(twos_complement_to_hex(1))
        # SCM_x01 = binascii.unhexlify(twos_complement_to_hex(0))
        # SCM_x02 = binascii.unhexlify(twos_complement_to_hex(0)) 
        # SCM_y10 = binascii.unhexlify(twos_complement_to_hex(0))
        # SCM_y11 = binascii.unhexlify(twos_complement_to_hex(np.cos(theta)))
        # SCM_y12 = binascii.unhexlify(twos_complement_to_hex(-np.sin(theta))) 
        # SCM_z20 = binascii.unhexlify(twos_complement_to_hex(0)) 
        # SCM_z21 = binascii.unhexlify(twos_complement_to_hex(np.sin(theta))) 
        # SCM_z22 = binascii.unhexlify(twos_complement_to_hex(np.cos(theta))) 
        # SCM_xoff = binascii.unhexlify(twos_complement_to_hex(0))
        # SCM_yoff = binascii.unhexlify(twos_complement_to_hex(0))
        # SCM_zoff = binascii.unhexlify(twos_complement_to_hex(0))
        
        # matrix_1 = [[SCM_x00, SCM_x01, SCM_x02],[SCM_y10, SCM_y11, SCM_y12], [SCM_z20, SCM_z21, SCM_z22]]
        #ser_write(FPGA_ser,Sync_Pat+OpcodeSetMatrix+SCM_x00+SCM_x01+SCM_x02+SCM_y10+SCM_y11+SCM_y12+SCM_z20+SCM_z21+SCM_z22+SCM_xoff+SCM_yoff+SCM_zoff,False)
  
        theta = 15 * np.pi / 180 # Rotation matrix about z (15 deg about z)
        about_z       = np.array( [ [np.cos(theta), -np.sin(theta), 0], 
                            [np.sin(theta),  np.cos(theta), 0], 
                            [            0,              0, 1] ] )
        # SCM_x00 = binascii.unhexlify(twos_complement_to_hex(np.cos(theta)))
        # SCM_x01 = binascii.unhexlify(twos_complement_to_hex(-np.sin(theta)))
        # SCM_x02 = binascii.unhexlify(twos_complement_to_hex(0)) 
        # SCM_y10 = binascii.unhexlify(twos_complement_to_hex(np.sin(theta)))
        # SCM_y11 = binascii.unhexlify(twos_complement_to_hex(np.cos(theta)))
        # SCM_y12 = binascii.unhexlify(twos_complement_to_hex(0)) 
        # SCM_z20 = binascii.unhexlify(twos_complement_to_hex(0)) 
        # SCM_z21 = binascii.unhexlify(twos_complement_to_hex(0)) 
        # SCM_z22 = binascii.unhexlify(twos_complement_to_hex(1)) 
        # SCM_xoff = binascii.unhexlify(twos_complement_to_hex(0))
        # SCM_yoff = binascii.unhexlify(twos_complement_to_hex(0))
        # SCM_zoff = binascii.unhexlify(twos_complement_to_hex(0))
        
        # matrix_2 = [[SCM_x00, SCM_x01, SCM_x02],[SCM_y10, SCM_y11, SCM_y12], [SCM_z20, SCM_z21, SCM_z22]]
        #ser_write(FPGA_ser,Sync_Pat+OpcodeSetMatrix+SCM_x00+SCM_x01+SCM_x02+SCM_y10+SCM_y11+SCM_y12+SCM_z20+SCM_z21+SCM_z22+SCM_xoff+SCM_yoff+SCM_zoff,False)
        
        theta = 47 * np.pi / 180 # Rotation matrix about y (47 deg about y)
        about_y       = np.array( [ [ np.cos(theta),  0, np.sin(theta)], 
                            [             0,  1,             0], 
                            [-np.sin(theta),  0, np.cos(theta)] ] )
        # SCM_x00 = binascii.unhexlify(twos_complement_to_hex(np.cos(theta)))
        # SCM_x01 = binascii.unhexlify(twos_complement_to_hex(-np.sin(theta)))
        # SCM_x02 = binascii.unhexlify(twos_complement_to_hex(0)) 
        # SCM_y10 = binascii.unhexlify(twos_complement_to_hex(np.sin(theta)))
        # SCM_y11 = binascii.unhexlify(twos_complement_to_hex(np.cos(theta)))
        # SCM_y12 = binascii.unhexlify(twos_complement_to_hex(0)) 
        # SCM_z20 = binascii.unhexlify(twos_complement_to_hex(0)) 
        # SCM_z21 = binascii.unhexlify(twos_complement_to_hex(0)) 
        # SCM_z22 = binascii.unhexlify(twos_complement_to_hex(1)) 
        # SCM_xoff = binascii.unhexlify(twos_complement_to_hex(0))
        # SCM_yoff = binascii.unhexlify(twos_complement_to_hex(0))
        # SCM_zoff = binascii.unhexlify(twos_complement_to_hex(0))
        # matrix_3 = [[SCM_x00, SCM_x01, SCM_x02],[SCM_y10, SCM_y11, SCM_y12], [SCM_z20, SCM_z21, SCM_z22]]

        final_matrix = np.matmul(about_x_matrix, about_z)
        final_matrix = np.matmul(final_matrix,about_y)
        
        SCM_x00 = binascii.unhexlify(twos_complement_to_hex(final_matrix[0][0]))
        SCM_x01 = binascii.unhexlify(twos_complement_to_hex(final_matrix[0][1]))
        SCM_x02 = binascii.unhexlify(twos_complement_to_hex(final_matrix[0][2]))
        SCM_y10 = binascii.unhexlify(twos_complement_to_hex(final_matrix[1][0]))
        SCM_y11 = binascii.unhexlify(twos_complement_to_hex(final_matrix[1][1]))
        SCM_y12 = binascii.unhexlify(twos_complement_to_hex(final_matrix[1][2]))
        SCM_z20 = binascii.unhexlify(twos_complement_to_hex(final_matrix[2][0]))
        SCM_z21 = binascii.unhexlify(twos_complement_to_hex(final_matrix[2][1]))
        SCM_z22 = binascii.unhexlify(twos_complement_to_hex(final_matrix[2][2]))
        ser_write(FPGA_ser,Sync_Pat+OpcodeSetMatrix+SCM_x00+SCM_x01+SCM_x02+SCM_y10+SCM_y11+SCM_y12+SCM_z20+SCM_z21+SCM_z22+SCM_xoff+SCM_yoff+SCM_zoff,False)
   
    # theta = 270 * np.pi / 180 # Change the theta value here when required
    # SCM_x00 = binascii.unhexlify(twos_complement_to_hex(np.cos(theta)))
    # SCM_x01 = binascii.unhexlify(twos_complement_to_hex(-np.sin(theta)))
    # SCM_x02 = binascii.unhexlify(twos_complement_to_hex(0)) # same
    # SCM_y10 = binascii.unhexlify(twos_complement_to_hex(np.sin(theta)))
    # SCM_y11 = binascii.unhexlify(twos_complement_to_hex(np.cos(theta)))
    # SCM_y12 = binascii.unhexlify(twos_complement_to_hex(0)) # same
    # SCM_z20 = binascii.unhexlify(twos_complement_to_hex(0)) # same
    # SCM_z21 = binascii.unhexlify(twos_complement_to_hex(0)) # same
    # SCM_z22 = binascii.unhexlify(twos_complement_to_hex(1)) # same
    # SCM_xoff = binascii.unhexlify(twos_complement_to_hex(0))
    # SCM_yoff = binascii.unhexlify(twos_complement_to_hex(0))
    # SCM_zoff = binascii.unhexlify(twos_complement_to_hex(0))

    # ser_write(FPGA_ser,Sync_Pat+OpcodeSetMatrix+SCM_x00+SCM_x01+SCM_x02+SCM_y10+SCM_y11+SCM_y12+SCM_z20+SCM_z21+SCM_z22+SCM_xoff+SCM_yoff+SCM_zoff,False)
   
    print('FPGA Configured')
    time.sleep(0.5)
    ser_write(FPGA_ser,Sync_Pat+Config+spec_core+testmode,False)
    print('Set Test Mode')
    time.sleep(0.5)
    ser_write(FPGA_ser,Sync_Pat+Test_Enable,False)
    print('FPGA Started')

    out_folder = 'HW-output'
    FPGA_rev = "60220713_"
    # to get timestamp
    now = datetime.now()
    date_time = now.strftime("_%m%d%Y_%H%M%S")
    vals = readFPGA(FPGA_ser,readcon=readcon,num_read=num,outpath=out_folder+'/FPGA-' + FPGA_rev + amp + phase + f + '_iter' + str(iterate) + date_time)

    v=int(vals[0][0])
    print('First Entry: ',v) #Let's look at the first datum
    iterate+=1

# #--------------------------------------David's code--------------------------------

# #################
# ## Input signal
# #################

# Npts   = len(channels0_td) #1000

# f      = 1e3
# time   = np.arange(0,Npts) / Npts * 10
# amp    = 5

# #################
# ## Input case 1 (rotate x to y)
# #################
# input1 = channels0_td #input1 = np.sin(2 * np.pi * time) * amp
# input2 = channels1_td #input2 = np.zeros(Npts)
# input3 = channels0_td #input3 = np.zeros(Npts)

# ## Diagnostic 
# plt.plot(time, input1)
# plt.plot(time, input2)
# plt.plot(time, input3)
# plt.show()

# ## Rotation matrix from x to y (90 deg about z, counter-clockwise)
# theta         = 270 * np.pi / 180
# x_to_y_matrix = np.array( [ [np.cos(theta), -np.sin(theta), 0], 
#                             [np.sin(theta),  np.cos(theta), 0], 
#                             [            0,              0, 1] ] )

# ## Input array [N x 3]
# input_array = np.zeros([len(input1),3])
# input_array[:,0] = input1
# input_array[:,1] = input2
# input_array[:,2] = input3

# ## Apply rotatation to each point
# output = np.matmul(input_array, x_to_y_matrix)

# ## Diagnostic 
# plt.plot(time, output[:,0])
# plt.plot(time, output[:,1])
# plt.plot(time, output[:,2])
# plt.show()


# ## Print expected output values 
# print('')
# print('===================')
# print('Test 1 (rotation about z')
# print('===================')
# print('')
# print('Rotation Matrix:')
# print(x_to_y_matrix)
# print('')
# print('Input Amps')
# print(np.max(input_array[:,0]))
# print(np.max(input_array[:,1]))
# print(np.max(input_array[:,2]))
# print('')
# print('Output Amps')
# print(np.max(output[:,0]))
# print(np.max(output[:,1]))
# print(np.max(output[:,2]))
# print ('')
# print ('Also: check phase of Y output (should match phase of X input')
# print ('')

# ## Writing into text file for verification
# file = open('result.txt','w')
# file.write("I/P 1"+"\t"+"I/P 2"+"\t"+"I/P 3"+"\t"+"O/P 1"+"\t"+"O/P 2"+"\t"+"O/P 3"+"\n")
# for a in range(0,len(output[:,0])):
#     file.write(str(input_array[a,0])+"\t"+str(input_array[a,1])+"\t"+str(input_array[a,2])+"\t"+str(output[a,0])+"\t"+str(output[a,1])+"\t"+str(output[a,2])+"\n")
# file.close()
#--------------------------------End of David's code-------------------------------
print('Test Completed')


