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
from readFPGA import read_FPGA_input, twos_complement_to_hex, proper_twos_complement
import matplotlib.pyplot as plt
import binascii
from datetime import datetime
import pandas as pd


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
test_time = 0 # This is the number of times the test runs


#initialize serial ports
pic_ser = serial.Serial("COM5",115200)
pic_ser1 = serial.Serial("COM4",115200)
FPGA_ser = serial.Serial("COM6",115200)

while test_time < 6:

    if fromFile:
        inputs = 'Inputs/'
        if test_time == 0 or test_time == 1:
            amp = "low" # "low" "mid" "hi"
        elif test_time == 2 or test_time == 3:
            amp = "mid" # "low" "mid" "hi"
        elif test_time == 4 or test_time == 5:
            amp = "hi" # "low" "mid" "hi"
        phase = "5deg"
        f = "_03khz"
        #file0 = inputs+amp+"0deg"+f+'.txt'
        #file0 = 'Inputs/Increment_counter_input.txt'
        #file1 = 'Inputs/Increment_counter_input.txt'
        #file1 = inputs+amp+phase+f+'.txt' 
        #file2 = inputs+amp+phase+f+'.txt' 
        phase0 = "03khz"
        file0 = inputs+amp+"_amp_"+phase0+'.txt'
        phase1 = "10khz"
        file1 = inputs+amp+"_amp_"+phase1+'.txt'
        phase2 = "24khz"
        file2 = inputs+amp+"_amp_"+phase2+'.txt'
        channels0_td = read_FPGA_input(file0,signed=True,show_plots=False)
        channels1_td = read_FPGA_input(file1,signed=True,show_plots=False)
        channels2_td = read_FPGA_input(file2,signed=True,show_plots=False)
    else:
        channels0_td = test_signal(fs, sample_len, signal_freq0, amp0, shift=shift0, channel_num=0, show_plots=False, save_output='both')
        channels1_td = test_signal(fs, sample_len, signal_freq1, amp1, shift=shift1, channel_num=1, show_plots=False, save_output='both')
        channels2_td = test_signal(fs, sample_len, signal_freq1, amp1, shift=shift1, channel_num=1, show_plots=False, save_output='both')
    if len(channels0_td) > 20480:
        num_samples = 1024
    else:
        num_samples = int(len(channels0_td))
    print(num_samples)

    test0 = channels0_td[0:num_samples]
    test1 = channels1_td[0:num_samples]
    test2 = channels2_td[0:num_samples]


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
    iterate = 28
    #set test mode
    testmode = ADC_And_Rotation #change to ADC_And_Rotation for feedback test
    readcon = 'none'
    mode = 'ADC_Rot'

    ### Read Text File
    df = pd.read_csv("3_test_cmds.txt", sep="\t")
    angle_vals = df.Rot[:]
    test_number = df.Test[:]



    while iterate < len(test_number):
        
        # Initialize rotation 
        rot_num = 0
        ### Create initial unit matrix
        final_matrix = np.array( [ [1,             0,              0], 
                                [0,             1,             0], 
                                [0,             0,             1] ] )

        ### Creating Model Matrix
        model_matrix = np.zeros([len(test1),3])
        model_matrix[:,0] = test0
        model_matrix[:,1] = test1
        model_matrix[:,2] = test2

        matrix_model_mul = np.array( [ [1,             0,              0], 
                                [0,             1,             0], 
                                [0,             0,             1] ] )

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

        while rot_num < len(angle_vals[iterate]):

            ### Declaring theta
            if rot_num==0:
                theta = df.Angle1[iterate]
            elif rot_num==1:
                theta = df.Angle2[iterate]
            elif rot_num==2:
                theta = df.Angle3[iterate]
            model_theta = theta             # Change to 360 - theta

            # Convert to radian
            theta *= np.pi / 180
            model_theta *= np.pi / 180

            # Initializing xoff, yoff, zoff as 0
            xoff = 0
            yoff = 0
            zoff = 0

            ################
            # about X rotation
            ################
            if angle_vals[iterate][rot_num] == "X":
                about_x       = np.array( [ [1,             0,              0], 
                                        [0, np.cos(theta), -np.sin(theta)], 
                                        [0, np.sin(theta),  np.cos(theta)] ] )
                model_about_x = np.array( [ [1,             0,              0], 
                                        [0, np.cos(model_theta), -np.sin(model_theta)], 
                                        [0, np.sin(model_theta),  np.cos(model_theta)] ] )
                
                final_matrix = np.matmul(final_matrix,about_x)
                model_matrix = np.matmul(model_matrix,model_about_x)
                matrix_model_mul = np.matmul(matrix_model_mul,model_about_x)
            ################
            # about Y rotation
            ################
            elif angle_vals[iterate][rot_num] == "Y":
                about_y       = np.array( [ [ np.cos(theta),  0, np.sin(theta)], 
                                        [             0,  1,             0], 
                                        [-np.sin(theta),  0, np.cos(theta)] ] )
                model_about_y = np.array( [ [ np.cos(model_theta),  0, np.sin(model_theta)], 
                                        [             0,  1,             0], 
                                        [-np.sin(model_theta),  0, np.cos(model_theta)] ] )

                final_matrix = np.matmul(final_matrix,about_y)
                model_matrix = np.matmul(model_matrix,model_about_y)
                matrix_model_mul = np.matmul(matrix_model_mul,model_about_y)

            ################
            # about Z rotation
            ################
            elif angle_vals[iterate][rot_num] == "Z":
                about_z       = np.array( [ [np.cos(theta), -np.sin(theta), 0], 
                                        [np.sin(theta),  np.cos(theta), 0], 
                                        [            0,              0, 1] ] )
                model_about_z = np.array( [ [np.cos(model_theta), -np.sin(model_theta), 0], 
                                        [np.sin(model_theta),  np.cos(model_theta), 0], 
                                        [            0,              0, 1] ] )

                final_matrix = np.matmul(final_matrix,about_z) 
                model_matrix = np.matmul(model_matrix,model_about_z)
                matrix_model_mul = np.matmul(matrix_model_mul,model_about_z)

            #######
            # Offsets
            #######
            elif angle_vals[iterate][rot_num] == "x":
                xoff = df.Xoff[iterate]
            elif angle_vals[iterate][rot_num] == "y":
                yoff = df.Yoff[iterate]
            elif angle_vals[iterate][rot_num] == "z":
                zoff = df.Zoff[iterate]
            rot_num += 1

        # Model rotation matrix
        #model_matrix = np.matmul(model_matrix,final_matrix) #debug
        #model_matrix = np.matmul(model_matrix,matrix_model_mul)
        # FPGA rotation matrix
        SCM_x00 = binascii.unhexlify(twos_complement_to_hex(final_matrix[0][0]))
        SCM_x01 = binascii.unhexlify(twos_complement_to_hex(final_matrix[0][1]))
        SCM_x02 = binascii.unhexlify(twos_complement_to_hex(final_matrix[0][2]))
        SCM_y10 = binascii.unhexlify(twos_complement_to_hex(final_matrix[1][0]))
        SCM_y11 = binascii.unhexlify(twos_complement_to_hex(final_matrix[1][1]))
        SCM_y12 = binascii.unhexlify(twos_complement_to_hex(final_matrix[1][2]))
        SCM_z20 = binascii.unhexlify(twos_complement_to_hex(final_matrix[2][0]))
        SCM_z21 = binascii.unhexlify(twos_complement_to_hex(final_matrix[2][1]))
        SCM_z22 = binascii.unhexlify(twos_complement_to_hex(final_matrix[2][2]))
        SCM_xoff = binascii.unhexlify(proper_twos_complement(xoff))
        SCM_yoff = binascii.unhexlify(proper_twos_complement(yoff))
        SCM_zoff = binascii.unhexlify(proper_twos_complement(zoff))
        
        ser_write(FPGA_ser,Sync_Pat+OpcodeSetMatrix+SCM_x00+SCM_x01+SCM_x02+SCM_y10+SCM_y11+SCM_y12+SCM_z20+SCM_z21+SCM_z22+SCM_xoff+SCM_yoff+SCM_zoff,False)
        # ser_write(FPGA_ser,Sync_Pat+OpcodeSetMatrix+SCM_x00+SCM_y10+SCM_z20+SCM_x01+SCM_y11+SCM_z21+SCM_x02+SCM_y12+SCM_z22+SCM_xoff+SCM_yoff+SCM_zoff,False)
        
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
        vals = readFPGA(FPGA_ser,readcon=readcon,num_read=num,outpath=out_folder+'/3-ch'+'/Looped_twice_all'+'/'+amp+'FPGA-' + FPGA_rev + amp + phase + f + '_iter' + str(iterate) + date_time)

        ## Writing into text file for verification of Model
        file = open(out_folder+'/3-ch'+'/Looped_twice_all'+'/'+amp+'Model-' + str(iterate) + date_time+ '.txt','w')
        file.write("R3"+"\t"+"R2"+"\t"+"R1"+"\t"+"A3"+"\t"+"A2"+"\t"+"A1"+"\n")
        for a in range(0,len(model_matrix[:,0])):
            file.write(str(int(model_matrix[a,2]))+"\t"+str(int(model_matrix[a,1]))+"\t"+str(int(model_matrix[a,0]))+"\t"+str(int(test2[a]))+"\t"+str(int(test1[a]))+"\t"+str(int(test0[a]))+"\n")
        file.close()

        ## Comparing the model values with FPGA values
        df1 = pd.read_csv(out_folder+'/3-ch'+'/Looped_twice_all'+'/'+amp+'FPGA-' + FPGA_rev + amp + phase + f + '_iter' + str(iterate) + date_time +'ADCLoopback_int'+'.txt', sep="\t")
        print(df1[0:5])

        df2 = pd.read_csv(out_folder+'/3-ch'+'/Looped_twice_all'+'/'+amp+'Model-' + str(iterate) + date_time+ '.txt', sep="\t")
        print(df2[0:5])

        diff_3 = abs(df1.R3 - df2.R3)
        diff_2 = abs(df1.R2 - df2.R2)
        diff_1 = abs(df1.R1 - df2.R1)

        ## Writing into text file for verification of Model
        file = open(out_folder+'/3-ch'+'/Looped_twice_all'+'/'+amp+'Verification-' + str(iterate) + date_time+ '.txt','w')
        file.write("R3_df"+"\t"+"R2_df"+"\t"+"R1_df"+"\n")  
        for a in range(0,len(df2.R1)):
            file.write(str(diff_3[a])+"\t"+str(diff_2[a])+"\t"+str(diff_1[a])+"\n")
        file.close()

        # v=int(vals[0][0])]
        # print('First Entry: ',v) #Let's look at the first datum   
        if iterate<28:
        ## Verifying if the difference is greater than what its supposed to be
            df_v = pd.read_csv(out_folder+'/3-ch'+'/Looped_twice_all'+'/'+amp+'Verification-' + str(iterate) + date_time+ '.txt', sep="\t")
            if not all(num<3 for num in df_v.R1_df):
                print("OVER THE LIMIT!!" * 10)
            elif not all(num<3 for num in df_v.R2_df):
                print("OVER THE LIMIT!!" * 10)
            elif not all(num<3 for num in df_v.R3_df):
                print("OVER THE LIMIT!!" * 10)
            # else:
            #     print("You're safe, for now")   
        iterate+=1
    test_time+=1
print('Test Completed')