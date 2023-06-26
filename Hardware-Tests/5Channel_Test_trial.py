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
from readFPGA import read_FPGA_input

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

#define the science packet headers
Spec1 = b'\x51'
Spec2 = b'\x52'
Spec3 = b'\x53'
Spec4 = b'\x54'
Spec5 = b'\x55'
Pad = b'\xCC'

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

#initialize serial ports
pic_ser = serial.Serial("COM3",115200)
pic_ser1 = serial.Serial("COM4",115200)
FPGA_ser = serial.Serial("COM5",115200)

### 5 Channel Inputs - Generate input signal from file or aribitrarily
fromFile = True
num = 1

if fromFile:
    inputs = 'Inputs/'
    amp = "high-high_"
    phase = "5deg"
    f = "_03khz"
    phase0 = "03khz"
    file0 = inputs+"hi_amp_"+phase0+'.txt'
    phase1 = "10khz"
    file1 = inputs+"hi_amp_"+phase1+'.txt'
    phase2 = "24khz"
    file2 = inputs+"hi_amp_"+phase2+'.txt'
    phase3 = "24khz"
    file3 = inputs+"hi_amp_"+phase2+'.txt'
    phase4 = "24khz"
    file4 = inputs+"hi_amp_"+phase2+'.txt'
    channels0_td = read_FPGA_input(file0,signed=True,show_plots=False)
    channels1_td = read_FPGA_input(file1,signed=True,show_plots=False)
    channels2_td = read_FPGA_input(file2,signed=True,show_plots=False)
    channels3_td = read_FPGA_input(file3,signed=True,show_plots=False)
    channels4_td = read_FPGA_input(file4,signed=True,show_plots=False)
else: # Mostly redundant unless required
    channels0_td = test_signal(fs, sample_len, signal_freq0, amp0, shift=shift0, channel_num=0, show_plots=False, save_output='both')
    channels1_td = test_signal(fs, sample_len, signal_freq1, amp1, shift=shift1, channel_num=1, show_plots=False, save_output='both')
    channels2_td = test_signal(fs, sample_len, signal_freq1, amp1, shift=shift1, channel_num=1, show_plots=False, save_output='both')
    channels3_td = test_signal(fs, sample_len, signal_freq1, amp1, shift=shift1, channel_num=1, show_plots=False, save_output='both')
    channels4_td = test_signal(fs, sample_len, signal_freq1, amp1, shift=shift1, channel_num=1, show_plots=False, save_output='both')
if len(channels0_td) > 20480:
    num_samples = 20480
else:
    num_samples = int(len(channels0_td))
print(num_samples)

test0 = channels0_td[0:num_samples]
test1 = channels1_td[0:num_samples]
test2 = channels2_td[0:num_samples]
test3 = channels3_td[0:num_samples]
test4 = channels4_td[0:num_samples]

### Initialize Serial Ports
pic_ser = serial.Serial("COM5",115200)
pic_ser1 = serial.Serial("COM4",115200)
pic_ser2 = serial.Serial("COM4",115200)
FPGA_ser = serial.Serial("COM6",115200)


##
##
##      1 channel test
##
##

# # 1 - Generate input signal from file or aribitrarily           # HAVE TO CHANGE ONCE PARSER READY
# fromFile = True
# testmode = Spectra_Results
# num = 1

# print('1 channel test begin ->')

# if fromFile:
#     inputs = 'Inputs/'
#     f = "512Hz"
#     amp = "hi_amp_"
#     file = inputs+amp+f+'.txt'  
#     channels0_td = read_FPGA_input(file,signed=True,show_plots=False)
# else:
#     channels0_td = test_signal(fs, sample_len, signal_freq0, amp0, shift=shift0, channel_num=0, show_plots=False, save_output='both')
# if len(channels0_td) > 20480:
#     num_samples = 20480
# else:
#     num_samples = int(len(channels0_td))
# print(num_samples)
# #num_samples = 11
# test = channels0_td[0:num_samples]
# #test = [i for i in range(num_samples)]

# # 1 - initialize serial ports
# pic_ser = serial.Serial("COM4",115200)
# FPGA_ser = serial.Serial("COM5",512000)

# #reset PIC
# time.sleep(0.5)
# ser_write(pic_ser,ResetPIC+lf,True)
# time.sleep(0.5)
# #response_check(pic_ser,ack)
# #print('Reset Received')
# response_check(pic_ser,initiated)
# print('PIC Reset')

# #configure PIC
# ser_write(pic_ser,SetConfig+testmode+lf)

# #Wait for acknowledge
# response_check(pic_ser,ack)
# print('FPGA Configured')

# #Set number of samples to be buffered
# to_Send = num_samples.to_bytes(4,'big',signed=False)
# ser_write(pic_ser,SetLength+to_Send+lf)

# #Wait for acknowledge
# response_check(pic_ser,ack)
# print('Data Length Set')

# #buffer data
# t0=time.perf_counter()
# var = 0
# for i in test:
#     val = i.to_bytes(2,byteorder='big',signed=True)
#     ser_write(pic_ser,Data + val + delim + val + lf)
#     if var%1000 == 0:
#         print('buffering ', var)
#     var = var+1
#     #response_check(pic_ser,ack)

# #check for complete from PIC
# response_check(pic_ser,complete)



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



t1 = time.perf_counter()
del_t = t1-t0
print('Data buffered after %f seconds', del_t)

# 1 - start writing
ser_write(pic_ser,StartFPGA+lf)

#Wait for acknowledge
response_check(pic_ser,ack)
print('FPGA Started')

out_folder = 'HW-output'
FPGA_rev = "Rev14p1_"                                                       # Might need a change, check with system

vals,bits = readFPGA(FPGA_ser,readcon="none",num_read=num,outpath=out_folder+'/FPGA-' + FPGA_rev + amp + f)

# if testmode == ADC_And_Rotation:
#     adc3r = vals[:,0]
#     adc2r = vals[:,1]
#     adc1r = vals[:,2]
#     adc3 = vals[:,3]
#     adc2 = vals[:,4]
#     adc1 = vals[:,5]

#     out_path = out_folder+'/FPGA-'+FPGA_rev+'_ADC_And_Rotation'+f

#     save_output_txt(adc3r,out_path+'adc3r','both',bits)
#     save_output_txt(adc2r,out_path+'adc2r','both',bits)
#     save_output_txt(adc1r,out_path+'adc1r','both',bits)
#     save_output_txt(adc3,out_path+'adc3','both',bits)
#     save_output_txt(adc2,out_path+'adc2','both',bits)
#     save_output_txt(adc1,out_path+'adc1','both',bits)

# elif testmode == FFT_Results:
#     bin= vals[:,0]
#     re = vals[:,1]
#     im = vals[:,2]

#     out_path = out_folder+'/FPGA-'+FPGA_rev+'_FFT'+f


# elif testmode == FFT_Power:
#     bin = vals[:,0]
#     pwr = vals[:,1]

#     out_path = out_folder+'/FPGA-'+FPGA_rev+'_FFT_PWR'+f

# elif testmode == Spectra_Results:
#     bin = vals[:,0] 
#     comp_rst= vals[:,1] 
#     uncomp_rst = vals[:,2] 

#     out_path = out_folder+'/FPGA-'+FPGA_rev+'_Spectra_Result'+f

if testmode == Spectra_Results:
    bin = vals[:,0] 
    comp_rst= vals[:,1] 
    uncomp_rst = vals[:,2] 

    out_path = out_folder+'/FPGA-'+FPGA_rev+'_Spectra_Result'+f

v=int(vals[0][0])
print('First Entry: ',v) #Let's look at the first datum

print('1 channel test complete')

##
##
##      2 channel test
##
##

#Generate input signal from file or aribitrarily           # HAVE TO CHANGE ONCE PARSER READY
fromFile = True
num = 1

print('2 channel test begin ->')

if fromFile:
    inputs = 'Inputs/'
    
    amp = "high-high_"
    phase = "83deg"
    f = "_24khz"
    file0 = inputs+amp+"0deg"+f+'.txt'
    #file0 = 'Inputs/Increment_counter_input.txt'
    #file1 = 'Inputs/Increment_counter_input.txt'
    file1 = inputs+amp+phase+f+'.txt' 
    channels0_td = read_FPGA_input(file0,signed=True,show_plots=False)
    channels1_td = read_FPGA_input(file1,signed=True,show_plots=False)
else:
    channels0_td = test_signal(fs, sample_len, signal_freq0, amp0, shift=shift0, channel_num=0, show_plots=False, save_output='both')
    channels1_td = test_signal(fs, sample_len, signal_freq1, amp1, shift=shift1, channel_num=1, show_plots=False, save_output='both')
if len(channels0_td) > 20480:
    num_samples = 20480
else:
    num_samples = int(len(channels0_td))
print(num_samples)

test0 = channels0_td[0:num_samples]
test1 = channels1_td[0:num_samples]

#main loop
spec_core = b'\x01'
iterate = 0
while iterate < 2:
    #set test mode
    if xspec_test:
        if iterate == 0:
            testmode = X_Spec_Real_Results #change to ADC_And_Rotation for feedback test
            readcon = 'none'
            mode = 'xspec_real'
        else: 
            #testmode = X_Spec_Imaginary_Results
            testmode = X_Spec_Imaginary_Results #For testing, change this testmode
            readcon = 'none' #valid options are 'all' or 'none'. All dumps all data to a file, none proceeds with normal mode
            mode = 'xspec_imaginary'
    else:
        iterate+=1
        testmode = Spectra_Results
        readcon = 'none'
        mode = ''

    #reset FPGA
    ser_write(FPGA_ser,Sync_Pat+SW_Reset,False)
    print('FPGA Reset')

    #reset PIC and flush FPGA Serial Port
    ser_write(pic_ser,ResetPIC+lf,True)

    FPGA_ser.close()
    time.sleep(0.5)
    FPGA_ser.open()

    #response_check(pic_ser,ack)
    #print('Reset Received')
    response_check(pic_ser,initiated)
    print('PIC Reset')

    #Set number of samples to be buffered
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
    print('Data buffered after %f seconds', del_t)

    #configure and start FPGA
    ser_write(FPGA_ser,Sync_Pat+Config+spec_core+testmode,False)
    print('FPGA Configured')
    ser_write(FPGA_ser,Sync_Pat+Test_Enable,False)
    print('FPGA Started')
 
    out_folder = 'HW-output'
    FPGA_rev = "60220713_"

    vals = readFPGA(FPGA_ser,readcon=readcon,num_read=num,outpath=out_folder+'/FPGA-' + FPGA_rev + amp + phase + f)

    v=int(vals[0][0])
    print('First Entry: ',v) #Let's look at the first datum
    iterate+=1
print('2 channel test complete')
