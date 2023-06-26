import matplotlib.pyplot as plt
from readFPGA import int_to_twos_complement, twos_complement

# Getting input data from FPGA file
file_path1 = '.\\Data_compare\\x-spec\\FPGA_Results\\FPGA-60220713_high-high_5deg_03khz_xpec_im_avg_hex.txt'
with open(file_path1, 'r') as file:
    # # Step 2: Read the contents of the text file
    lines = file.readlines()
    # # Step 3: Split the lines into individual values
    delimiter = '\t'  # specify the delimiter used in the text file
    values = [line.strip().split(delimiter) for line in lines]

    # # Step 4: Create lists to store the values
    fpga_hex = []

    # # Step 5: Loop through the values and append them to the lists
    for value in values:
        fpga_hex.append(value[2])

    # Step 6: Close the file
    # Step 6: Close the file
    file.close()

# Getting input data from Python file
file_path2 = '.\\Data_compare\\x-spec\\Python_Results\\high-high_5deg_03khz_channel_i_avg_hex.txt'
with open(file_path2, 'r') as file2:
    # # Step 2: Read the contents of the text file
    lines2 = file2.readlines()
    delimiter = '\t'  # specify the delimiter used in the text file
    values = [line2.strip().split(delimiter) for line2 in lines2]

    py_hex=[]

    for value in values:
        py_hex.append(value[0])
    
    # Step 6: Close the file
    # Step 6: Close the file
    file2.close()

# Convert to int
py_data=[]
for j in py_hex:
    py_data.append(twos_complement(j,16))

# Convert to int
fpga_data=[]
for j in fpga_hex[2:]:
    fpga_data.append(twos_complement(j,16))

plt.plot(py_data,list(range(1,len(py_data)+1)))

# Set labels and title
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Plot of Data')

# Show the plot
plt.show()

plt.plot(fpga_data,list(range(1,len(fpga_data)+1)))

# Set labels and title
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Plot of Data')

# Show the plot
plt.show()
