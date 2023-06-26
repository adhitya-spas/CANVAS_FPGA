import pandas as pd
import matplotlib.pyplot as plt

for m in [0, 1, 2]:
    for l in ["5", "35", "83"]:
        j = ["03", "10", "24"]
        size_end = [39, 44, 54]
        size_start = [0, 29, 44]

        fpga_data_r = pd.read_table("D:/CANVAS_work/Canvas-Algorithm/Canvas_FPGA/Data_compare/x-spec/FPGA_Results/FPGA-60220713_high-high_" + l + "deg_" + j[m] + "khz_xpec_re_avg_int.txt")

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(fpga_data_r["Comp"][size_start[m]:size_end[m]], linewidth=1, color='cyan')

        py_data_r = pd.read_table("D:/CANVAS_work/Canvas-Algorithm/Canvas_FPGA/Data_compare/x-spec/Python_Results/high-high_" + l + "deg_" + j[m] + "khz_channel_r_cmprs_int.txt")

        plt.plot(py_data_r.iloc[size_start[m]-1:size_end[m]], linewidth=1, linestyle="--", color='red')
        plt.title(j[m] + "kHz " + l + "deg" + " (r)")
        plt.legend(["fpga", "py"])

        fpga_data_i = pd.read_table("D:/CANVAS_work/Canvas-Algorithm/Canvas_FPGA/Data_compare/x-spec/FPGA_Results/FPGA-60220713_high-high_" + l + "deg_" + j[m] + "khz_xpec_im_avg_int.txt")

        plt.subplot(1, 2, 2)
        plt.plot(fpga_data_i["Comp"][size_start[m]:size_end[m]], linewidth=1, color='cyan')

        py_data_i = pd.read_table("D:/CANVAS_work/Canvas-Algorithm/Canvas_FPGA/Data_compare/x-spec/Python_Results/high-high_" + l + "deg_" + j[m] + "khz_channel_i_cmprs_int.txt")

        plt.plot(py_data_i.iloc[size_start[m]-1:size_end[m]], linestyle="--", color='red')
        plt.title(j[m] + "kHz " + l + "deg" + " (im)")
        plt.legend(["fpga", "py"])
        plt.show()
