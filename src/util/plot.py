from cProfile import label
import matplotlib.pyplot as plt

def plot(out_file, **curves):
    for curve_name, curve in curves.items():
        length = len(curve)
        x_axis = range(1, length+1)
        plt.plot(x_axis, curve, label=curve_name)
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.legend(loc='upper right')
        
    plt.savefig(out_file)

if __name__ == "__main__":
    plot("/home/chkuo/chkuo/experiment/ASR-Rescoring/result/test.png",
        train_loss=[113.52010623594032, 68.13816797259514, 47.67990661016606, 5.27558755782372, 26.93418793566811,
        21.076311116351352, 16.903001159470527, 13.975159048380164, 11.83802457567839, 10.23299552619175], 
        dev_loss=[386.6370990978612, 453.76494021480056, 525.0091712493684, 578.8862402753089, 595.9172450225816, 651.2801696361082,
        676.508151508365 ,710.5192038201909,736.2623575950789 ,752.0357146950976])