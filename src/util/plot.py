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
        train_loss=[779.572998046875, 608.0465087890625, 477.38140869140625],
        dev_loss=[383.90045166015625, 320.151611328125, 278.0235595703125, 249.1951904296875])