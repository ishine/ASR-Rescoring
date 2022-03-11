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
        train_loss=[0.03900464360486311, 0.024581876384877306, 0.01729039039288094, 0.01361271147303835,
         0.011470267097935223, 0.010248804294734218, 0.00954468423756414, 0.008809597185811558,
          0.007426869729567909, 0.007220313084026066], 
        dev_loss=[0.12936815237566643, 0.12936813976454797, 0.12936813401467634, 0.12936813179717552,
         0.12936813962181046, 0.12936813200330674, 0.12936813516993922, 0.12936812903599146,
          0.12936813181814072, 0.1293681365905318])