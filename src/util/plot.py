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
        train_loss=[537.1259720015034,311.46366274101194,215.26763017859818,161.04821276327752,126.90998558963477,103.79342116459478,87.14342638276989,74.53650303448545,65.35032499763867,57.89608969564797],
        dev_loss=[1510.4365036699483,2047.5553977827851,2416.620277191773,2735.570992536798,3214.5724760031576,3160.294129617939,3382.7238565429857,3723.0286064911243,3778.0763983560087,3690.8246074301965])