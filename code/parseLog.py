import matplotlib.pyplot as plt
import numpy as np


def parse(logFileName):
    d = {}
    curr_epoch = 0
    f = open(logFileName,'r')
    for line in f.xreadlines():
        line = line.replace('\n','').split(':')
        if line[0] == 'epoch':
            curr_epoch = int(line[1])
        if line[0] == 'dev_loss':
            d[curr_epoch] = float(line[1])
    
    f.close()
    return d

if __name__ == '__main__':
    d1 = parse('res/train1.log')
    d2 =  parse('res/train2.log')
    d3 = parse('res/final.log')
    x = np.arange(33)
    plt.plot(x, [d1[i] for i in x])
    plt.plot(x, [d2[i] for i in x])
    plt.plot(x, [d3[i] for i in x])
    plt.legend(['setting 1', 'setting 2', 'setting 3'], loc='upper right')
    plt.show()


