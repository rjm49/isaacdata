'''
Created on 18 Oct 2017

@author: Russell
'''
from matplotlib import pyplot as plt
import numpy

def draw_ret_vs_f1():
    mx = numpy.genfromtxt("retains_to_plot.csv")    
#     plt.plot(mx[:,0],mx[:,1], label="OUT")
    plt.plot(mx[:,0],mx[:,1], label="TOO EASY")
    plt.plot(mx[:, 0], mx[:, 2], label="IN ZPD")
    plt.plot(mx[:, 0], mx[:, 3], label="TOO HARD")
    plt.xlabel("Stepwise History Retention")
    plt.ylabel("F-score")
    plt.legend()
    plt.show()
    
def draw_3prf():
    mx = numpy.genfromtxt("retains_to_plot.csv")    
    n_groups = 3
    
    p = (42, 36, 38)
    r = (89, 87, 87)
    F = (57, 51, 53)
    
    fig, ax = plt.subplots()
    
    index = numpy.arange(n_groups)
    bar_width = 0.3
    
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    
    rects1 = plt.bar(index, p, bar_width,
                     alpha=opacity,
                     color='b',
                     error_kw=error_config,
                     label='F1 In')
    
    rects2 = plt.bar(index + bar_width, r, bar_width,
                     alpha=opacity,
                     color='r',
                     error_kw=error_config,
                     label='F1 Out')

    rects3 = plt.bar(index + 2*bar_width, F, bar_width,
                     alpha=opacity,
                     color='g',
                     error_kw=error_config,
                     label='F1 Average')

    
    plt.xlabel('Question difficulty weighting')
    plt.ylabel('F1')
    plt.xticks(index + bar_width, ('Attempt Ratio', 'Expert Level', 'Constant'))
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
if __name__=="__main__":
    draw_ret_vs_f1()