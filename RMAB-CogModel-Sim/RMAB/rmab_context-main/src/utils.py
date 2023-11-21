import matplotlib
# matplotlib.use('pdf')

import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
import sys
import glob
from tqdm import tqdm
from matplotlib.lines import Line2D
from matplotlib import rcParams

#Ensure type 1 fonts are used
import matplotlib as mpl
# mpl.rcParams['ps.useafm'] = True
# mpl.rcParams['pdf.use14corefonts'] = True
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.unicode']=True


# SMALL_SIZE = 30
# MEDIUM_SIZE = 36
# BIGGER_SIZE = 36
SMALL_SIZE = 16
MEDIUM_SIZE = 24
BIGGER_SIZE = 24
plt.rc('font', weight='bold')
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


class Point:
    def __init__(self, initx, inity):
        self.x = initx
        self.y = inity
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def __str__(self):
        return "x=" + str(self.x) + ", y=" + str(self.y)
    def distance_from_point(self, the_other_point):
        dx = the_other_point.getX() - self.x
        dy = the_other_point.getY() - self.y
    def slope(self,other_point):
        if self.x - other_point.getX() == 0 :
            return 0
        else:
            panta = (self.y - other_point.getY())/ (self.x - other_point.getX())
            return panta
    def distance_to_line(self, p1, p2):
        x_diff = p2.x - p1.x
        y_diff = p2.y - p1.y
        num = abs(y_diff*self.x - x_diff*self.y + p2.x*p1.y - p2.y*p1.x)
        den = math.sqrt(y_diff**2 + x_diff**2)
        return num / den

def swap_axes(T,R):
    # T_i = np.swapaxes(T,0,1)
    T_i = np.transpose(T,(1,0,2))
    R_i = np.zeros(T_i.shape)
    for x in range(R_i.shape[0]):
        for y in range(R_i.shape[1]):
            R_i[x,:,y] = R

    return T_i, R_i

def list_valid_action_combinations(N,C,B,options):

    costs = np.zeros(options.shape[0])
    for i in range(options.shape[0]):
        costs[i] = C[options[i]].sum()
    valid_options = costs <= B
    options = options[valid_options]
    return options


def epsilon_clip(T, epsilon):
    return np.clip(T, epsilon, 1-epsilon)


    
def groupedBarPlot(infile_prefix, ylabel='Average Adherence out of 180 days',
            title='', filename='image.png', root='.'):
    
    import glob
    d={}
    labels=[]
    for fname in glob.glob(infile_prefix+'*'):
        df = pd.read_csv(fname)
        d[fname] = {}
        d[fname]['labels'] = df.columns.values
        labels = df.columns.values
        d[fname]['values'] = df.values[0]
        d[fname]['errors'] = df.values[1]

    print(d)

    fname = os.path.join(root,'test.png')

    # plt.figure(figsize=(8,6))
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    
    rects = []
    fig, ax = plt.subplots(figsize=(8,6))
    for i,key in enumerate(d.keys()):
        rects1 = ax.bar(x+i*width, d[key]['values'], width, yerr=d[key]['errors'], label='average adherence'+key[-8:])
        rects.append(rects1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)   
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60)
    ax.legend()
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    # for r in rects:
    #     autolabel(r)       
    plt.tight_layout() 
    plt.savefig(fname)
    # plt.show()

def barPlot(labels, values, errors, ylabel='Average Adherence out of 180 days',
            title='Adherence simulation for 20 patients/4 calls', filename='image.png', root='.',
            bottom=0):
    
    fname = os.path.join(root,filename)
    # plt.figure(figsize=(8,6))
    x = np.arange(len(labels))  # the label locations
    width = 0.85  # the width of the bars
    fig, ax = plt.subplots(figsize=(8,5))
    rects1 = ax.bar(x, values, width, yerr=errors, bottom=bottom, label='average adherence')
    # rects1 = ax.bar(x, values, width, bottom=bottom, label='Intervention benefit')
    
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)   
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.legend()
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
            
    autolabel(rects1)       
    plt.tight_layout() 
    plt.savefig(fname)
    plt.show()


def rewardPlot(labels, values, fill_between_0=None, fill_between_1=None, ylabel='Average Adherence out of 180 days',
            title='Adherence simulation for 20 patients/4 calls', filename='image.png', root='.', x_ind_for_line=-1, ylim=None):
    
    fname = os.path.join(root,filename)
    # plt.figure(figsize=(8,6))
    # x = np.arange(len(labels))  # the label locations
    
    colors = ['r','g','b','k','c','y','m','r','g','b']

    fig, ax = plt.subplots(figsize=(16,10))
    for i in range(len(labels)):
        ax.plot(values[i], label=labels[i], color=colors[i], alpha=0.75)
        if fill_between_0 is not None:
            ax.fill_between(np.arange(len(values[i])), fill_between_0[i], fill_between_1[i], color=colors[i], alpha=0.2)

    # import pdb
    # pdb.set_trace()

    # for i in range(len(labels)):
    #     x0 = 0
    #     # print(len(values))
    #     x1 = len(values[i])
    #     ax.plot([x0, x1],[np.mean(values[i]), np.mean(values[i])], color = colors[i])
    # x0 = 0
    # x1 = len(values[1])
    # ax.plot([x0, x1],[values[-2][x_ind_for_line], values[-2][x_ind_for_line]], color = colors[len(labels)-2])
    # rects1 = ax.bar(x, values, width, bottom=bottom, label='Intervention benefit')
    
    ax.set_ylabel(ylabel, fontsize=14)
    if ylim is not None:
        ax.set_ylim(ylim)
    # ax.set_yscale("log")
    ax.set_title(title, fontsize=14)   
    ax.legend()
    
    
    plt.tight_layout() 
    plt.savefig(fname)
    # plt.show()

# def plothist():
#     plt.hist(x1, **kwargs, color='g', label='Ideal')
#     plt.hist(x2, **kwargs, color='b', label='Fair')
#     plt.hist(x3, **kwargs, color='r', label='Good')

def plotLambdas(true_values, decoupled_values, filename, root):
    
    fname = os.path.join(root,filename)
    # plt.figure(figsize=(8,6))
    for i in range(len(true_values)):
        true_lam, decoupled_lams = true_values[i], np.array(decoupled_values[i])

        # print('True vs others')
        # print(true_lam)
        # print(decoupled_lams)
        fig, ax = plt.subplots(figsize=(8,5))
        vals, bins, p = ax.hist(decoupled_lams, 50, label=r'$\lambda^{i}$')

        ax.plot([true_lam, true_lam],[0,max(vals)], label=r'$\lambda_{min}$', linestyle='--', linewidth=5)

        # mean_decoupled_lam = np.mean(decoupled_lams[decoupled_lams>0])
        # ax.plot([mean_decoupled_lam, mean_decoupled_lam],[0,max(vals)], label='Nonzero Mean decoupled lambda', linestyle='-.')

        mean_decoupled_lam = np.mean(decoupled_lams)
        ax.plot([mean_decoupled_lam, mean_decoupled_lam],[0,max(vals)], label=r'Mean($\lambda^{i}$)', linestyle='-.', linewidth=5)




        ax.set_ylabel('Count')
        ax.set_xlabel(r'$\lambda^i$')
        # ax.set_title('Distribution of decoupled lambdas about the coupled value', fontsize=14)   
        ax.legend(loc='upper right')
        
         
        plt.tight_layout() 
        plt.savefig(fname)
        plt.show()
        break


def plotIterativeLambdas(true_values, iterative_values, filename, root, only_goods_lambda=None):
    
    fname = os.path.join(root,filename)
    # plt.figure(figsize=(8,6))
    for i in range(len(true_values)):
        true_lam, iter_lams = true_values[i], iterative_values[i]
        only_goods_lambda_val = None
        if only_goods_lambda is not None:
            only_goods_lambda_val = only_goods_lambda[i]

        # print('True vs others')
        # print(true_lam)
        # print(iter_lams)
        fig, ax = plt.subplots(figsize=(8,5))
        
        ax.plot(iter_lams, label='iterative lambdas')
        ax.plot([0,len(iter_lams)], [true_lam, true_lam], label='Coupled lambda', linestyle='--')

        if only_goods_lambda is not None:
            ax.plot([0,len(iter_lams)], [only_goods_lambda_val, only_goods_lambda_val], label='Only "goods" lambda', linestyle=':')            
        # mean_decoupled_lam = np.mean(decoupled_lams)
        # ax.plot([mean_decoupled_lam, mean_decoupled_lam],[0,max(vals)], label='Mean decoupled lambda', linestyle='-.')

        ax.set_ylabel('lambda value', fontsize=14)
        ax.set_title('Progression of iterative lambdas against true coupled value', fontsize=14)   
        ax.legend()
        
         
        plt.tight_layout() 
        plt.savefig(fname%i)
        plt.show()


def plotBLambdas(lbs, ubs, xvals, true_lambdas, filename, root):

    fname = os.path.join(root,filename)
    # plt.figure(figsize=(8,6))
    for i in range(len(lbs)):
        lb, ub, xval, true_lam = lbs[i], ubs[i], xvals[i], true_lambdas[i]

        # print('True vs others')
        # print(true_lam)
        # print(iter_lams)
        fig, ax = plt.subplots(figsize=(8,5))
        
        ax.plot(xval, lb, label='Lower bound')
        ax.plot(xval, ub, label='Upper bound')
        ax.plot([xval[0],xval[-1]], [true_lam, true_lam], label='True lambda')

        ax.set_ylabel('lambda value', fontsize=14)
        ax.set_title('Progression of BLam', fontsize=14)   
        ax.legend()

        ax.set_ylim([0,0.5])
        
         
        plt.tight_layout() 
        plt.savefig(fname%i)
        plt.show()
