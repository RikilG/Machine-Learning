import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

#Function to estimate point of intersection of normal curves of two classes, at the point of intersection diff
#is zero or a point where the difference changes it sign
def intersection(f,g,x): 
    d = f - g
    for i in range(len(d) - 1):
        if d[i] == 0. or d[i] * d[i + 1] < 0.:
            x_ = x[i]
            return x_

   

def FisherLDA(ds,s):
    row,col = ds.shape
    ds_0 = ds[ds[col-1]==0].iloc[:,0:-1].values
    ds_1 = ds[ds[col-1]==1].iloc[:,0:-1].values
    m0 = np.mean(ds_0,axis=0)  
    m1 = np.mean(ds_1,axis=0)
    SW_0 = np.cov(np.transpose(ds_0))             
    SW_1 = np.cov(np.transpose(ds_1))             
    SW = SW_0 + SW_1                                
    SW_inv = np.linalg.inv(SW)                          
    w = np.dot(SW_inv,(m1-m0))
    wT = np.transpose(w)      
    y_0 = []
    y_1 = []
    for i in range(len(ds_0)):
        y_0.append(np.dot(wT,ds_0[i,:]))
    for i in range(len(ds_1)):
        y_1.append(np.dot(wT,ds_1[i,:])) 

    fig1, ax1 = plt.subplots()
    ax1.scatter(y_0,np.zeros(np.shape(y_0)),s=1,color='r',label="Points in Class 0")
    ax1.scatter(y_1,np.zeros(np.shape(y_1)),s=1,color='g',label="Points in Class 1") 
    ax1.legend(loc='upper right')
    plt.title("One Dimension transformation for dataset "+str(s))

    mu_0 = np.mean(y_0)
    std_0 = np.std(y_1)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 10000)
    p_0 = norm.pdf(x, mu_0, std_0)
    fig2, ax2 = plt.subplots()
    ax2.set_xlim([-15,20])
    ax2.set_ylim([0,0.7])
    ax2.plot(x, p_0, 'k', linewidth=2,color='red',label="Class 0")
    mu_1 = np.mean(y_1)
    std_1 = np.std(y_1)
    x = np.linspace(xmin, xmax, 10000)
    p_1 = norm.pdf(x, mu_1, std_1)
    ax2.plot(x, p_1, 'k', linewidth=2,color='green',label="Class 1")
    ax2.legend(loc='upper right')
    intr = intersection(p_0,p_1,x)
    plt.title("Normal curve for dataset "+str(s)+"\nThreshold is "+str(intr))
    print ("Threshold values is" ,intr )
    plt.show()

def main():
    ds1=pd.read_csv("datasets/a1_d1.csv",header=None)
    ds2=pd.read_csv("datasets/a1_d2.csv",header=None)
    FisherLDA(ds1,1)
    FisherLDA(ds2,2) 

main()


