from random import sample
import numpy as np 
from matplotlib import pyplot as plt
from math import floor
import os.path as osp
import os
import pandas as pd
from NARMADataset import NARMA

from numpy import random

class MackeyGlass():
    
    def __init__(self, sample_len, tau):
        self.sample_len = sample_len
        self.tau = tau
        self.a = 0.2
        self.b = 0.1
        self.order = 10
        self.sample = self.sample_vectorized()



    def df(self, x):
        y = self.a*x/(1+x**self.order)
        return y
    
    def sample_vectorized(self,):
        x = np.zeros((self.sample_len,))
        t = np.zeros((self.sample_len,))

        h = 1
        x[0] = 1.2

        for k in range(self.sample_len-1):
            t[k+1] = t[k]+h
            if k < self.tau:
                k1 = -self.b*x[k]
                k2 = -self.b*(x[k]+h*k1/2)
                k3 = -self.b*(x[k]+k2*h/2)
                k4 = -self.b*(x[k]+k3*h)
                x[k+1] = x[k]+(k1+2*k2+2*k3+k4)*h/6 
            else:
                n = floor((t[k]-self.tau-t[0])/h+1)
                k1 = self.df(x[n])-self.b*x[k]
                k2 = self.df(x[n])-self.b*(x[k]+h*k1/2)
                k3 = self.df(x[n])-self.b*(x[k]+2*k2*h/2)
                k4 = self.df(x[n])-self.b*(x[k]+k3*h)
                x[k+1] = x[k]+(k1+2*k2+2*k3+k4)*h/6

        return x

    def show(self,):
        plt.plot(self.sample)
        plt.show()
    
    def save(self, location):
        np.save(os.path.join('data/paper.esn', location), self.sample)
        
        
if __name__=="__main__":
	#test()
    data_dir = "./data/paper.esn"
    if not osp.exists(data_dir):
        os.makedirs(data_dir)

    dataset = MackeyGlass(7000, 17)
    # dataset = NARMA(5600)

    dataset.show()
    dataset.save('mg')