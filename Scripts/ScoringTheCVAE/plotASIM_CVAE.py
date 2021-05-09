#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:45:12 2021

@author: rcj
"""
import matplotlib.pyplot as plt

def plotASIM(x,x_reconstruct,i, labels):
    #data = x.view(3,-1).cpu() 
    data = x
    fig = plt.figure(figsize=(18,10))
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(data[0][:],'-b', label = 'MMIA 337 nm')
    ax1.legend();
    ax1.set_ylabel(u"\u03bcW/m²")
    ax1.set_xlabel('Sample')
    ax1.set_title(r'Observation $\mathbf{x}$')
    
    ax2=plt.subplot(gs[1,0],sharex=ax1)
    ax2.plot(data[1][:],'-m', label = 'MMIA UV (180-230 nm)')
    ax2.legend();
    ax2.set_ylabel(u"\u03bcW/m²")
    ax2.set_xlabel('Sample')
    
    ax3=plt.subplot(gs[2,0],sharex=ax1)
    ax3.plot(data[2][:],'-r', label = 'MMIA 777.4 nm')
    ax3.legend();
    ax3.set_ylabel(u"\u03bcW/m²")
    ax3.set_xlabel('Sample')
    
    # Plot reconstruction
    ax4=plt.subplot(gs[0,1],sharex=ax1)
    ax4.plot(x_reconstruct[0][20:-5],'-b', label = 'MMIA 337 nm')
    ax4.legend();
    ax4.set_ylabel(u"\u03bcW/m²")
    ax4.set_xlabel('Sample')
    ax4.set_title(r'Reconstruction $\mathbf{x} \sim p_\theta(\mathbf{x} | \mathbf{z}), \mathbf{z} \sim q_\phi(\mathbf{z} | \mathbf{x})$')
    
    ax5=plt.subplot(gs[1,1],sharex=ax1)
    ax5.plot(x_reconstruct[1][10:-5],'-m', label = 'MMIA UV (180-230 nm)')
    ax5.legend();
    ax5.set_ylabel(u"\u03bcW/m²")
    ax5.set_xlabel('Sample')
    
    ax6=plt.subplot(gs[2,1],sharex=ax1)
    ax6.plot(x_reconstruct[2][10:-10],'-r', label = 'MMIA 777.4 nm')
    ax6.legend();
    ax6.set_ylabel(u"\u03bcW/m²")
    ax6.set_xlabel('Sample')
    plt.suptitle('Cluster Group: %i' %(labels[i]))
    