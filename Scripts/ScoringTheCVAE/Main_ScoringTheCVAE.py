#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:56:29 2021

@author: rcj
"""
import os
import torch
from ReadMMIAfits_OCRE import Read_MMIA_Observation
from ASIM_DataLoader_Slicing_Func import ASIM_DataLoader
from typing import * # The Dict class for the Forward pass is in the typing package
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from CVAE_ASIM import CVAENet
from CVAE_ASIM import VariationalInference
from plotASIM_CVAE import plotASIM
from sklearn.cluster import KMeans #K-Means Clustering
from readClasses import readClassification
from collections import defaultdict, OrderedDict
from collections import Counter
import re
wd = '/home/rcj/Desktop/DL_ASIM/'
#modelParameters = 'Epoch300_LatentSpace50_MaxPooling'
modelParameters = 'Epoch800_LatentSpace50_MaxPooling_WU400_NoDropout_BNBeforeLin_LearningR_2em4'
finalResults = '/home/rcj/Dropbox/DTU/SynthesisProject/FinalResults/'
#modelParameters = 'Epoch300_LatentSpace10_MaxPooling'

resultsPath = os.path.join(wd, 'Results/')
modelPath = os.path.join(wd,'TrainedModel/' + modelParameters +'.pth')
#BlueJetPath = os.path.join(wd, 'Known_Events_Data/')
BlueJetPath = '/home/rcj/Desktop/SpriteData/Data/Ground_Truth_Dataset'

# Define meta parameters
n = 699
latent_features = 50
input_shape = 15000
# Be aware that in CVAE_ASIM.py - CVAENET we still have to change the batch_size in  line 157 and 173
# Couldn't get it to work with importing it from here for some reason... Will be fixed eventually
batch_size = 100 
B = 1 # beta in variational inference
vi = VariationalInference(b=B)
# Initiallize cvae
vae = CVAENet(input_shape = 15000, latent_features = 50, n = 699, batch_size=100)
# Load parameters from trained mode.
vae.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')))
# Initialize evaluation
vae.eval()
# setup params dict 
params = { 'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 0
    }
# Load data
Data = ASIM_DataLoader(BlueJetPath, 1, 0, 210, p = 1, sliceSize = 5000)
# Load into dataloader
Data_Loader = torch.utils.data.DataLoader(Data, **params, drop_last=True)

# Initilize empty for later plotting
x_dataAll = []
x_reconstruct = []
z = []
# Load data - initilize the iter (has to be done outside loop to not initilize a new one each loop)
iterDataLoader = iter(Data_Loader)
for i in range(0, int(len(Data_Loader.dataset.P1D.dataset)/batch_size)):
    # Pick a new dataset
    x = next(iterDataLoader)
    # Save fofr later plotting
    for j in range(0,len(x[0])):
        x_dataAll.append([x[0][j],x[1][j],x[2][j], x[3][j], x[4][j], x[5][j]])
    # Concatenate to match expected input size
    x = torch.cat((x[0],x[1],x[2]), dim=1).view(batch_size, 1, -1)
    # Run it through the CVAE
    loss, diagnostics, outputs = vi(vae, x)
    # Save reconstruction
    x_reconstruct.append(outputs['px'].sample().view(batch_size ,3,-1).cpu().detach().numpy())
    # Save the latent space of the current dataset
    z.append(outputs['z'].view(batch_size, -1).cpu())

# Flatten reconstruction list
x_reconstruct = [item for sublist in x_reconstruct for item in sublist]
# Stack tensors for plotting.
z = torch.stack([item for sublist in z for item in sublist])
z_plot2D = TSNE(n_components=2).fit_transform(z.detach().numpy())
z_plot3D = TSNE(n_components=3).fit_transform(z.detach().numpy())


#%%
numberPlot = 8
data = x_dataAll[numberPlot]
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
ax4.plot(x_reconstruct[numberPlot][0][20:-5],'-b', label = 'MMIA 337 nm')
ax4.legend();
ax4.set_ylabel(u"\u03bcW/m²")
ax4.set_xlabel('Sample')
ax4.set_title(r'Reconstruction $\mathbf{x} \sim p_\theta(\mathbf{x} | \mathbf{z}), \mathbf{z} \sim q_\phi(\mathbf{z} | \mathbf{x})$')

ax5=plt.subplot(gs[1,1],sharex=ax1)
ax5.plot(x_reconstruct[numberPlot][1][10:-5],'-m', label = 'MMIA UV (180-230 nm)')
ax5.legend();
ax5.set_ylabel(u"\u03bcW/m²")
ax5.set_xlabel('Sample')

ax6=plt.subplot(gs[2,1],sharex=ax1)
ax6.plot(x_reconstruct[numberPlot][2][10:-10],'-r', label = 'MMIA 777.4 nm')
ax6.legend();
ax6.set_ylabel(u"\u03bcW/m²")
ax6.set_xlabel('Sample')
plt.savefig(finalResults+'ObservationAndRecon_EdgesOmitted.pdf', bbox_inches='tight')

# Plot data and reconstruction side by side with the edges, which shows the boundary problems. See report.
# Original data
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

# Reconstruction
ax4=plt.subplot(gs[0,1],sharex=ax1)
ax4.plot(x_reconstruct[numberPlot][0],'-b', label = 'MMIA 337 nm')
ax4.legend();
ax4.set_ylabel(u"\u03bcW/m²")
ax4.set_xlabel('Sample')
ax4.set_title(r'Reconstruction $\mathbf{x} \sim p_\theta(\mathbf{x} | \mathbf{z}), \mathbf{z} \sim q_\phi(\mathbf{z} | \mathbf{x})$')

ax5=plt.subplot(gs[1,1],sharex=ax1)
ax5.plot(x_reconstruct[numberPlot][1],'-m', label = 'MMIA UV (180-230 nm)')
ax5.legend();
ax5.set_ylabel(u"\u03bcW/m²")
ax5.set_xlabel('Sample')

ax6=plt.subplot(gs[2,1],sharex=ax1)
ax6.plot(x_reconstruct[numberPlot][2],'-r', label = 'MMIA 777.4 nm')
ax6.legend();
ax6.set_ylabel(u"\u03bcW/m²")
ax6.set_xlabel('Sample')
plt.savefig(finalResults+'ObservationAndRecon_EdgesIncluded.pdf', bbox_inches='tight')


#%%
# Link fileName til x_dataAll navn og frame. Alt andet lader til at virke, men fordi at x_reconstruct har 
# en tilfældig rækkefølge, men readClassification er fast, skaber det problemer. 
# Combine frameNumber and title for keys in dict to sort by
data_titleFrameNumber = [i[4] +',' +str(int(j[5])) for i, j in zip(x_dataAll, x_dataAll)]
# Load classifications, match format of title from the data
fileName, frameNumber, Classification = readClassification('/home/rcj/Desktop/DL_ASIM/Classification/GroundTruth_Table.xlsx')
fileName = ['/home/rcj/Desktop/SpriteData/Data/Ground_Truth_Dataset/' + x for x in fileName]
# combine frametitle and number forkeys
classi_titleFrameNumber = [i +',' + str(j) for i, j in zip(fileName, frameNumber)]
# Create dict from classification document
classi_tmp_dict = dict(zip(classi_titleFrameNumber, Classification))
# Order classification dict wrt. the order of the date, so that it matches
order_tmp_dict = [(key, classi_tmp_dict[key]) for key in data_titleFrameNumber if key in classi_tmp_dict]
# Extract the classes to match the rest of the script (this was written after the following code, since the 
# problem of random data from the net and fixed order in classes wasn't considered initially).
Classification = [x[:][1] for x in order_tmp_dict]

# Create Clusters
clusters = 5
latentSpace = z.detach().numpy()
kmeans = KMeans(n_clusters=clusters, random_state=0).fit(latentSpace)
# Get label of latentspace
labels = kmeans.labels_
# Get the center of each cluster
clusterCenter = kmeans.cluster_centers_

# Adaptation of: https://stackoverflow.com/questions/40828929/sklearn-mean-distance-from-centroid-of-each-cluster
# Compute the mean distance of the points to their respective cluster
def k_mean_distance(data, clusterCenter, currentLabel, all_labels):
    # Calculate Euclidean distance for each data point assigned to centroid
    distances = [np.linalg.norm(x-clusterCenter) for x in data[all_labels == currentLabel]]
    # return the mean value
    return np.mean(distances)
# Compute mean distance for each cluster
c_mean_distances = []
for i, cent_features in enumerate(clusterCenter):
            mean_distance = k_mean_distance(latentSpace, cent_features, i, labels)
            c_mean_distances.append(mean_distance)


# TMP labels
labels = labels[-len(Classification):]

ClassificationDict = defaultdict(list)
for k, v in zip(labels, Classification):
    ClassificationDict[k].append(v)
    
# Find the most common class in a current label. Since a label can have a shared most common label
# e.g. both N and HCLA has 6 occurences, which is the most, we have to account for this. The followin function
# counts this, found on : https://stackoverflow.com/questions/50463202/how-to-find-equal-most-common-values-in-a-list-together-with-count
def max_counter(lst):
    values, counts = np.unique(lst, return_counts=True)
    idx = np.where(counts == counts.max())[0]
    return list(zip(values[idx], counts[idx]))
# Find the most frequent class in a given label using the counter from collections
# Calculate how much of the label has the most fequent label.
# Loop over mostFrequentLabel, in case that the most common is shared, in which we need both.
# Save in FinalClasses
FinalClasses = defaultdict(list)
for key, values in ClassificationDict.items():
    mostFrequentLabel = max_counter(ClassificationDict[key])
    for i in range(0,len(mostFrequentLabel)):
        perc = mostFrequentLabel[i][1] / len(ClassificationDict[key])
        FinalClasses[key].append([mostFrequentLabel[i][0], int(perc*100)])

# Order with respect to the keys (labels), so we have 0:4 order to match plotting
orderedClasses = OrderedDict(sorted(FinalClasses.items()))
# Rewrite to a format that is expected for colorbar labeling and easy to ready
# Since we can have a label in which the most common label is shared between two, we have to consider this.
# The else conditions writes the case of two sharaed most common classes in the form of:
# label Label1/Label2/... % up 
gatheredFinalLabel = [None]*len(orderedClasses)
for i in range(0,len(orderedClasses)):
    if len(orderedClasses[i]) == 1:
        gatheredFinalLabel[i] = str(i) + ' ' + str(orderedClasses[i][0][0]) + ' ' + str(orderedClasses[i][0][1]) + '%'
    else: # If more than 1 class is the most common
        gatheredFinalLabel[i] = str(i) + ' '
        for j in range(0, len(gatheredFinalLabel[i])):
            gatheredFinalLabel[i] = gatheredFinalLabel[i] + str(orderedClasses[i][j][0]) + '/'
        gatheredFinalLabel[i] = gatheredFinalLabel[i][:-1] + ' ' + str(orderedClasses[i][0][1]) + '%'
            
with open(os.path.join(resultsPath, 'ModelResults_' + modelParameters + '.txt'),'w') as f:
    f.write(modelParameters + '\n')
    f.write('-'*50 + '\n')

    for i in range(0,len(gatheredFinalLabel)):
            f.write('\t' + str(i))
    f.write('\nClass' + '\t')
    for i in range(0, len(gatheredFinalLabel)):
        writeclasses = re.findall(r'([A-Za-z]+)', gatheredFinalLabel[i])
        if len(writeclasses) == 1:
            f.write(writeclasses[0] + '\t')
        else:
            writemultipleclasses  = ""
            for j in range(0,len(writeclasses)):
                writemultipleclasses = writemultipleclasses + str(writeclasses[j]) + '/'
            f.write(writemultipleclasses[:-1] + '\t')
    f.write('\n%'+ '\t')
    for i in range(0,len(gatheredFinalLabel)):
        percentage = re.findall(r'\d+%', gatheredFinalLabel[i])
        f.write(str(percentage[0]) + '\t')
    
    f.write('\nC.Dist'+ '\t')
    for i in c_mean_distances:
        f.write('%.2f' %i + '\t')
    
fig = plt.figure(figsize=(8,10))
gs = fig.add_gridspec(2, 1)

ax1 = fig.add_subplot(gs[0,0:1])
plotZ = ax1.imshow(z.detach().numpy(), aspect='auto')#, vmin=-1, vmax=1)
ax1.set_title('Feature map of Latent Space')
cbar = fig.colorbar(plotZ)
ax1.set_ylabel('Batch Number')
ax1.set_xlabel('Latent Feature')
cbar.set_label('Latent Feature Value')

ax2 = fig.add_subplot(gs[1,0])
sc = ax2.scatter(z_plot2D[:,0], z_plot2D[:,1], c=labels[:], cmap=plt.cm.rainbow)
ax2.set_title('Latent space, 2D Proj')
ax2.set_xlabel('Dimension 1')
ax2.set_ylabel('Dimension 2')
cb = plt.colorbar(sc, ticks=np.arange(0, clusters)) # Create tick locations
cb.set_label('Cluster Group')
cb.ax.set_title('Label (Percentage)')
cb.ax.set_yticklabels(gatheredFinalLabel) # Change ticks to our given ones.
plt.grid()
plt.savefig(finalResults+'LatentSpace.pdf', bbox_inches='tight')

#%%

idx = []
for i in range(0,clusters):
    idx.append(np.where(labels == i)[0][1])

for i in idx:
    plotASIM(x_dataAll[i],x_reconstruct[i],i, labels)
    plt.savefig(finalResults+'SampleReconstructCluster_'+str(labels[i])+'.pdf', bbox_inches='tight')
