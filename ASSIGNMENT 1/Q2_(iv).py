import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

num_of_clusters=2
dataset1 = pd.read_csv("cm_dataset_2 - cm_dataset.csv",header=None)
dataset1=np.array(dataset1)

X=dataset1.T
N = X.shape[1]  

# CLUSTERING BY GIVEN ASSIGNEMENT

# radial kernel

degrees = [2,3,4,5,10,15,20,25,30]
sigmas=[0.1,1,4,100,1000,10000]
K=2
fig,axs=plt.subplots(2,3,figsize=(15,10))
b=0
for sigma in sigmas :  
    
    R_Kernel_matrix=np.zeros((N,N),dtype=np.float64)
    for i in range(N) :
        for j in range(N):
            R_Kernel_matrix[i][j]=math.exp(-1*(np.dot((X[:,i]-X[:,j]).T,(X[:,i]-X[:,j])))/(2*(sigma*sigma)))

    sum_of_elements_row=np.sum(R_Kernel_matrix,axis=1)/N
    sum_of_elements_row=sum_of_elements_row.reshape(-1,1)
    sum_of_elements_col=np.sum(R_Kernel_matrix,axis=0)/N
    totalsum_of_elements =np.sum(R_Kernel_matrix)/((N)*(N))
    Centered_R_Kernel_matrix=np.zeros((N,N))
    Centered_R_Kernel_matrix=R_Kernel_matrix -sum_of_elements_col-sum_of_elements_row+totalsum_of_elements

    R_Kernel_eigenvalues,R_Kernel_eigenvectors=np.linalg.eigh(Centered_R_Kernel_matrix)
    R_Kernel_eigenvalues=R_Kernel_eigenvalues[::-1]
    R_Kernel_eigenvectors=R_Kernel_eigenvectors[:,::-1]
    H = R_Kernel_eigenvectors[:, :K]
    dataset=H
    current_cluster_assignement= np.zeros((dataset.shape[0])).astype(int)
    for i in range(N) :
     current_cluster_assignement[i] = np.argmax(H[i])
    
    colors =plt.cm.tab10(np.linspace(0,1,num_of_clusters)) 
    color_of_each_cluster=colors[current_cluster_assignement]
    axs[int(b/3),b%3].scatter(dataset1[:,0],dataset1[:,1],c=color_of_each_cluster)
    axs[int(b/3),b%3].set_xlabel("X-axis")
    axs[int(b/3),b%3].set_ylabel("Y-axis")
    axs[int(b/3),b%3].set_title("$\\sigma$ = "+str(sigma))
    #axs[int(a/2),a%2].scatter(H[:,0],H[:,1],c=color_of_each_cluster)
    b+=1
plt.tight_layout()
plt.show()   