import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from mpl_toolkits import mplot3d
'exec(%matplotlib inline)'
from sklearn.datasets import load_breast_cancer 
  
# instantiating 
cancer = load_breast_cancer() 
  
# creating dataframe 
df = pd.DataFrame(cancer['data'], columns = cancer['feature_names']) 
  
# checking head of dataframe 
df.head() 
from sklearn.preprocessing import StandardScaler 
  
scalar = StandardScaler() 
  
# fitting 
scalar.fit(df) 
scaled_data = scalar.transform(df) 
  
# Importing PCA 
from sklearn.decomposition import PCA 
  
# Let's say, components = 3 
pca = PCA(n_components = 3) 
pca.fit(scaled_data) 
x_pca = pca.transform(scaled_data) 
  
x_pca.shape 
ax = plt.axes(projection='3d')
ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=cancer['target'], cmap='viridis', linewidth=1);
ax.set_xlabel('First Principal  component')

ax.set_ylabel('Second Principal  component')

ax.set_zlabel('Third Principal  component')


plt.show()