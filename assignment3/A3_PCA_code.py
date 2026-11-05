## # ---------------------------------------------------
# Irina Khan
# SYDE 572 Assignment 3, Parts 2 to 5
#
# Objective: 
# This assignment is aimed to conduct PCA on a dataset containing 150 data points and 5 features 
# describing properties of iris flowers.
#
# According to the Iris dataset:
#
# "6. Number of Attributes: 4 numeric, predictive 
#     attributes and the class
#
# 7. Attribute Information:
#    1. sepal length in cm
#    2. sepal width in cm
#    3. petal length in cm
#    4. petal width in cm
#    5. class: 
#       -- Iris Setosa
#       -- Iris Versicolour
#       -- Iris Virginica           ..."
# ------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

# The following imports are used in Part 3
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# The following imports are used in Part 5
from sklearn.manifold import Isomap

TRAINING_SETS = 150
FEATURES = 5

# The following code imports the data set 'iris.data' (Note that the 'bezdekIris.data' file has the same contents as 'iris.data')
with open('iris.data', 'r') as f:
    lines = f.readlines()[:]

# Get every individual number (or text string) from the data file
features = []
for line in lines:
    if line.strip() == "":
        continue

    features.extend(line.strip().split(','))

sepalLengths = []
sepalWidths = []
petalLengths = []
petalWidths = []
classOfIris = []


# The following code splits the file into 5 different feature lists of size 150, since there are 150 data points.
counter = 0
for value in features:
    if (counter % (FEATURES) == 0):
        sepalLengths.append(value)
    elif (counter % (FEATURES) == 1):
        sepalWidths.append(value)
    elif (counter % (FEATURES) == 2):
        petalLengths.append(value)
    elif (counter % (FEATURES) == 3):
        petalWidths.append(value)
    elif (counter % (FEATURES) == 4):
        classOfIris.append(value)
    else:
        print("Error has occured with splitting the 'iris.data' file.")
        break
    counter = counter + 1

# ------------------------------------------------------------------------
# Part 2: Data Visualization of Two Features (Sepal Width vs Sepal Length)
# ------------------------------------------------------------------------
setosaData = []
versicolorData = []
virginicaData = []

# Displaying two corresponding features
# I arbitrarily chose the sepal length and width (aka the 1st and 2nd feature in the data)
for i in range(TRAINING_SETS):
    if (classOfIris[i] == "Iris-setosa"):
        setosaData.append([sepalLengths[i], sepalWidths[i]])
    elif (classOfIris[i] == "Iris-versicolor"):
        versicolorData.append([sepalLengths[i], sepalWidths[i]])
    elif (classOfIris[i] == "Iris-virginica"):
        virginicaData.append([sepalLengths[i], sepalWidths[i]])
    
setosaData = np.array(setosaData, dtype=float)
versicolorData = np.array(versicolorData, dtype=float)
virginicaData = np.array(virginicaData, dtype=float)


plt.title('Sepal Width (cm) vs. Sepal Length (cm) for Iris Setosa, Iris Versicolor, and Iris Virginica Flower Classes')
plt.grid(True, which="both", linestyle=':', alpha=0.6)

# Each of the iris flower classes (setosa, versicolor, and virginica) are plotted in the order of lowest to highest zorder
# The setosa class of data points do not overlap with the versicolor or virginica flower classes, and so the size of the points of 
# the virginica class data points are smaller (s=5) so that the data points can be shown overlapping each other. (You will see yellow
# dots on top of the blue dots)
plt.scatter(setosaData[:, 0], setosaData[:, 1], c='r', s=20, label='Iris Setosa Class', zorder=3)
plt.scatter(versicolorData[:, 0], versicolorData[:, 1], c='b', s=20, label='Iris Versicolor Class', zorder=4)
plt.scatter(virginicaData[:, 0], virginicaData[:, 1], c='#FFD417', s=5, label='Iris Virginica Class', zorder=5)

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend()
plt.show()
# ------------------------------------------------------
# Part 3: PCA 
# ------------------------------------------------------

# The following lines of code split the 'features' list into 4 columns of 150 data points and 1 column of 150 classifications
data = np.array(features).reshape(150, FEATURES)
feature_lists = data[:, :(FEATURES-1)] 
class_list = data[:, (FEATURES-1)]  

# The first step is to normalize the data so that it has a mean of 0 and a variance of 1
# The normalization is handled through 'scaler', an instance of the StandardScaler class from the scikit-learn library
scaler = StandardScaler()
feature_lists_scaled = scaler.fit_transform(feature_lists)

# The second step is to apply PCA by using 'pca', an instance of the PCA class from the scikit-learn library
pca = PCA(n_components=(FEATURES-1))
pca.fit(feature_lists_scaled)

print("PCA axes:")
print(pca.components_)

# The third step is to marginalize the data by projecting the data points from the 4 features onto the PCA axes
# found in the second step
feature_lists_PCA = pca.transform(feature_lists_scaled)


# The fourth step is to visualize the data on the major axes, the top 2 axes, and the top 3 axes,
# which is written below.

# -----------------------------------
# Visualizing the major axis, aka PC1
# -----------------------------------
# Plotting the 1D scatter plot as visualization
major_axis = feature_lists_PCA[:, 0]

# The following code differentiates the scatter points vertically BY CLASS so you can clearly see the seperate flower classes
plt.scatter(major_axis[:50], np.ones(50)*-1, c='r', s=25, label="Iris Setosa Class", zorder=3, alpha=0.4)
plt.scatter(major_axis[50:100],np.zeros(50), c='b', s=25, label="Iris Versicolor Class", zorder=4, alpha=0.4)
plt.scatter(major_axis[100:150], np.ones(50), c='#FFD417', s=25, label="Iris Virginica Class", zorder=5, alpha=0.4)

plt.xlabel("PC1")
plt.ylabel("Classes")
plt.yticks([-1, 0, 1], labels=["Setosa", "Versicolor", "Virginica"])
plt.ylim(-3, 3)
plt.legend()
plt.grid(True, which="both", linestyle=':', alpha=0.6)
plt.title("Dataset projected onto major axis PC1, for Iris Setosa,\nIris Versicolor, and Iris Virginica Flower Classes")
plt.show()

# ---------------------------------------------
# Visualizing the top two axes, aka PC1 and PC2
# ---------------------------------------------

two_axes = feature_lists_PCA[:, :2]
# The difference data points from each class overlap each other, so different data point sizes and opacities are used are used
plt.scatter(two_axes[:50,0], two_axes[:50,1], c='r', s=25, label="Iris Setosa Class", zorder=3, alpha=0.4)
plt.scatter(two_axes[50:100,0], two_axes[50:100,1], c='b', s=25, label="Iris Versicolor Class", zorder=4, alpha=0.4)
plt.scatter(two_axes[100:150,0], two_axes[100:150,1], c='#FFD417', s=25, label="Iris Virginica Class", zorder=5, alpha=0.4)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True, which="both", linestyle=':', alpha=0.6)
plt.title("Dataset projected onto PC1 and PC2, for Iris Setosa,\nIris Versicolor, and Iris Virginica Flower Classes")
plt.show()

# ----------------------------------------------------
# Visualizing the top three axes, aka PC1, PC2 and PC3
# ----------------------------------------------------
three_axes = feature_lists_PCA[:, :3]

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

ax.scatter(three_axes[:50, 0], three_axes[:50, 1], three_axes[:50, 2], c='r', s=25, label="Iris Setosa Class", alpha=0.8)
ax.scatter(three_axes[50:100, 0], three_axes[50:100, 1], three_axes[50:100, 2], c='b', s=25, label="Iris Versicolor Class", alpha=0.8)
ax.scatter(three_axes[100:150, 0], three_axes[100:150, 1], three_axes[100:150, 2], c='#FFD417', s=25, label="Iris Virginica Class", alpha=0.8)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

ax.set_title("Dataset projected onto PC1, PC2 and PC3, for Iris Setosa,\nIris Versicolor, and Iris Virginica Flower Classes")
ax.legend()

# # Code to see the top view of the 3D plot (aka PC3 comes out of the page)
#ax.view_init(elev=90, azim=-90)

# # Code to see the front view of the 3D plot (aka PC2 goes into the page)
#ax.view_init(elev=0, azim=-90)

# # Code to see the side view of the 3D plot (aka PC1 comes out of the page)
#ax.view_init(elev=0, azim=0)

plt.show()

# ------------------------------------------------------
# Part 4: Data Compression
# ------------------------------------------------------
print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

datapoints_centered = feature_lists_scaled - np.mean(feature_lists_scaled, axis=0)

frobenius_squared = np.linalg.norm(datapoints_centered, 'fro')**2

total_variance = frobenius_squared / (len(datapoints_centered) - 1)
PCA_total_variance = np.sum(pca.explained_variance_)

print("Total variance using Frobenius norm:", total_variance)
print("Total variance using PCA:", PCA_total_variance )
print("The difference between the Frobenius norm and PCA:", total_variance-PCA_total_variance)
# # ------------------------------------------------------
# # Part 5: Non-linear Components
# # ------------------------------------------------------

# As requested in the Assignment 3 instructions, the ISOMAP function is used with 3 neighbors
# The ISOMAP function creates the neighrhood graph, calculates the geodesic coordinates and completes the 
# 'MDS' to obtain three dimensions for our 3D plot visualization.
isomap = Isomap(n_neighbors=3, n_components=3)


feature_lists_ISOMAP = isomap.fit_transform(feature_lists_scaled)


fig, ax_5 = plt.subplots(subplot_kw={'projection': '3d'})

ax_5.scatter(feature_lists_ISOMAP[:50, 0], feature_lists_ISOMAP[:50, 1], feature_lists_ISOMAP[:50, 2], c='r', s=25, label="Iris Setosa Class", alpha=0.8)
ax_5.scatter(feature_lists_ISOMAP[50:100, 0], feature_lists_ISOMAP[50:100, 1], feature_lists_ISOMAP[50:100, 2], c='b', s=25, label="Iris Versicolor Class", alpha=0.8)
ax_5.scatter(feature_lists_ISOMAP[100:150, 0], feature_lists_ISOMAP[100:150, 1], feature_lists_ISOMAP[100:150, 2], c='#FFD417', s=25, label="Iris Virginica Class", alpha=0.8)

ax_5.set_xlabel("ISOMAP Dimension 1")
ax_5.set_ylabel("ISOMAP Dimension 2")
ax_5.set_zlabel("ISOMAP Dimension 3")

ax_5.set_title("Dimensionally-Reduced 3D Plot Dataset via ISOMAP \n(with k = 3 neighbors), for Iris Setosa, Iris Versicolor, \nand Iris Virginica Flower Classes")
ax_5.legend()

# # Code to see the top view of the 3D plot (aka ISOMAP Dimension 3 comes out of the page)
#ax_5.view_init(elev=90, azim=-90)

# # Code to see the front view of the 3D plot (aka ISOMAP Dimension 2 goes into the page)
# ax_5.view_init(elev=0, azim=-90)

# # Code to see the side view of the 3D plot (aka ISOMAP Dimension 1 comes out of the page)
# ax_5.view_init(elev=0, azim=0)

plt.show()



