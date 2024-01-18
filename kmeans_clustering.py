import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

def generate_random_dots(num_dots_per_color, spread):
    all_dots = np.random.rand(3 * num_dots_per_color, 2) * spread
    return all_dots

def plot_dots(ax, all_dots, labels, title, cmap='viridis'):
    if labels is not None:
        ax.scatter(all_dots[:, 0], all_dots[:, 1], c=labels, cmap=cmap, marker='o')
    else:
        #Color
        third = len(all_dots) // 3
        colors = np.concatenate([np.full(third, i) for i in range(3)])
        ax.scatter(all_dots[:, 0], all_dots[:, 1], c=colors, cmap=cmap, marker='o')

    ax.set_title(title)
    #Hide Grid
    ax.set_xticks([])
    ax.set_yticks([])  
    ax.axis('off')  

num_clusters = 3
num_dots_per_color = 333
spread_factor = 10

#Initialize dots
all_dots = generate_random_dots(num_dots_per_color * 3, spread=spread_factor)

#Create the two subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

#Left subplot
plot_dots(axs[0], all_dots, None, 'Initial Setup')

#Perform clustering
kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=6, random_state=42, n_init=3)
labels = kmeans.fit_predict(all_dots)

#Right subplot
plot_dots(axs[1], all_dots, labels, 'After Clustering')

plt.show()
