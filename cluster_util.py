import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 

def cluster_kmeans(input_texts, n_clusters=2, max_iter=500, random_state=28):
    # vectorizer 
    vectorizer = TfidfVectorizer(stop_words='english') 
    vectorized_texts = vectorizer.fit_transform(input_texts)  

    # cluster using k-means 
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state) 
    kmeans.fit(vectorized_texts) 

    # save results
    results = pd.DataFrame() 
    results['errdesc'] = input_texts 
    results['cluster'] = kmeans.labels_
    
    return kmeans, results, vectorized_texts

def print_clusters(num_clusters, res_clusters, n_sample=10):
    for i in range(num_clusters):
        res_cluster = res_clusters[res_clusters['cluster']==i]
        print('cluster {} has {} samples'.format(i, len(res_cluster)))
        print(res_cluster.sample(n_sample))
        print('\n')

def plot_clusters(vectorized_texts, n_clusters, cluster_labels):
    # reduce the dimensionality of the data using PCA 
    pca = PCA(n_components=2) 
    reduced_data = pca.fit_transform(vectorized_texts.toarray())
    
    # plot
    labels = list(map(str, range(n_clusters)))
    for i in range(n_clusters): 
        plt.scatter(reduced_data[cluster_labels == i, 0], 
                    reduced_data[cluster_labels == i, 1],  
                    s=10, label=labels[i]) 
    plt.legend() 
    plt.show()