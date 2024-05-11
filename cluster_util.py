import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans, DBSCAN, OPTICS
import matplotlib.pyplot as plt 
import re
import numpy as np
#from scipy.spatial.distance import cdist
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import nltk
import string
from kneed import KneeLocator
import math
import editdistance
import hashlib
from typing import Sequence, Iterable, Hashable, List, Optional, Union
T = Iterable[Hashable]

def preprocess_text(text):
#     # remove url
#     pattern = r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
#     cleaned_text = re.sub(pattern, "", str(text))
    cleaned_text = re.sub(r'\r\n|\r|\n|\\n', ' ', str(text))
    # remove url/filepath..
    cleaned_text = re.sub(r'\S+\.\S+', ' ', cleaned_text)
    # remove words containing digits
    cleaned_text = re.sub(r'([a-zA-Z_.|:;-]*\d+[a-zA-Z_.|:;-]*)+', ' ', cleaned_text)
    # replace strange symbols to whitespace
    cleaned_text = re.sub(r'[^A-Za-z0-9\s]', ' ', cleaned_text)
    # just keep one space
    cleaned_text = re.sub(r' +', r' ', cleaned_text)
    
    # tokenize
    tokens = nltk.word_tokenize(cleaned_text)
    res = []
    for token in tokens:
        if (token not in string.punctuation) and bool(re.search(r'\d', token)) != True:
            res.append(re.sub('[^A-Za-z-]+', '', token).strip().lower())
    return " ".join(res)

def preprocess_text_transformer(text):
    cleaned_text = re.sub(r'\r\n|\r|\n|\\n', ' ', str(text))
    # remove url/filepath..
    cleaned_text = re.sub(r'\S+\.\S+', ' ', cleaned_text)
    # just keep one space
    cleaned_text = re.sub(r' +', r' ', cleaned_text)
    return cleaned_text.strip().lower()

def scaling(X_array):
    return StandardScaler().fit_transform(X_array)

def vectorizer_tfidf(sentences):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(sentences).toarray()

def vectorizer_sentence2vec(sentences):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    f = lambda text: model.encode(text, convert_to_numpy=True).flatten()
    return np.array([f(sentence) for sentence in sentences])

def load_glove(path_glove_txt):
    glove_vectors = {}
    with open(path_glove_txt, 'r', encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            glove_vectors[word] = vector
    print('Loaded %s word vectors from GloVe.' % len(glove_vectors))
    return glove_vectors

# turn A sentence to vector via word embeddings by
# taking the mean/sum of all word embeddings of the sentence
def vectorizer_word2vec(sentence, w2v_vectors, embedding_dim, aggregation='mean', is_subword=False):
    vec = np.zeros(embedding_dim).reshape((1, embedding_dim))
    count = 0
    if is_subword==False:
        for word in sentence.split():
            if word in w2v_vectors:
                vec += w2v_vectors[word].reshape((1, embedding_dim)) # update vector with new word
                count += 1 # counts every word in sentence
    else:
        for word in sentence.split():
            vec += w2v_vectors[word].reshape((1, embedding_dim)) # update vector, sub word doesn't check key existence
            count += 1 # counts every word in sentence
    if aggregation == 'mean':
        if count != 0:
            vec /= count  #get average of vector to create embedding for sentence
        return vec.flatten()
    elif aggregation == 'sum':
        return vec.flatten()
    
# elbow method for optimal value of k in kmeans
def elbow_for_kmean(X_array, K_range = range(1, 10)):
    #distortions = []
    inertias = []
    for k in K_range:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(X_array)
        kmeanModel.fit(X_array)
        #distortions.append(sum(np.min(cdist(X_array, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X_array.shape[0])
        inertias.append(kmeanModel.inertia_)
    return inertias

# X_array is assumed to be vectorized
def cluster_kmeans(X_array, n_clusters=2, max_iter=500, random_state=28):
    # cluster using k-means 
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state) 
    kmeans.fit(X_array) 

    return kmeans.labels_

def cluster_optics(X_array, min_samples = 50, metric = "minkowski"):
    optics = OPTICS(min_samples=min_samples, metric=metric).fit(X_array)
    no_clusters = len(set(optics.labels_)) - (1 if -1 in optics.labels_ else 0) # -1 is noise
    no_noise = np.sum(np.array(optics.labels_) == -1, axis=0)
    print('Estimated no. of clusters: %d' % no_clusters)
    print('Estimated no. of noise points: %d' % no_noise)
    return optics.labels_

def epsilon_search_dbscan(X_array):
    k = round(math.sqrt(len(X_array)))
    # print('K-neighbours = {}'.format(k))
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1, algorithm='auto').fit(X_array)
    distances, indices = nbrs.kneighbors(X_array)
    distances = [np.mean(d) for d in np.sort(distances, axis=0)]
    
    kneedle = KneeLocator(distances, list(range(len(distances))), online=True)
    epsilon = np.mean(list(kneedle.all_elbows))
    if epsilon == 0.0:
        epsilon = np.mean(distances)
    return epsilon

# eps: https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdf
# For 2-dimensional data, use DBSCAN’s default value of MinPts = 4 (Ester et al., 1996).
# If more than 2 dimensions, choose MinPts = 2*dim, where dim= the dimensions of your data set (Sander et al., 1998).
def cluster_dbscan(X_array, eps=0.2, min_samples=400):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X_array)
    cluster_labels = dbscan.labels_
    no_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0) # -1 is noise
    no_noise = np.sum(np.array(cluster_labels) == -1, axis=0)
    print('Estimated no. of clusters: %d' % no_clusters)
    print('Estimated no. of noise points: %d' % no_noise)
    return cluster_labels

# reduce the dimensionality of the data using PCA 
def pca(vectorized_arr, n_component = 2):
    pca = PCA(n_components=n_component) 
    reduced_data = pca.fit_transform(vectorized_arr)
    return reduced_data

## ===================evaluate based on levenshtein similarity===================

def levenshtein_similarity(a: Sequence[T], b: Sequence[T]) -> float:
    return 1 - editdistance.eval(a, b) / max(len(a), len(b))

def levenshtein_similarity_1_to_n(many: Sequence[Sequence[T]], single: Optional[Sequence[T]] = None) -> Union[List[float], float]:
    if len(many) == 0:
        return 1.
    if single is None:
        single, many = many[0], many[1:]
    if len(many) == 0:
        return [1.0]
    return [levenshtein_similarity(single, item) for item in many]

def cluster_statistics_lev(cluster_name, cluster_size, messages):
    similarity = levenshtein_similarity_1_to_n(messages)
    return {'cluster_name': cluster_name,
            'cluster_size': cluster_size,
            'mean_similarity': np.mean(similarity),
            'std_similarity': np.std(similarity)}

def statistics_lev(df, res_col_name, value_col_name="evalue",):
    patterns = []
    n_groups = df[res_col_name].nunique()-1
    for i in range(n_groups):
        messages = df[df[res_col_name]==i][value_col_name]
        patterns.append(cluster_statistics_lev(i, len(messages), messages.tolist()))
    res = pd.DataFrame(patterns, columns=['cluster_name', 'cluster_size', 'mean_similarity', 'std_similarity'])\
                .round(2)\
                .sort_values(by='cluster_size', ascending=False)
    res_ws = np.sum(res.mean_similarity*res.cluster_size)/sum(res.cluster_size)
    res_ws_std = np.sum(res.std_similarity*res.cluster_size)/sum(res.cluster_size)
    res_n_noise = sum(df[res_col_name]==-1)
#     print("weighted similarity:", np.sum(res.mean_similarity*res.cluster_size)/sum(res.cluster_size))
#     print("weighted similarity std:", np.sum(res.std_similarity*res.cluster_size)/sum(res.cluster_size))
    return res, [res_ws, res_ws_std, res_n_noise/len(df)]

## ===================evaluate based on silhouette score===================

def eval_cluster_silhouette(X_array, y_predicted):
    return silhouette_score(X_array, y_predicted)

## ===================evaluate based on ground truth===================

# evaluate clustering quality with knowing the true labels
def eval_cluster_groundtruth(Y_true, X_array):
    y_pred = kmeans.fit_predict(X_array)
    
    # Evaluate the performance using ARI, NMI, and FMI
    ari = adjusted_rand_score(Y_true, y_pred)
    nmi = normalized_mutual_info_score(Y_true, y_pred)
    fmi = fowlkes_mallows_score(Y_true, y_pred)

    # Print Metrics scores
    print("Adjusted Rand Index (ARI): {:.3f}".format(ari))
    print("Normalized Mutual Information (NMI): {:.3f}".format(nmi))
    print("Fowlkes-Mallows Index (FMI): {:.3f}".format(fmi))

## ===================other similarity===================

def jaccard_similarity(list1, list2):
    s1 = set(list1.split(" "))
    s2 = set(list2.split(" "))
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

def generateHash(target_str):
    return hashlib.md5(repr(target_str).encode('utf-8')).hexdigest()


    
def print_clusters(num_clusters, res_clusters, n_sample=10):
    for i in range(num_clusters):
        res_cluster = res_clusters[res_clusters['cluster']==i]
        print('cluster {} has {} samples'.format(i, len(res_cluster)))
        print(res_cluster.sample(min(len(res_cluster), n_sample)))
        print('\n')

def print_clusters_general(cluster_values, cluster_labels, cluster_ids, n_sample=10):
    for i in cluster_ids:
        res_cluster = cluster_values[cluster_labels==i]
        print('Cluster {} has {} samples'.format(i, len(res_cluster)))
        n_sample = min(len(res_cluster), n_sample)
        print('Printing {} samples of cluster {}'.format(n_sample, i))
        print(res_cluster.sample(n_sample))
        print('\n')
    
def plot_clusters(vectorized_texts, n_clusters, cluster_labels):
    reduced_data = pca(vectorized_texts.toarray())
    
    # plot
    labels = list(map(str, range(n_clusters)))
    for i in range(n_clusters): 
        plt.scatter(reduced_data[cluster_labels == i, 0], 
                    reduced_data[cluster_labels == i, 1],  
                    s=10, label=labels[i]) 
    plt.legend() 
    plt.show()