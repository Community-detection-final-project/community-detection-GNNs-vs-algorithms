
import networkx as nx
import numpy as np
from karateclub import LabelPropagation
from sklearn.cluster import SpectralClustering
from communities_dataset import createDataset
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

def create_dataset(num_nodes, num_classes, q):
    dataset = createDataset(num_nodes, num_classes, q)
    return dataset.generate_grap(), dataset.classes
       

def check_karatclub(grap = None):
    model = LabelPropagation()
    model.fit(grap)
    cluster_membership = model.get_memberships()
    return [v for k, v in sorted(cluster_membership.items())]


def check_spectral_clustering(grap, n_clusters):
    adj = nx.adjacency_matrix(grap).toarray()
    sc = SpectralClustering(n_clusters, affinity='precomputed')
    sc.fit(adj)

    return sc.labels_



#I want to make a function in Python that receives two arrays with numbers which uses "from scipy.optimize import linear_sum_assignment" To find the best permutation of try_results and match it to original_results

def accuracy(original_results, try_results):
    # Get the unique class labels
    original_labels = np.unique(original_results)
    try_labels = np.unique(try_results)
    
    # Create a cost matrix
    cost_matrix = np.zeros((len(original_labels), len(try_labels)))
    
    # Fill the cost matrix with the number of mismatches
    for i, orig_label in enumerate(original_labels):
        for j, try_label in enumerate(try_labels):
            matches = np.sum((original_results == orig_label) & (try_results == try_label))
            cost_matrix[i, j] = -matches  # We use negative matches because we want to maximize matches
    
    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create the mapping from try_labels to original_labels
    label_mapping = {try_labels[j]: original_labels[i] for i, j in zip(row_ind, col_ind)}
    
    # Apply the mapping to try_results
    remapped_results = np.array([label_mapping[label] for label in try_results])
    # print(try_results)
    print(remapped_results)  
    
    # Calculate accuracy
    accuracy = np.mean(remapped_results == original_results)
    print(accuracy)
    return accuracy

    
    
def run(num_nodes, n_classes, q):
    grap, classes= create_dataset(num_nodes, n_classes, q)
    # print("The correct answer",classes)
    
    k = check_karatclub(grap)    

    sc =check_spectral_clustering(grap, n_classes)
    print("karatclub")
    accuracy_of_karatclub=accuracy(classes, k)

    
    print("SpectralClustering")
    accuracy_of_spectral_clustering=accuracy(classes, sc)
 

    return accuracy_of_karatclub, accuracy_of_spectral_clustering
 
if __name__ == '__main__':
    num_nodes = 100
    nun_classes = 6 
    maxQ = 0.55
    minQ = 0.05
    step = 0.05
    
    # print(run(10, 3, 0.5))
   
    q_values = np.arange(minQ, maxQ, step)
    karatclub_accuracies = []
    spectral_clustering_accuracies = []
    for q in q_values:        
        print('\n---------------------------------')
        print(f'Running for q={q}')
        karatclub_accuracy, spectral_clustering_accuracy = run(num_nodes, nun_classes, q)
        karatclub_accuracies.append(karatclub_accuracy)
        spectral_clustering_accuracies.append(spectral_clustering_accuracy)
        
    plt.plot(q_values, karatclub_accuracies, label='Label Propagation')
    plt.plot(q_values, spectral_clustering_accuracies, label='Spectral Clustering')
    plt.xlabel('q')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
