import numpy as np


#VAST Lab
#VAST Lab Clustering Accuracy
def vca(args, y_true, y_pred, length):
    # y_true: ground truth, y_pred: clusters by clustering algorithm
    # 0 at beginning, the entropy will be added by processing each cluster.
    t_vca = 0
    # Give ground truth K
    k = args.vca
    # Get size of clusters with index
    size_clusters_index = [[len(cluster), i] for i, cluster in enumerate(y_pred)]
    sorted_clusters_index = sorted(size_clusters_index, reverse=True)
    # Get sorted cluster index by it's size
    sorted_clusters = [y_pred[i[1]] for i in sorted_clusters_index]
    # Define a set contains which label we already picked.
    picked_labels = set()
    # Process large cluster first ordered by size
    for i, cluster in enumerate(sorted_clusters):
        # We only pick the labels <= number of ground truth clusters
        if len(picked_labels) >= k:
            break
        else:
            # Get ground truth labels for the current cluster. How many different labels we can get for a cluster.
            gt_labels = [y_true[point] for point in cluster]
            # Count the appearance of each label. 
            labels_with_count = [[gt_labels.count(k), k] for k in gt_labels]
            # Sort the appearance of each label
            sorted_labels_with_count = sorted(labels_with_count, reverse=True) 
            # Process most counted label first
            for label_with_count in sorted_labels_with_count:
                # Expand label_with_count by the most counted label and it's counting
                major_class, major_class_count = label_with_count[1], label_with_count[0]
                # if selected_label in the picked_labels, then go the second most counted label, if still in then continue to next.
                if major_class not in picked_labels:
                    # count how many ground truth match the major_class
                    num_gts = [l for l in y_true if l == major_class] 
                    # TP (True Positive), how many samples match the ground truth.
                    TP = major_class_count
                    # count how many samples match the major_class
                    #num_samples = [l for l in convert_clusters_to_label(cluster, length) if l == major_class] 
                    # Mark selected_label
                    picked_labels.add(major_class)
                    # Add entropy of current cluster by total TPs divide number of the sample which match the selected_label
                    t_vca += TP / len(num_gts)
                    print(TP , len(num_gts), len(cluster))
                    # Break if got them then go the next cluster
                    break
    print("picked_labels", len(picked_labels))
    #print("picked_labels", picked_labels)
    top_labels = []
    for i in picked_labels:
        count = 0
        for j in y_true:
            if i == j:
                count += 1
        top_labels.append([count, i])
    print(sorted(top_labels, reverse = True))
    #print(y_pred)
    #print(y_true)
    vca = t_vca / k # Devide by ground truth K
    return vca


def purity_score(y_true, y_pred):
    from sklearn.metrics import accuracy_score
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size




def convert_clusters_to_label(clusters, length):
    cluster_id = 0
    labels = [-1 for i in range(length)]
    for i in clusters:
        for j in i:
            labels[j] = cluster_id
        cluster_id += 1
    return labels
