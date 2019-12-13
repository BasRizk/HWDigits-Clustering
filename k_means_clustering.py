# -*- coding: utf-8 -*-

from utils import get_data, print_progress, draw_plot
import numpy as np

# =============================================================================
# ------------------------------Pathes
# =============================================================================
images_path = "Images"
count_img_path = "Counts.jpg"

# =============================================================================
# -------------------------Algorithm Implementation
# =============================================================================

def calc_distance(vector1, vector2):
    """
    
    Parameters
    ----------
    vector1 : 1D numpy array
    vector2 : 1D numpy array
    
    Returns
    -------
    Float
        Euclidean distance between the 2 vectors.

    """
    return np.linalg.norm(vector1 - vector2)

def initialize_centers(data, k=10):
    """
    
    Parameters
    ----------
    data : 2D numpy array
        each row corresponds to a sample.
    k : integer, optional
        number of clusters. The default is 10.

    Returns
    -------
    2D numpy array
        1- Pick one of the dataset points randomly as the center of the first cluster
        2- For the next cluster, find the point with maximum distance to
        the center of the previous cluster that has not been already chosen as a center
        3- Choose this point as the center of the next cluster
        4- Repeat steps 2 and 3 until you initialize the centers of all clusters
    """
#    data_not_means = data
    means = []
#    np.zeros((k, data.shape[1]))
    means.append(data[np.random.choice(len(data), 1)[0]])
    for i in range(1, k):        
        prev_mean = means[i-1]
#        data_not_means = data_not_means[data_not_means != prev_mean]
        mag_vector = np.apply_along_axis(calc_distance,
                                         1,
                                         data,
                                         vector2=prev_mean)
        while True:
            max_pt_index = np.argmax(mag_vector, axis=0)
            farthest_point = data[max_pt_index]
            already_mean = False
            for old_mean in means:
                if (old_mean == farthest_point).all():
                    already_mean = True
            if already_mean:
                # print("Max. distance vector is already a mean.")
                mag_vector[max_pt_index] = 0    
            else:
                means.append(farthest_point)
                break
    means = np.array(means)
    return means

def calc_r_nk(vector, means):
    """

    Parameters
    ----------
    vector : 1D numpy array
        corresponds to one row of data sample.
    means : 2D numpy array
        corresonds to the current means of the data.

    Returns
    -------
    1D numpy array
        with all columns equal to 0 except one equals 1,
        which corresponds to vector's assigned cluster

    """
    all_distances = np.zeros(means.shape[0])
#    print(str(vector.shape))
    for mean_i in range(len(means)):
        a_mean = means[mean_i]
        all_distances[mean_i] = calc_distance(vector, a_mean)
    min_dist_i = np.argmin(all_distances)
    r_nk = np.zeros(means.shape[0])
    r_nk[min_dist_i] = 1
    return r_nk
    
def update_means(data, r_nk):
    """
    
    Parameters
    ----------
    data : 2D numpy array
        each row corresponds to a sample.
    r_nk : 2D numpy array
        current labeling of the data, according to current means.

    Returns
    -------
    2D numpy array
        newly calculated means of the data according to k-means clustering
        algorithm

    """
    def mult_x_rk(big_vector, cut_index, mean_index):
#        print(big_vector.shape)
        x = big_vector[:cut_index]
        r_k = big_vector[cut_index:][mean_index]
        return x*r_k
    
    # def mult(data, r_nk, mean_index):
    #     result = np.zeros((data.shape[1],))
    #     for row_i in range(len(data)):
    #         result += data[row_i]*r_nk[row_i][mean_index]
    #     return result
            
        
    num_of_means = r_nk.shape[1]
    updated_means = np.zeros((num_of_means, data.shape[1]))

    concat_data = np.concatenate((data, r_nk), axis=1)    

    for mean_i in range(num_of_means):
        nominator = np.sum(np.apply_along_axis(mult_x_rk,
                                                1,
                                                concat_data,
                                                cut_index=data.shape[1],
                                                mean_index=mean_i),
                            axis=0)
        
        # nominator_2 = mult(data, r_nk, mean_i)
            
#        r_nk_reshaped = r_nk.reshape(r_nk.shape[1], r_nk.shape[0])
        
#        nominator = np.multiply(r_nk_reshaped[mean_i], data)
        denominator = np.sum(r_nk.T[mean_i])
        
        if denominator == 0:
            print("WARNING: denominator = 0 @ mean_i %s" % str(mean_i))
            continue
            
        updated_means[mean_i] = nominator/denominator

    return updated_means

def apply_k_means(data, k=10, iterations=1):
    """

    Parameters
    ----------
    data : 2D numpy array
        each row corresponds to a sample.
    k : integer, optional
        number of clusters. The default is 10.
    iterations : integer, optional
        number of different initializations. The default is 1.

    Returns
    -------
    dict
        each of its keys correspond to iteration.
        each of its values correspond  to a tuple containing:
            (r_nk, u_k)
            where
                r_nk :: clustering results
                u_k :: converged means

    """
    all_converges = {}
    for i in range(iterations):
        print("Running initialization %s " % str(i))

        u_k = initialize_centers(data, k)
        epoch = 1
        while(True):
            print_progress("..Epoch: ", epoch)
            epoch += 1
            
            r_nk = np.apply_along_axis(calc_r_nk, 1, data, means=u_k)    
#            print(r_nk)
            past_u_k = u_k
            u_k = update_means(data, r_nk)
            r_nk
            
            if (past_u_k == u_k).all():
                break  
        
        all_converges[i] =  (r_nk, u_k)
        print()
        print("Finished.")
        
    return all_converges

# =============================================================================
# ----------------------------Other Utils
# =============================================================================
def gray_scale_to_binary(np_img_array, threshold=140):
    """

    Parameters
    ----------
    np_img_array : 2D numpy array
        it should correspond to a gray-scale image.
    threshold : number, optional
        The default is 140.

    Returns
    -------
    np_img_array
        corresponding to a binary (black and white) image.

    """
    return np.where(np_img_array > threshold, 1, 0)
    
def calc_classification(r_nk):
    """
    
    Parameters
    ----------
    r_nk : 2D numpy array
        r_nk according to k-means algorithm.

    Returns
    -------
    y : 1D numpy array
        calculated succesful samples per cluster,
        based on prior classification of samples.

    """
    num_of_classes = r_nk.shape[1]
    num_of_samples_per_class = int(r_nk.shape[0]/num_of_classes)

    clustering_results = np.zeros((num_of_classes,num_of_classes))
    
    for digit in range(num_of_classes):
        class_begin = digit*num_of_samples_per_class
        class_end = class_begin + num_of_samples_per_class
        for cluster in range(num_of_classes):
            clustering_results[digit][cluster] =\
                sum(r_nk[sample_i][cluster]\
                    for sample_i in range(class_begin, class_end))
                
    y = np.zeros((num_of_classes,))
    # which_cluster = np.zeros((num_of_classes,))
    for digit in range(len(clustering_results)):
        digit_cluster = clustering_results[digit]
        # cluster_i = np.argmax(digit_cluster)
        # which_cluster[digit]=cluster_i
        cluster_count = np.max(digit_cluster)
        y[digit] = cluster_count
        
    return y


# def which_is_best(clusterings, num_of_samples_per_class):
#     """
#     @returns best convergance out of the given converges
    
#     based on ....TODO 
#     """
#     all_performances = np.zeros((clusterings.shape[0],))
#     for clustering_i in range(len(clusterings)):
#         performance = 0
#         for cluster_count in clusterings[clustering_i]:
#             performance += (cluster_count/num_of_samples_per_class)
#         all_performances[clustering_i] = performance
        
#     index_of_best = np.argmax(all_performances)
#     return clusterings[index_of_best], index_of_best

# =============================================================================
# --------------------------- Application
# =============================================================================

data, _ = get_data(images_path, 2400)
# Convert the images to be binary instead of gray-scale.
# Use a threshold of 140 for binarization.
data = np.apply_along_axis(gray_scale_to_binary, 1, data, threshold=140)

"""
=> The initialization strategy outlined above applied 30 different times.
    For each initialization, the code should then apply the K-means algorithm
    until it converges.
"""
all_converges = apply_k_means(data, k=10, iterations=30)

"""
=> A plot of the number of images clustered together for each digit in the
  best clustering result.
  The x-axis should show the digit number (0, 1, …, 9) while the y-axis
  should show the count. 
  When the images of one digit are clustered in different clusters,
  use the count of the cluster that has the majority of images.
"""
all_organizations = np.zeros((30, 10))
for i in range(len(all_converges)):
    all_organizations[i] = calc_classification(all_converges[i][0])

"""
=> Determine which of the 30 outputs is the best clustering result.

index_of_best : index of best clustering iteration
    chosen according to the maximum calculated sum of each clustering outcome
    per digit, which was calculated bearing in mind the prior classification
    of the data
"""
index_of_best = np.argmax([sum(all_organizations[i]) for i in range(30)])

x = np.arange(0, 10, step=1)
y = all_organizations[index_of_best]
plot_label = "Clustering " + str(index_of_best)
draw_plot(x, y, plot_label, img_path=count_img_path)