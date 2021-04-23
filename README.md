# Imputation-of-missing-data
Imputation of missing data using Hybrid approach: Machine learning techniques like Hierarchical-Clustering, Maximum-Relevance, Cosine-Similarity and Euclidean-Distance is used 

Algorithm:

Step 1: Divide the entire data set into two subsets. One subset has the records which are complete (D-Complete) and the other has records with missing data(D-missing)

Step 2: Select features which can be used in the process. Two ways to select the features: Either manually or using MaxRelevance.

Step 3: Apply Hierarchical clustering on D-Complete and find the clusters (HCs).

Step 4: Loop through D-missing and find the cluster in HCs which is very related to it using Euclidean Distance. The result is, a cluster to which the record is in and the record in the cluster which is the closest to it.

Step 5: Find the trimmed mean for that feature in that particular cluster. ( P = 0.25, which means: if we have n elements, k = P * n; remove k elements from first and k elements from last of the sorted n elements. Now for R = n - 2k elements are used for finding mean.)

Step 6: Cosine Similarity is employed between the record from cluster (which is complete record), and the missing data record. 
If Cosine value is: 
1. Greater or equal to 0.5, then cosine value is multiplied with the value from D-Complete max relevant record and added to trimmed mean of that particular cluster. This is the imputing value.

2. Less than 0.5, then cosine value is multiplied with the value from D-Complete max relevant record and subtracted from trimmed mean of that particular cluster. This is the imputing value.
