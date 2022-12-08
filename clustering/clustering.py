import pandas as pd
import matplotlib.pyplot as plt
from os import path
from tslearn.clustering import TimeSeriesKMeans


class distanceClustering():
    '''
    Conducts distance-based clustering of all advertisers in the data set
    based on Dynamic-Time-Warping (DTW) of the CPC time series.
    '''
    def __init__(self, CONFIG):
        self.data_path    = CONFIG['data_path']
        self.cluster_path = CONFIG['cluster_path']
        self.name_prefix  = CONFIG['name_prefix']
        
    def dtw_kmeans(self, k_min, k_max, k_step=1):
        
        # Check if cluster with prefix already exists and load if exists
        clust_path = f'{self.cluster_path}{self.name_prefix}_dtw_kmeans_clustering.csv'
        if path.exists(clust_path):
            print('... load existing cluster assignment')
            clust = pd.read_csv(clust_path, index_col=0).clust
            
        else:
            # Read data
            input_data = pd.read_csv(self.data_path)

            # Build representation of cpc data per advertiser
            cpc_data = input_data.pivot(
                index='time_idx',
                columns='adwordscustomerid', 
                values='cpc'
            )

            # Smooth the data series and transpose
            for col in cpc_data.columns:
                cpc_data[col] = cpc_data[col].rolling(14).mean()
            cpc_data = cpc_data.dropna().T

            # Build k-range to test from user input
            K = list(range(k_min, k_max+1, k_step))

            # Generate cluster assignments for values of k in input range
            kmeans_memory = {}
            for k in K:
                print('... calculating k =', k, end='\r')
                kmeans = TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=10)
                kmeans.fit(cpc_data)
                kmeans_memory[k] = kmeans

            # Plot elbow and take user decision of final k
            inertias = []
            for kmeans in kmeans_memory.values():
                inertias.append(kmeans.inertia_)
            plt.figure(figsize=(5, 5))
            plt.plot(kmeans_memory.keys(), inertias, 'x-', c='black', lw=1)
            plt.xlabel('Values of K')
            plt.ylabel('Inertia')
            plt.show()
            k_user = int(input('... input k to use:'))

            # Get cluster assignment for selected k and save in defined path
            clust = pd.Series(
                kmeans_memory[k_user].labels_, 
                index=cpc_data.index,
                name='clust'
            )
            clust.to_csv(clust_path, index=True)

            print(f'Clustering finished (n_clusters: {len(clust.unique())})\n')
        
        return clust