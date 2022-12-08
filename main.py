import pandas as pd
import warnings
import clustering
import formatting
import training

CONFIG = {
    'data_path': 'data/demo_data.csv',         # data file path
    'target_advertiser_id': ['id_0001'],       # advertiser to predict (multiple possible)
    'max_horizon': 60,                         # longest horizon to predict (max look forward)
    'min_horizon': 14,                         # shortest horizon to predict (min look forward)
    'encoder_length': 90,                      # encoder window length (look back)
    'holdout_split_idx': 1000,                 # time index to split holdout data
    'include_competition': True,               # set False to benchmark against model without competition
    'model_path': 'trained_models/',           # directory to save trained model
    'cluster_path': 'cluster_results/',        # directory to save trained model
    'name_prefix': 'demo'                      # prefix to add to cluster result and model 
} 


def main():
    # Ignore warnings
    warnings.simplefilter('ignore')
    
    # Time series clustering
    clusterer = clustering.distanceClustering(CONFIG)
    clust_assignment = clusterer.dtw_kmeans(k_min=7, k_max=7, k_step=1)
    
    # Data formatting
    formatter = formatting.dataFormatter(CONFIG, clustering=clust_assignment)
    adv_data, adv_holdout = formatter.get_adv_datasets()
    
    train_pydata, valid_pydata = formatter.get_py_datasets(adv_data)
    train_pyloader, valid_pyloader = formatter.get_py_dataloaders(train_pydata, valid_pydata)
    
    # Model training
    trainer = training.trainTFT(CONFIG)
    models = trainer.train_model(train_pydata, train_pyloader, valid_pyloader)
    
    
if __name__ == '__main__':
    main()