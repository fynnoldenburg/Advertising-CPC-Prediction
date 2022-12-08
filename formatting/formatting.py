import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from pytorch_forecasting import TimeSeriesDataSet

H_PARAMS = {
    'target_lags': [7],
    'budget_known': True,
    'batch_size': 32
}


class dataFormatter():
    '''
    Converts input data into pyTorch data format including external
    advertiser CPC from same cluster. Outputs are lists including one
    element per target advertiser defined in CONFIG.
    '''
    def __init__(self, CONFIG, clustering):
        self.data_path = CONFIG['data_path']
        self.target_advertiser_id = CONFIG['target_advertiser_id']
        self.include_competition = CONFIG['include_competition']
        self.clustering = clustering
        self.holdout_split_idx = CONFIG['holdout_split_idx']
        self.min_horizon = CONFIG['min_horizon']
        self.max_horizon = CONFIG['max_horizon']
        self.encoder_length = CONFIG['encoder_length']
        
        
    def get_adv_datasets(self):
        
        # Load input data
        input_data = pd.read_csv(self.data_path)
        
        # Initialize output list
        DATA, HOLDOUT = [], []
        
        # Create for every target advertiser
        for adv in self.target_advertiser_id:
            
            # Filter data set for advertiser
            adv_data = input_data[input_data.adwordscustomerid == adv].reset_index(drop=True)
            
            # Append CPC series of advertisers in same cluster with target
            if self.include_competition:
                c = self.clustering.loc[adv]
                clust_adv = list(self.clustering[self.clustering == c].index)
                clust_adv.remove(adv)

                for ca in clust_adv:
                    ca_data = input_data[input_data.adwordscustomerid == ca].reset_index(drop=True)
                    adv_data['cpc_'+ca] = ca_data['cpc']
            
            # Split data
            adv_data_split = adv_data[adv_data.time_idx < self.holdout_split_idx]
            adv_holdout = adv_data[
                (adv_data.time_idx >= self.holdout_split_idx - self.encoder_length - max(H_PARAMS['target_lags'])) &
                (adv_data.time_idx <  self.holdout_split_idx + self.max_horizon)
            ].reset_index(drop=True)
            
            # Append to output lists
            DATA.append(adv_data_split)
            HOLDOUT.append(adv_holdout)
            
        return DATA, HOLDOUT
        
    
    def get_py_datasets(self, DATA):
        
        # Initialize output lists
        TRAIN_PYDATA, VALID_PYDATA = [], []
        
        # Create for every target advertiser
        for i, adv in enumerate(self.target_advertiser_id):
            
            # Get advertiser data
            adv_data = DATA[i]
            
            # Prepare feature lists
            known = ['time_idx', 'month', 'dayofweek', 'dayofyear']
            if H_PARAMS['budget_known']: known = known+['adbudget']

            unknown = [x for x in list(adv_data.columns) if x not in 
                       known+['adwordscustomerid', 'category', 'date']]
                
            # Build pyTorch training data set
            train_cutoff = adv_data['time_idx'].max() - self.max_horizon
            
            train_pydata = TimeSeriesDataSet(
                adv_data[lambda x: x.time_idx <= train_cutoff],
                time_idx='time_idx',
                target='cpc',
                group_ids=['adwordscustomerid'],
                min_encoder_length=self.encoder_length, 
                max_encoder_length=self.encoder_length,
                min_prediction_length=self.min_horizon,
                max_prediction_length=self.max_horizon,
                static_categoricals=['adwordscustomerid', 'category'],
                static_reals=[],
                time_varying_known_categoricals=[],
                time_varying_known_reals=known,
                time_varying_unknown_categoricals=[],
                time_varying_unknown_reals=unknown,
                lags = {'cpc': H_PARAMS['target_lags']},
            )
            
            # Build pyTorch validation data set
            valid_pydata = TimeSeriesDataSet.from_dataset(
                train_pydata, 
                adv_data, 
                predict=True, 
                stop_randomization=True
            )
            
            # Append to output lists
            TRAIN_PYDATA.append(train_pydata)
            VALID_PYDATA.append(valid_pydata)
            
        return TRAIN_PYDATA, VALID_PYDATA
    
    
    def get_py_dataloaders(self, TRAIN_PYDATA, VALID_PYDATA):
        
        # Initialize output lists
        TRAIN_PYLOADER, VALID_PYLOADER = [], []
        
        # Create for every target advertiser
        for i, adv in enumerate(self.target_advertiser_id):
            
            train_pyloader = TRAIN_PYDATA[i].to_dataloader(
                train=True,
                batch_size=H_PARAMS['batch_size'],
                num_workers=0)
            
            valid_pyloader = VALID_PYDATA[i].to_dataloader(
                train=False,
                batch_size=H_PARAMS['batch_size'],
                num_workers=0)
            
            # Append to output lists
            TRAIN_PYLOADER.append(train_pyloader)
            VALID_PYLOADER.append(valid_pyloader)
        
        return TRAIN_PYLOADER, VALID_PYLOADER
        
                
        