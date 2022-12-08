from os import path
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting.metrics import QuantileLoss
import pytorch_lightning as pl

H_PARAMS = {
    'hidden_size': 12,
    'lstm_layers': 1,
    'dropout': 0.1,
    'attention_head_size': 4,
    'hidden_continuous_size': 9,
    'learning_rate': 0.02
}


class trainTFT():
    '''
    Train a separate Temporal Fusion Transformer for every target advertiser
    '''
    def __init__(self, CONFIG):
        self.target_advertiser_id = CONFIG['target_advertiser_id']
        self.name_prefix = CONFIG['name_prefix']
        self.model_path = CONFIG['model_path']
        
        
    def train_model(self, train_pydata, train_pyloader, valid_pyloader):
        
        # Initialize output list
        MODELS = []
        
        # Build for every target advertiser
        for i,adv in enumerate(self.target_advertiser_id):
            print(f'Train model: {self.name_prefix}_tft_{adv}')
        
            # Check if model for advertiser with prefix already exists and load if
            adv_model_path = f'{self.model_path}{self.name_prefix}_tft_{adv}.ckpt'
            if path.exists(adv_model_path):
                print('... load existing model')
                adv_model = TemporalFusionTransformer.load_from_checkpoint(adv_model_path)
                
            else:
                # Define early stopping
                early_stop_callback = EarlyStopping(
                    monitor='val_loss',
                    min_delta=1e-4,
                    patience=10,
                    verbose=False,
                    mode='min'
                )
                
                # Define model parameters
                net = TemporalFusionTransformer.from_dataset(
                    train_pydata[i],
                    loss=QuantileLoss(),
                    log_interval=10, # comment?
                    reduce_on_plateau_patience=4,
                    **H_PARAMS
                )
                
                # Define save location of top 1 model
                checkpoint_callback = ModelCheckpoint(
                    dirpath=self.model_path,
                    filename=f'{self.name_prefix}_tft_{adv}',
                    save_top_k=1,
                    verbose=False,
                    monitor='val_loss',
                    mode='min'
                )
                
                # Define pytorch lightning trainer
                trainer = pl.Trainer(
                    max_epochs=20,
                    gpus=1,
                    weights_summary='top',
                    gradient_clip_val=0.1,
                    callbacks=[early_stop_callback, checkpoint_callback],
                    limit_train_batches=30 # here same as batch size?
                )
                
                # Fit model
                trainer.fit(
                    net,
                    train_pyloader[i],
                    valid_pyloader[i],
                )
                
                # Load fitted model
                adv_model = TemporalFusionTransformer.load_from_checkpoint(adv_model_path)
            
            MODELS.append(adv_model)
            
        return MODELS