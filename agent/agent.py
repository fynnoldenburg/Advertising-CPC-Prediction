import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sktime.performance_metrics.forecasting import mean_absolute_error as mae
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error as mape


class agentTFT():
    '''
    Module to predict unseen data and output scores, feature importance 
    and visualizations
    '''
    def __init__(self, CONFIG):
        self.encoder_length = CONFIG['encoder_length']
        self.holdout_split_idx = CONFIG['holdout_split_idx']
    
    
    def predict(self, data, prev_data, model, horizon):
        self.prev_data = prev_data
        
        # Predict data
        raw_predictions, x = model.predict(data, mode="raw", return_x=True)
        
        # Prediction data frame including confidence intervals fit to horizon
        self.pred_df = pd.DataFrame(
            raw_predictions.prediction[0].numpy(), 
            columns=['pred_lb_3','pred_lb_2','pred_lb_1','pred','pred_ub_1','pred_ub_2','pred_ub_3'],
            index=range(
                self.holdout_split_idx,
                self.holdout_split_idx+len(raw_predictions.prediction[0].numpy())
            )
        ).loc[:self.holdout_split_idx+horizon-1]
        
        # Add actuals
        self.pred_df['actuals'] = data.cpc.to_list()[self.encoder_length:self.encoder_length+horizon]
        
        # Get past data
        self.pred_df = pd.concat([
            self.pred_df, 
            pd.DataFrame(
                {'encoder': x['encoder_target'].numpy()[0]},
                index=range(self.holdout_split_idx-self.encoder_length, self.holdout_split_idx)
            )
        ])
        
        # Get previous data (one time the encoder length on top for wider visualization)
        self.pred_df = pd.concat([
            self.pred_df, 
            pd.DataFrame(
                {'previous': prev_data.cpc.to_list()[-self.encoder_length*2:-self.encoder_length]},
                index=range(
                    self.holdout_split_idx-2*self.encoder_length, 
                    self.holdout_split_idx-self.encoder_length
                )
        )
        ])
        
        # Sort and close gap between previous and encoder
        self.pred_df = self.pred_df.sort_index()
        first_enc = self.pred_df.encoder.first_valid_index()
        self.pred_df.loc[first_enc, 'previous'] = self.pred_df.loc[first_enc, 'encoder']
        
        # Save just prediction values
        self.prediction = self.pred_df.pred.dropna()
        
        # Save just actual values
        self.actuals = self.pred_df.actuals.dropna()
        
        # Save performance scores
        self.mae = mae(self.prediction, self.actuals)
        self.mape = mape(self.prediction, self.actuals)
        self.smape = mape(self.prediction, self.actuals, symmetric=True)
        
        # Get feature importances
        self.feature_importance = model.interpret_output(raw_predictions, reduction="sum")
        self.encoder_variables = model.encoder_variables
        self.decoder_variables = model.decoder_variables
        
    
    def pred_viz(self, save=False, path=None):
        
        D = self.pred_df[['pred', 'actuals', 'encoder', 'previous']]

        ## GENERAL SETUP ----------------------------------------------
        fig, ax = plt.subplots(figsize=(22,7))

        # Data borders and axis lengths as reference values for consistent viz of all TS
        x_data_min, x_data_max = D.index.min(), D.index.max()
        y_data_min, y_data_max = D.min().min(), D.max().max()
        x_len = x_data_max - x_data_min
        y_len = y_data_max - y_data_min

        # Set figure borders
        ax.set_xlim(x_data_min,            x_data_max+0.3*x_len)
        ax.set_ylim(y_data_min-0.25*y_len, y_data_max+0.5*y_len)

        # Plot competitors average CPC
        comp_data = self.prev_data.loc[:, 'dayofyear':].iloc[:, 1:]
        comp_mean_smooth = comp_data.mean(axis=1).rolling(7).mean().rolling(7).mean()[x_data_min:]
        ax.plot(comp_mean_smooth[-self.encoder_length:], c='white', lw=5) # white shading
        ax.plot(comp_mean_smooth, c='lightgrey', lw=1)

        # Plot pre-prediction data
        ax.plot(D[['previous', 'encoder']].loc[
            self.holdout_split_idx-self.encoder_length:self.holdout_split_idx-1, :
        ], c='white', lw=5) # white shading
        ax.plot(D[['previous', 'encoder']], c='black', lw=1)

        # Plot actuals
        ax.plot(D['actuals'], c='black', lw=0.75, ls='--')

        # Plot predictions
        ax.plot(D['pred'], c='red', lw=1)
        
        ## BRACKET ----------------------------------------------------
        bracket_space = 0.1 # set bracket spacing from data in percent
        x_split = self.holdout_split_idx
        x_bracket_min = x_split-self.encoder_length
        x_bracket_max = x_split+len(self.pred_df.pred.dropna())
        y_bracket_min, y_bracket_max = y_data_min-bracket_space*y_len, y_data_max+bracket_space*y_len

        # Draw bracket
        ax.add_patch(Rectangle(
            (x_bracket_min, y_bracket_min),
            x_bracket_max-x_bracket_min-1, y_bracket_max-y_bracket_min,
            edgecolor = 'black',
            fill=False,
            lw=1))

        # Draw bracket entry cover
        ax.add_patch(Rectangle(
            (x_bracket_min-0.1*x_len, y_bracket_min+0.1*y_len),
            x_bracket_max-x_bracket_min-1+0.2*x_len, y_bracket_max-y_bracket_min-0.2*y_len,
            edgecolor = None,
            facecolor = 'white',
            fill=True))

        # Draw split line
        ax.plot([x_split, x_split], [y_bracket_min, y_bracket_max], c='black', lw=1)
        
        
        ## TEMPORAL ATTENTION -----------------------------------------
        def norm(X):
            return [(x - min(X)) / (max(X) - min(X)) for x in X]

        # Join attention to time idx
        temp_att = pd.Series(
            norm(self.feature_importance['attention'].numpy()),
            index=self.pred_df.encoder.dropna().index
        )

        # Set bar transparency by creating colours
        rgba_colors = np.zeros((len(temp_att.values),4))
        rgba_colors[:,0] = 1.0
        rgba_colors[:, 3] = temp_att.values*0.4 # set strength factor here

        # Draw attention
        ax.bar(
            x=temp_att.index+0.5, 
            height=y_bracket_max-y_bracket_min, 
            bottom=y_bracket_min, color=rgba_colors,
            width=1
        )
        
        ## LABELLING --------------------------------------------------
        # Draw labels
        def draw_label(x_pos, x_len, y_len, y_data_max, label,
                       x_space=0.01, # space from defined x_pos relative to x_len
                       y_space=0.4   # space from y_data_max (must be < defined y_lim_max)
                      ):
            y_pos = y_data_max+y_space*y_len
            y_line = [y_pos, y_pos-0.5*y_space*y_len]
            ax.text(x_pos+x_space*x_len, y_pos, label, weight='bold')
            plt.plot([x_pos, x_pos], y_line, c='black', lw=1)

        # Draw legends
        def draw_legend(x_pos, x_len, y_len, y_data_max, label, c,
                        y_space, # space from y_data_max (must be < defined y_lim_max)
                        x_space=0.016, # space from defined x_pos relative to x_len
                        alpha=1
                       ):
            y_pos = y_data_max+y_space*y_len
            ax.scatter(x_pos+x_space*x_len, y_pos, s=150, marker='s', c=c, alpha=alpha)
            ax.text(x_pos+x_space*2*x_len, y_pos-0.02*y_len, label)

        draw_label( x_data_min, x_len, y_len, y_data_max, 'Previous data')
        draw_legend(x_data_min, x_len, y_len, y_data_max, 'Past advertiser CPC', 'black', 0.33)
        draw_legend(x_data_min, x_len, y_len, y_data_max, 'Average competitors CPC', 'lightgrey', 0.23)

        draw_label( x_bracket_min, x_len, y_len, y_data_max, 'Encoder')
        draw_legend(x_bracket_min, x_len, y_len, y_data_max, 'Temporal attention', 'red', 0.33, alpha=0.25)

        draw_label( x_split, x_len, y_len, y_data_max, 'Decoder')
        draw_legend(x_split, x_len, y_len, y_data_max, 'Prediction', 'red', 0.33)
        draw_legend(x_split, x_len, y_len, y_data_max, 'Actuals', 'black', 0.23)

        draw_label( x_data_max, x_len, y_len, y_data_max, 'Feature importance (encoder & decoder)')
        draw_legend(x_data_max, x_len, y_len, y_data_max, 'Advertiser inherent features', 'black', 0.33, alpha=0.75)
        draw_legend(x_data_max, x_len, y_len, y_data_max, 'Exogenous competitor features', 'lightgrey', 0.23, alpha=0.75)
        
        ## FEATURE IMPORTANCE -----------------------------------------
        # set cornerpoints (relative!)
        x_feat_min, x_feat_max = 0.778, 0.92
        y_feat_min, y_feat_max = 0.08, 0.725
        space_between = 0.05

        # get encoder importances
        top10_encoder = pd.Series(
            self.feature_importance['encoder_variables'].numpy(),
            index=self.encoder_variables
        ).sort_values(ascending=False)[:10]

        # get decoder importances
        decoder = pd.Series(
            self.feature_importance['decoder_variables'].numpy(),
            index=self.decoder_variables
        ).sort_values(ascending=True)

        # encoder coloring
        c1, c2 = 'lightgrey', 'black'
        c = [c1 if x in comp_data.columns else c2 for x in top10_encoder.index]
        
        # create axes with appropriate ratio
        total_height = y_feat_max-y_feat_min
        ratio_enc = (len(top10_encoder)/(len(top10_encoder)+len(decoder)))*total_height
        ratio_dec = (len(decoder)/(len(top10_encoder)+len(decoder)))*total_height
        ax_enc = ax.inset_axes([x_feat_min, (y_feat_min+ratio_dec+space_between), 
                                (x_feat_max-x_feat_min), ratio_enc])
        ax_dec = ax.inset_axes([x_feat_min, y_feat_min, 
                                (x_feat_max-x_feat_min), ratio_dec])

        # build horizontal bars
        enc_label_dummies = range(len(top10_encoder.index))
        ax_enc.barh(enc_label_dummies, top10_encoder.values, tick_label=top10_encoder.index, color=c, alpha=0.75)
        dec_label_dummies = range(len(decoder.index))
        ax_dec.barh(dec_label_dummies, decoder.values, tick_label=decoder.index, color='black', alpha=0.75)

        # labeling
        feat_axs = [ax_enc, ax_dec]
        for i,x in enumerate([top10_encoder, decoder]):
            rects = feat_axs[i].patches
            for rect, label in zip(rects, x.index):
                width = rect.get_width()
                feat_axs[i].text(
                    width+0.005, rect.get_y() + rect.get_height() / 2, label, style='italic'
                )

        # Remove axis ticks and frame
        [x.set_yticklabels([]) for x in feat_axs]
        [x.set_yticks([]) for x in feat_axs]
        [x.set_xticklabels([]) for x in feat_axs]
        [x.set_xticks([]) for x in feat_axs]
        [x.spines['top'].set_visible(False) for x in feat_axs]
        [x.spines['right'].set_visible(False) for x in feat_axs]
        [x.spines['bottom'].set_visible(False) for x in feat_axs]
        
        ## OUTPUT -----------------------------------------------------
        if save:
            plt.savefig(
                path, 
                format='pdf', 
                dpi=300,
                bbox_inches='tight'
            )
        plt.show()