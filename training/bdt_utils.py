import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, roc_curve

pd.options.mode.chained_assignment = None

from timeit import default_timer as timer

class BDT(object):

    # FIXME: add option for dropping negative weights
    def __init__(self, data, train_vars, train_frac=0.7, eq_weights=True, neg_weights="abs", options=None, proc_map={"bkg":0,"ggH":1,"VBF":2}):

        self.data = data
        self.train_vars = train_vars
        self.train_frac = train_frac
        self.options = options
        self.proc_map = proc_map
        self.seed = 1357
        self.weight_var = "weight"
        self.neg_weights = neg_weights
        self.eq_weights = eq_weights
        self.weight_scaler = 1e5

        # Class data variables
        self.X_train = None
        self.y_train = None
        self.train_weights = None
        self.train_weights_derived = None
        self.y_pred_train = None

        self.X_test = None
        self.y_test = None
        self.test_weights = None
        self.test_weights_derived = None
        self.y_pred_test = None

        # Create classifier
        self.clf = xgb.XGBClassifier(objective='mlogloss', n_estimators=100, 
                                     eta=0.05, max_depth=6, #min_child_weight=1, 
                                     subsample=0.6, colsample_bytree=0.6, gamma=1)

        # Delete data object from memory
        del data


    # Function to create X train/test and y train/test
    def create_X_and_y(self):

        # Add y target label
        y = {}
        y_list = []
        for k,v in self.proc_map.items():
            y[k] = self.data['proc_id']==v
            y_list.append(y[k])
        y_vals = np.arange(len(y_list))
        Y = np.select(y_list, y_vals)
           
        #y_ggH = self.data['proc_id']==1
        #y_VBF = self.data['proc_id']==2
        #y_bkg = self.data['proc_id']==0
        #Y = np.select([y_ggH,y_VBF,y_bkg], [0,1,2])

        weights = abs( self.data[self.weight_var] ) if self.neg_weights=="abs" else self.data[self.weight_var]

        # Do splitting
        X_train, X_test, train_w, test_w, y_train, y_test, data_train, data_test = \
            train_test_split( self.data[self.train_vars], weights, Y, self.data,
                              train_size = self.train_frac, shuffle=True, random_state = self.seed )

        # Equalise weights in train and test samples
        if self.eq_weights:
            sumw = np.zeros_like(y_train, dtype='float64')
            for i in y_vals:
                sumw += (y_train==i)*(train_w.values[y_train==i].sum())
            train_w_final = train_w.values/sumw

            sumw = np.zeros_like(y_test, dtype='float64')
            for i in y_vals:
                sumw += (y_test==i)*(test_w.values[y_test==i].sum())
            test_w_final = test_w.values/sumw
        else:
            train_w_final = train_w.values
            test_w_final = test_w.values

        # Store objects
        self.X_train = X_train.values
        self.y_train = y_train
        self.train_weights = train_w_final*self.weight_scaler
        self.data_train = data_train

        self.X_test = X_test.values
        self.y_test = y_test
        self.test_weights = test_w_final*self.weight_scaler
        self.data_test = data_test

    # Function to train classifier
    def train_classifier(self):

        #train_weights = self.train_weights_derived if BLAH else self.train_weights
        train_weights = self.train_weights

        print(" --> Training classifier...")
        start = timer()
        clf = self.clf.fit( self.X_train, self.y_train, sample_weight = train_weights )
        end = timer()
        print(" --> Training finished, time taken: %.3f s"%(end-start))
        self.clf = clf

    # Evaluate classifier
    def evaluate_classifier(self):
        self.y_pred_train = self.clf.predict_proba( self.X_train )
        for i,k in enumerate(self.proc_map.keys()):
            self.data_train['y_pred_%s'%k] = self.y_pred_train.T[i] 
        self.y_pred_test = self.clf.predict_proba( self.X_test )
        for i,k in enumerate(self.proc_map.keys()):
            self.data_test['y_pred_%s'%k] = self.y_pred_test.T[i] 


    # Output dataframe
    def package_output(self):
        df_train = self.data_train
        df_train['dataset_type'] = 'train'
        df_train['weight_ml'] = self.train_weights
        df_test = self.data_test
        df_test['dataset_type'] = 'test'
        df_test['weight_ml'] = self.test_weights
        dfs = [df_train,df_test]
        return pd.concat(dfs)
