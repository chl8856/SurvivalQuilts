import numpy as np
import pandas as pd
import os 
import pickle
import joblib

import warnings
warnings.filterwarnings('ignore')

import json
from pathlib import Path
from tqdm import tqdm


from sksurv.metrics import concordance_index_ipcw, brier_score, integrated_brier_score
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

import GPyOpt
from plugin_models import CoxPH, WeibullAFT, LogNormalAFT, LogLogisticAFT, RSF, ExtSurv, CoxBoost


def get_tte_structured(T,Y):
    tte_structured = [(np.asarray(Y)[i], np.asarray(T)[i]) for i in range(len(Y))]
    tte_structured = np.array(tte_structured, dtype=[('status', 'bool'),('time','<f8')])
    return tte_structured


class SurvivalQuilts():
    
    def __init__(self,                                 
                 baselines = { 'CoxPH': {'alpha': 0.05},
                                   'WeibullAFT': {'alpha': 0.05, 'penalizer':0.01, 'l1_ratio':0.},
                                   'LogNormalAFT': {'alpha': 0.05, 'penalizer':0.01, 'l1_ratio':0.},
                                   'LogLogisticAFT': {'alpha': 0.05, 'penalizer':0.01, 'l1_ratio':0.},
                                   'RSF': {'n_estimators': 100},  
                                   'ExtSurv': {'n_estimators': 100},
                                   'CoxBoost': {'n_estimators': 100} },
                num_validation = 10,
                num_outer      = 2,
                num_bo         = 30,
                random_seed    = 1234,
                path           = './'
        ):

        self.num_validation    = num_validation
        self.num_outer         = num_outer
        self.num_bo            = num_bo
        
        self.lmbda             = 0.
        self.rho               = 0.5
        self.step_ahead        = 3
        
        

        self.baselines         = baselines
        self.num_baselines     = len(self.baselines)

        self.baselines_indexes = [m for m in range(self.num_baselines)]

        self.ens_domain        = [{'name': 'w_' + str(m), 'type': 'continuous', 'domain': (0,1),'dimensionality': 1} for m in range(self.num_baselines)] 


        self.log_folder  = path+'/log/'     # path to saved the validation models
        self.save_folder = path+'/saved/'  # path to svaed the final model

        self.seed              = random_seed    

        

    def _make_ModelList(self):
        models = []
        for m in list(self.baselines):
            if m == 'CoxPH':
                models += [CoxPH(**self.baselines[m])]
            elif m == 'WeibullAFT':
                models += [WeibullAFT(**self.baselines[m])]
            elif m == 'LogNormalAFT':
                models += [LogNormalAFT(**self.baselines[m])]
            elif m == 'LogLogisticAFT':
                models += [LogLogisticAFT(**self.baselines[m])]
            elif m == 'RSF':
                models += [RSF(**self.baselines[m])]
            elif m == 'ExtSurv':
                models += [ExtSurv(**self.baselines[m])]
            elif m == 'CoxBoost':
                models += [CoxBoost(**self.baselines[m])]
            else:
                models += [] # ignore models not in the baseline candidates
        return models
    


    def _get_trained_models(self, X, T, Y):
        models_ = self._make_ModelList()

        for m in range(self.num_baselines):
            print(models_[m])        
            models_[m].fit(X, T, Y)

        return models_
    

    def _get_normalized_X_step(self,X_step):
        for k in range(np.shape(X_step)[0]):
            X_step[k, :] = X_step[k, :]/(np.sum(X_step[k, :])+1e-8)
        return X_step


    def _get_AL_constraint(self,g, beta, lmbda, rho):
        return np.asarray(lmbda * (g-beta) + 0.5 / rho * max(0,(g-beta))**2).reshape([-1,1])


    def _fit_ValidationSet(self, X, T, Y, cv_itr):
        tmp_log_folder = self.log_folder + '/itr_{}/'.format(cv_itr) 

        if not os.path.exists(tmp_log_folder):
            os.makedirs(tmp_log_folder)

        skf = KFold(n_splits=self.num_validation, random_state=self.seed, shuffle=True)

        tr_indx = list(skf.split(X))[cv_itr][0]
        va_indx = list(skf.split(X))[cv_itr][1]

        tr_X, va_X = X.loc[tr_indx].reset_index(drop=True), X.loc[va_indx].reset_index(drop=True)
        tr_Y, va_Y = Y.loc[tr_indx].reset_index(drop=True), Y.loc[va_indx].reset_index(drop=True)
        tr_T, va_T = T.loc[tr_indx].reset_index(drop=True), T.loc[va_indx].reset_index(drop=True)
                
        tr_tte_structured = get_tte_structured(tr_T, tr_Y)
        va_tte_structured = get_tte_structured(va_T, va_Y)
        
        # to compute brier-score without error
        va_T2 = va_T.copy(deep=True)
        va_T2.loc[va_T['time'] > tr_T['time'].max(), 'time'] = tr_T['time'].max()
        va_tte_structured2 = get_tte_structured(va_T2, va_Y)
        
        # train models for a given cross-validation
        baselines = self._get_trained_models(tr_X, tr_T, tr_Y)
        
        metric_CINDEX = np.zeros([self.num_baselines, len(self.time_optimization)])
        metric_BRIER = np.zeros([self.num_baselines, len(self.time_optimization)])
        
        for m_idx in range(self.num_baselines):

            pred      = baselines[m_idx].predict(va_X, self.time_optimization)
            
            log_name1 = tmp_log_folder + 'cv{}_baseline{}.joblib'.format(cv_itr, m_idx)
            log_name2 = tmp_log_folder + 'cv{}_baseline{}_pred.csv'.format(cv_itr, m_idx)

            joblib.dump(baselines[m_idx], log_name1)
            # np.savetxt(log_name2, pred, delimiter=',')
            
            for t, eval_time in enumerate(self.time_optimization):
                metric_CINDEX[m_idx, t] = concordance_index_ipcw(tr_tte_structured, va_tte_structured, pred[:,t], tau=eval_time)[0]
                metric_BRIER[m_idx, t]  = brier_score(tr_tte_structured, va_tte_structured2, 1.- pred[:,t], times=eval_time)[1][0]
                       
        return baselines, metric_CINDEX, metric_BRIER

        
    def _get_Y_step(self, W, X, T, Y, CV_models, time_horizons, K_step):

        metric_CINDEX, metric_BRIER = np.zeros([self.num_validation]), np.zeros([self.num_validation])

        skf = KFold(n_splits=self.num_validation, random_state=self.seed, shuffle=True)
       
        for cv_itr in range(self.num_validation):
            tr_indx = list(skf.split(X))[cv_itr][0]
            va_indx = list(skf.split(X))[cv_itr][1]

            va_X       = X.loc[va_indx].reset_index(drop=True)
            tr_Y, va_Y = Y.loc[tr_indx].reset_index(drop=True), Y.loc[va_indx].reset_index(drop=True)
            tr_T, va_T = T.loc[tr_indx].reset_index(drop=True), T.loc[va_indx].reset_index(drop=True)

            tr_tte_structured = get_tte_structured(tr_T, tr_Y)
            va_tte_structured = get_tte_structured(va_T, va_Y)
            
            # to compute brier-score without error
            va_T2 = va_T.copy(deep=True)
            va_T2.loc[va_T['time'] > tr_T['time'].max(), 'time'] = tr_T['time'].max()
            va_tte_structured2 = get_tte_structured(va_T2, va_Y)

            pred = self._get_ensemble_prediction(CV_models[cv_itr], W, va_X, time_horizons=self.time_optimization)
            
            new_K_step = min(K_step + 1 + self.step_ahead, len(self.time_optimization))
            for k in range(K_step, new_K_step):
                metric_CINDEX[cv_itr] += 1./(new_K_step - K_step) * concordance_index_ipcw(tr_tte_structured, va_tte_structured, pred[:,k], tau=self.time_optimization[k])[0]            
                metric_BRIER[cv_itr] += 1./(new_K_step - K_step) * brier_score(tr_tte_structured, va_tte_structured2, 1.- pred[:,k], times=self.time_optimization[k])[1][0]
        
        return (- metric_CINDEX.mean(),1.96*np.std(metric_CINDEX)/np.sqrt(self.num_validation)), (metric_BRIER.mean(),1.96*np.std(metric_BRIER)/np.sqrt(self.num_validation))


    def _get_ensemble_prediction(self, models, W, X, time_horizons=[]):
        # print('conducting ensemble prediction...')
        
        if len(time_horizons) == 0:
            time_horizons = self.all_time_horizons
        
        time_horizons_superset = np.unique(self.time_optimization + time_horizons).astype(int).tolist()

        pred = np.zeros([len(X), len(time_horizons_superset)])

        for m_idx in range(len(models)):
            # print('ing...' + models[m_idx].name)
            tmp_pred1 = models[m_idx].predict(X, time_horizons_superset)
            tmp_pred2 = models[m_idx].predict(X, self.time_optimization)

            for tt in range(len(self.time_optimization)):
                if tt == 0:
                    tmp_time_idx1 = np.asarray(time_horizons_superset) <= self.time_optimization[tt]
                    tmp_time_idx2 = np.asarray(time_horizons_superset) >  self.time_optimization[tt]

                    prev_val      = np.zeros([len(X), 1])
                    next_val      = tmp_pred2[:, [tt]]

                    increment              = tmp_pred1[:, tmp_time_idx1] - prev_val
                    pred[:, tmp_time_idx1] = pred[:, tmp_time_idx1] + W[tt,m_idx] * increment                
                    pred[:, tmp_time_idx2] = pred[:, tmp_time_idx2] + W[tt,m_idx] * (next_val - prev_val)

                elif tt == len(self.time_optimization) - 1: #the last index  
                    # tmp_time_idx1 = (np.asarray(time_horizons_superset) > self.time_optimization[tt-1]) & (np.asarray(time_horizons_superset) <= self.time_optimization[tt])
                    tmp_time_idx1 = (np.asarray(time_horizons_superset) > self.time_optimization[tt-1])
                    prev_val      = tmp_pred2[:, [tt-1]]
                    
                    increment              = tmp_pred1[:, tmp_time_idx1] - prev_val
                    pred[:, tmp_time_idx1] = pred[:, tmp_time_idx1] + W[tt,m_idx] * increment                

                else:
                    tmp_time_idx1 = (np.asarray(time_horizons_superset) > self.time_optimization[tt-1]) & (np.asarray(time_horizons_superset) <= self.time_optimization[tt])
                    tmp_time_idx2 = np.asarray(time_horizons_superset) >  self.time_optimization[tt]
                    prev_val      = tmp_pred2[:, [tt-1]]
                    next_val      = tmp_pred2[:, [tt]]
                    
                    increment              = tmp_pred1[:, tmp_time_idx1] - prev_val
                    pred[:, tmp_time_idx1] = pred[:, tmp_time_idx1] + W[tt,m_idx] * increment                
                    pred[:, tmp_time_idx2] = pred[:, tmp_time_idx2] + W[tt,m_idx] * (next_val - prev_val)
                    
        return pred[:, [f_idx for f_idx, f in enumerate(time_horizons_superset) if f in time_horizons]]
    
    

    def fit(self, X, T, Y, time_optimization=[]):
        
        # path to saved validation and final models.
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)        
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # set time_horizons as the optimization points
        if len(time_optimization) == 0:
            self.time_optimization = []
            for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]:     
                self.time_optimization += [int(np.percentile(np.asarray(T[Y.iloc[:,0] == 1]), p))]
        else:
            self.time_optimization = time_optimization

        # this is for predicting overall all time-horizons
        self.all_time_horizons = [t for t in range(int(np.min(T[Y.iloc[:,0] == 1])), int(np.max(T[Y.iloc[:,0] == 1])))]
        

        # fit models on the validation set to make GPs for the black-box functions
        CV_models = [] 

        metric_CINDEX = np.zeros([self.num_validation, self.num_baselines, len(self.time_optimization)])
        metric_BRIER  = np.zeros([self.num_validation, self.num_baselines, len(self.time_optimization)])

        for cv_itr in range(self.num_validation):
            print('training on validation set.. validation no.' + str(cv_itr))

            cv_model, tmp_CINDEX, tmp_BRIER = self._fit_ValidationSet(X, T, Y, cv_itr)

            CV_models.append(cv_model)
            metric_CINDEX[cv_itr,:,:] = tmp_CINDEX
            metric_BRIER[cv_itr,:,:]  = tmp_BRIER

        # print a summary of trained models on cross-validations (for creating black-box)
        print('C-INDEX: ')
        print(pd.DataFrame(np.mean(metric_CINDEX, axis=0), index=list(self.baselines), columns=self.time_optimization))

        print('\n')
        print('BRIER-SCORE: ')
        print(pd.DataFrame(np.mean(metric_BRIER, axis=0), index=list(self.baselines), columns=self.time_optimization))


        ##### BO TRAINING  
        # initialization
        print('======= INITIALIZATION =======')        
        X_inits = np.zeros([1,self.num_baselines])
        X_inits[0, np.argmax(np.mean(metric_CINDEX, axis=0)[:,0])]=1  #put more weights on the "best" one (at the first step)
        X_inits = self._get_normalized_X_step(X_inits)

        W_prev      = np.zeros([len(self.time_optimization), self.num_baselines])
        W_prev[:,:] = X_inits

        for k in range(len(self.time_optimization)):
            lmbda       = self.lmbda
            rho         = self.rho

            beta = np.median(np.mean(np.mean(metric_BRIER, axis=0)[:,k:(k+self.step_ahead+1)], axis=1))

            # initialization for k step-ahead time steps
            X_inits = np.zeros([1,self.num_baselines])
            X_inits[0, np.argmax(np.mean(metric_CINDEX, axis=0)[:,k])]=1  #put more weights on the "best" one (at the first step)
            
            X_inits = self._get_normalized_X_step(X_inits)

            W = np.copy(W_prev)
            W[k:,:] = X_inits

            Yo_inits = []
            Yc_inits = []

            tmp_o_prev, tmp_c_prev = self._get_Y_step(W_prev, X, T, Y, CV_models, self.time_optimization, k)
            tmp_o, tmp_c           = self._get_Y_step(W, X, T, Y, CV_models, self.time_optimization, k)

            Yo_next = np.asarray(tmp_o[0])
            Yc_next = self._get_AL_constraint(tmp_c[0], beta, lmbda, rho)

            Yo_inits.append(Yo_next)
            Yc_inits.append(Yc_next)

            Yo_inits = np.asarray(Yo_inits).reshape([-1,1])
            Yc_inits = np.asarray(Yc_inits).reshape([-1,1])


            # BO update
            for out_itr in tqdm(range(self.num_outer), desc='BO at K = {}'.format(k)):
                X_step_ens   = X_inits
                Y_step_ens   = Yo_inits + Yc_inits

                for itr in range(self.num_bo):  # replace this with tqdm
                    tmp = GPyOpt.methods.BayesianOptimization(f = None, domain = self.ens_domain, X = X_step_ens, 
                                                                        Y = Y_step_ens, acquisition_type='EI', 
                                                                        model_type='GP', exact_feval = True,
                                                                        cost_withGradients=None)

                    X_next = tmp.suggest_next_locations()
                    X_next = self._get_normalized_X_step(X_next)
                    W[k:, :] = X_next

                    if itr < (self.num_bo-1):
                        tmp_o, tmp_c = self._get_Y_step(W, X, T, Y, CV_models, self.time_optimization, k)
                        
                        Yo_next = np.asarray(tmp_o[0]).reshape([-1,1])
                        Yc_next = self._get_AL_constraint(tmp_c[0], beta, lmbda, rho)
                        Y_next  = Yo_next + Yc_next

                        X_step_ens = np.vstack([X_step_ens, X_next])
                        Y_step_ens = np.vstack([Y_step_ens, Y_next])

                print('=========== BO Finished ===========')


                GP_ens = tmp.model.model

                if GP_ens is not None:
                    X_opt    = X_step_ens[np.argmin(Y_step_ens,axis=0)]

                    W[k:, :] = X_opt
                    print('out_itr: ' + str(out_itr) + ' | BEST X: ' + str(X_opt) )
                    tmp_o, tmp_c = self._get_Y_step(W, X, T, Y, CV_models, self.time_optimization, k)
                    print(tmp_o[0])

                    if max(0, tmp_c[0] - beta) < 0.005*beta: #0.5% off from the median
                        print('====================================')
                        print('THRESHOLD SATISFIED')
                        print('BEST: ' + str(X_opt))
                        print('Objective val.: ' + str(tmp_o[0]))
                        print('Constraint val.: ' + str(tmp_c[0]))
                        print('====================================')
                        break

                else:
                    print('nothing found...')
                    X_opt = np.random.randint(len(models), size=(1, 3))
                    W[k:, :] = X_opt
                    print('out_itr: ' + str(out_itr) + ' | BEST X: ' + str(X_opt) )
                    tmp_o, tmp_c = self._get_Y_step(W, X, T, Y, CV_models, self.time_optimization, k)


                lmbda = max(0, lmbda + 1./rho * tmp_c[0])

                if tmp_c[0] <= 0.:
                    rho = rho
                else:
                    rho = rho/2.

                X_inits  = X_opt
                Yo_inits = np.asarray(tmp_o[0]).reshape([-1,1])
                Yc_inits = self._get_AL_constraint(tmp_c[0], beta, lmbda, rho)

                print('out_itr: ' + str(out_itr) + ' | Lambda: ' + str(lmbda) + ' | Rho: ' + str(rho))


            thres_split = abs(tmp_o_prev[0] * 0.005) # 0.5% improvement -> update
            print('GAP: ' + str(-(tmp_o[0] - tmp_o_prev[0])))
            print('THRES: ' + str(thres_split))
            if -(tmp_o[0] - tmp_o_prev[0]) > thres_split: # since tmp_o is negative C-index
                W_prev = np.copy(W)   #only update if W is significantly better

        self.final_W      = W_prev
        self.final_models = self._get_trained_models(X, T, Y)
        self.final_X_step_ens = X_step_ens
        self.final_Y_step_ens = Y_step_ens

        # save models
        np.save(self.save_folder + 'W.npy', self.final_W)
        for m_idx, model in enumerate(self.final_models):
            joblib.dump(model, self.save_folder + 'model{}.joblib'.format(m_idx))
    
    
    def predict(self, X, time_horizons=[]):    
        if len(time_horizons) == 0:
            time_horizons = self.all_time_horizons
        return self._get_ensemble_prediction(self.final_models, self.final_W, X, time_horizons)
