import numpy as np
import pandas as pd

from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis, ExtraSurvivalTrees
from sksurv.linear_model import CoxPHSurvivalAnalysis
from lifelines import WeibullAFTFitter, LogLogisticAFTFitter, LogNormalAFTFitter


############# plug-in package: lifelines #############

class lifelinesSurvival( object ):
    """ A parent class for all survival estimators from sksuv. Particular survival models will inherit from this class."""
    # methods
    def __init__(self):     
        exec('1+1') # dummy instruction
        
    def fit(self,X,T,Y):
        # Put the data in the proper format # check data type first
        T.columns = ['time']
        Y.columns = ['event']
        
        # Add noise to avoid unstability issue due to T==0
        T_ = T.copy(deep=True)
        T_.iloc[:, 0] = T_.iloc[:, 0].apply(lambda x: x+np.abs(0.1*np.random.randn()) if x == 0 else x )
        
        self.model.fit(pd.concat([X, T_, Y], axis=1), 'time', 'event')

    def predict(self, X, time_horizons): 
        return 1. - self.model.predict_survival_function(X, times=time_horizons).T.to_numpy()


#-----------------------------------------------------------------
class WeibullAFT(lifelinesSurvival):
    """ Cox proportional hazard model."""

    def __init__(self, alpha=0.05, penalizer=0.05, l1_ratio=0):
        super(WeibullAFT, self).__init__()
        # super().__init__()

        self.name          = 'Weibull'
        self.explained     = "*Weibull AFT model"

        self.model         = WeibullAFTFitter(alpha=0.05, penalizer=penalizer, l1_ratio=l1_ratio)

    def get_hyperparameters(self):
        return {'alpha': self.model.alpha, 'penalizer': self.model.penalizer, 'l1_ratio': self.model.l1_ratio}


#-----------------------------------------------------------------
class LogNormalAFT(lifelinesSurvival):
    """ Cox proportional hazard model."""
    
    def __init__(self, alpha=0.05, penalizer=0.05, l1_ratio=0):
        super(LogNormalAFT, self).__init__()
        # super().__init__()

        self.name          = 'LogNormal'
        self.explained     = "*LogNormal AFT model"

        self.model         = LogNormalAFTFitter(alpha=0.05, penalizer=penalizer, l1_ratio=l1_ratio)
        

    def get_hyperparameters(self):
        return {'alpha': self.model.alpha, 'penalizer': self.model.penalizer, 'l1_ratio': self.model.l1_ratio}


#-----------------------------------------------------------------
class LogLogisticAFT(lifelinesSurvival):
    """ Cox proportional hazard model."""
    
    def __init__(self, alpha=0.0, penalizer=0.05, l1_ratio=0):
        super(LogLogisticAFT, self).__init__()
        # super().__init__()

        self.name          = 'LogLogistic'
        self.explained     = "*LogLogistic AFT model"

        self.model         = LogLogisticAFTFitter(alpha=0.05, penalizer=penalizer, l1_ratio=l1_ratio)
        

    def get_hyperparameters(self):
        return {'alpha': self.model.alpha, 'penalizer': self.model.penalizer, 'l1_ratio': self.model.l1_ratio}





############# plug-in package: scikit-survival #############
    
class sksurvSurvival( object ):
    """ A parent class for all survival estimators from sksuv. Particular survival models will inherit from this class."""
    # methods
    def __init__(self):     
        exec('1+1') # dummy instruction
        
    def fit(self,X,T,Y):
        # Put the data in the proper format # check data type first
        y = [(Y.iloc[i,0], T.iloc[i,0]) for i in range(len(Y))]
        y = np.array(y, dtype=[('status', 'bool'),('time','<f8')])
        # print(self.name)
        self.model.fit(X,y)

    def predict(self,X, time_horizons): 
        if self.name in ['CoxPH', 'CoxPHRidge']:
            surv = self.model.predict_survival_function(X)  #returns StepFunction object
            preds_ = np.zeros([np.shape(surv)[0], len(time_horizons)])
            for t, eval_time in enumerate(time_horizons):
                if eval_time > np.max(surv[0].x):  #all have the same maximum surv.x
                    eval_time = np.max(surv[0].x)
                preds_[:, t] = np.asarray([(1. - surv[i](eval_time)) for i in range(len(surv))])  #return cif at self.median_tte
        
        elif self.name in ['RSF', 'CoxBoost', 'ExtSurv']:
            tmp_pred = self.model.predict_survival_function(X)

            tmp_idx  = []
            for eval_time in time_horizons:
                if len(np.where(tmp_pred[0].x <= eval_time)[0]) == 0:
                    tmp_idx += [0]
                else:
                    tmp_idx += [np.where(tmp_pred[0].x <= eval_time)[0][-1]]

            preds_    = np.zeros([len(tmp_pred), len(tmp_pred[0].x)])
            for i in range(len(tmp_pred)):
                preds_[i, :] = 1. - tmp_pred[i].y
            preds_    = preds_[:, tmp_idx]
            
        else:
            preds_ = self.model.predict(X)

        return preds_



#-----------------------------------------------------------------
class CoxPH(sksurvSurvival):
    " Cox proportional hazard model "
    
    def __init__(self, alpha=0.05):
        super(CoxPH, self).__init__()
        # super().__init__()

        self.name          = 'CoxPH'
        self.explained     = "*Cox proportional model"

        self.model         = CoxPHSurvivalAnalysis(alpha=alpha)
        
        

    def get_hyperparameters(self):
        return {'alpha': self.model.alpha}
    

#-----------------------------------------------------------------
class RSF(sksurvSurvival):    
    " Random survival forest "

    def __init__(self, n_estimators=100):
        super(RSF, self).__init__()
        # super().__init__()

        self.name          = 'RSF'
        self.explained     = "*Random survival forest"

        self.model         = RandomSurvivalForest(n_estimators=n_estimators)
        

    def get_hyperparameters(self):
        return {'n_estimators': self.model.n_estimators}
    
    
#-----------------------------------------------------------------
class ExtSurv(sksurvSurvival):   
    def __init__(self, n_estimators=100):
        super(ExtSurv, self).__init__()
        # super().__init__()

        self.name          = 'ExtSurv'
        self.explained     = "*Extreme survival tree"

        self.model         = ExtraSurvivalTrees(n_estimators=n_estimators)
        
    def get_hyperparameters(self):
        return {'n_estimators': self.model.n_estimators}
    
    
#-----------------------------------------------------------------
class CoxBoost(sksurvSurvival):   
    def __init__(self, n_estimators=100):
        super(CoxBoost, self).__init__()
        # super().__init__()

        self.name          = 'CoxBoost'
        self.explained     = "*Gradient Boosting Survival Analysis"

        self.model         = GradientBoostingSurvivalAnalysis(n_estimators=n_estimators)
        

    def get_hyperparameters(self):
        return {'n_estimators': self.model.n_estimators}
    