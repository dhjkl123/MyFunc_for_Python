from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import *

def forward_stepwise_linear(x_train, y_train):

    # 변수목록, 선택된 변수 목록, 단계별 모델과 AIC 저장소 정의
    features = list(x_train)
    selected = []
    step_df = pd.DataFrame({ 'step':[], 'feature':[],'aic':[]})

    # 
    for s in range(0, len(features)) :
        result =  { 'step':[], 'feature':[],'aic':[]}

        # 변수 목록에서 변수 한개씩 뽑아서 모델에 추가
        for f in features :
            vars = selected + [f]
            x_tr = x_train[vars]
            model = OLS(y_train, add_constant(x_tr)).fit()
            result['step'].append(s+1)
            result['feature'].append(vars)
            result['aic'].append(model.aic)
        
        # 모델별 aic 집계
        temp = pd.DataFrame(result).sort_values('aic').reset_index(drop = True)

        # 만약 이전 aic보다 새로운 aic 가 크다면 멈추기
        if step_df['aic'].min() < temp['aic'].min() :
            break
        step_df = pd.concat([step_df, temp], axis = 0).reset_index(drop = True)

        # 선택된 변수 제거
        v = temp.loc[0,'feature'][s]
        features.remove(v)

        selected.append(v)
    
    # 선택된 변수와 step_df 결과 반환
    return selected, step_df


def plot_feature_importance(importance, names):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    fi_df.reset_index(drop=True, inplace = True)

    plt.figure(figsize=(10,8))
    sns.barplot(x='feature_importance', y='feature_names', data = fi_df)

    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.grid()

    return fi_df

def Regressor_report(y_val,pred):
    print('MAE : ', mean_absolute_error(y_val,pred))
    print('MAPE : ' , mean_absolute_percentage_error(y_val,pred))
    print('RMSE : ',mean_squared_error(y_val,pred,squared=False))
    
    
def mean_test_score_lineplot(df,x,hue):
    tmp = df.loc[:,[x,'mean_test_score',hue]]

    plt.figure(figsize=(8,6))
    sns.lineplot(x=x,y='mean_test_score',data=tmp,hue=hue)
    plt.grid()
    
def Regressors_report(desc,pred,y_val):
    RMSE, MAE, MAPE = [],[],[]
    
    for i, p in enumerate(pred) :
        RMSE.append(mean_squared_error(y_val, p, squared=False))
        MAE.append(mean_absolute_error(y_val, p))
        MAPE.append(mean_absolute_percentage_error(y_val, p))

    result = pd.DataFrame({'model_desc':desc,'RMSE':RMSE,'MAE':MAE,'MAPE':MAPE})
    return result
    
    