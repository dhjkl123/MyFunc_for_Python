import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as spst
import statsmodels.api as sm
from statsmodels.graphics.mosaicplot import mosaic

def Bivariate(data,x,y,kind,bins = 64):
    
    Nanflag = False
    res = None
    
    
    if data[x].isna().sum():
        print('x data has Nan')
        Nanflag = True
        
    if data[y].isna().sum():
        print('y data has Nan')
        Nanflag = True
    
    
    if kind == 'numnum' and Nanflag == False:
        sns.regplot(data[x],data[y])
        plt.show()

        res = spst.pearsonr(data[x],data[y])
        
        print('r = ',res[0])
        print('p-Val = ',res[0])
        
    elif kind == 'numcat' and Nanflag == False:  
        plt.subplot(2,2,1)
        sns.kdeplot(x=x, data = data, hue =y, common_norm = False)

        plt.subplot(2,2,2)
        sns.kdeplot(x=x, data = data, hue =y, multiple = 'fill')

        plt.subplot(2,2,3)
        sns.histplot(x=x, data = data, hue = y)

        plt.subplot(2,2,4)
        sns.histplot(x=x, data = data, bins = 64, hue =y, multiple = 'fill')

        plt.tight_layout()
        plt.show()

        model = sm.Logit(data[y], data[x])
        res = model.fit()
        print(res.pvalues)
        
    elif kind == 'catcat':
        tab = pd.crosstab(data[y], data[x], normalize = 'index')
        print(tab)


        mosaic(data,[x,y])
        plt.show()

        res = spst.chi2_contingency(tab)
        print('x^2 = ', res[0])
        print('p-Val = ', res[1])
        print('k = ',res[2])
        print('expected frequency\n',res[3])
        
    elif kind == 'catnum' and Nanflag == False:
        sns.barplot(x=x,y=y,data=data)
        plt.grid()
        plt.show()
        
        data_list = []

        cnt = len(data[x].unique())

        for i in range(0,cnt):
            data_list.append(data.loc[data[x] == i,y])

        res = spst.f_oneway(*data_list)
        print('t = ',res[0])
        print('p-Val = ',res[1])
        

 
    
    return res
    
    
def Unvariate(data,column,kind,opt = False):
    if kind == 'num':
        
        plt.subplot(1,2,1)
        sns.displot(data[column])
        
        plt.subplot(1,2,2)
        box = plt.boxplot(data[column],vert=opt)
        
        plt.tight_layout()
        plt.show()
        
        if opt:
            print('smaller : ',box['whiskers'][0].get_ydata()) 
            print('bigger : ',box['whiskers'][1].get_ydata())
        else:
            print('smaller : ',box['whiskers'][0].get_xdata()) 
            print('bigger : ',box['whiskers'][1].get_xdata())
        
        
        print(data[column].describe())
    elif kind =='cat':
        print(data[column].value_counts()/data.shape[0])
        
        sns.countplot(data[column])
        plt.show()
               
    return