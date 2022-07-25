#!/usr/bin/env python
# coding: utf-8

# # Machine Learning - Classification - Supervised Learning 
# 
# 
# ***Developer by ***: Larissa Lorenzi
# 
# ***Date:*** Jul/2022

# # Imports

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

plt.rcParams['figure.figsize'] = (6.0, 5.0)

#setup the rows and cols dimension
pd.set_option('max_rows', 200)
pd.set_option('max_columns', 1000)

#to change the scientific notation precision
pd.set_option('display.float_format', lambda x: '%.2f' % x)

import warnings
warnings.filterwarnings('ignore')

path = os.environ['USERPROFILE']
path


# **Função para exibir gráficos de precisão**

# In[2]:


def plot_accs(values, accs_train, accs_test):
    plt.subplots(figsize=(10, 8))
    plt.plot(values, accs_train, label='train')
    plt.plot(values, accs_test, label='test')
    plt.ylabel('Accuracy')
    plt.xlabel('Max Depth')
    plt.grid()
    plt.legend()
    #plot_accs(values, accs_train, accs_test)


# **Funções para exibir o mapa de calor com correlações de dataframe**

# In[3]:


def dheatmap (dfCorr, sWidth, sHeight, sYlim):
    plt.subplots(figsize=(sWidth, sHeight)) # Set up the matplotlib figure
    maskTriu = np.triu(dfCorr)  #applying a mask for the triangle in the left side.
    s = sns.heatmap(dfCorr, mask=maskTriu, annot=True, cmap="YlGnBu", vmax=1,    center=0,square=True, linewidths=.5,cbar_kws={"shrink": .5})
    s.set_ylim(sYlim, 0.0)
    s.set_xticklabels(s.get_xticklabels(), rotation = 60)


# **Função para exibir a comparação com valores verdadeiros e valores previstos**

# In[4]:


def plot_compar(ytrue, ypred):
    plt.figure(figsize=(10, 6))
    plt.plot(ytrue, label='True Values')
    plt.plot(ypred, label='Predict Values')
    plt.title("Prediction")
    plt.xlabel('Observation')
    plt.ylabel('Values')
    plt.legend()
    plt.show();


# In[5]:


import plotly
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio
plotly.offline.init_notebook_mode()
pio.templates.default = "plotly_white"

def plotlyCompar(ytrue, ypred):
    # Create traces
    trace0 = go.Scatter(
        y = ytrue,    
        mode = 'lines',
        line={"color": 'orange'},  
        name = 'True values'
    )
    trace1 = go.Scatter(
        y =  ypred,
        mode = 'lines',
        line={"color": '#1f78b4' },  
        name = 'Predict values'

    )

    data = [trace0, trace1]
    py.iplot(data, filename='scatter-mode')


# Criei uma ***árvore de decisão de classificação*** em Python para fazer previsões

# # Credit Risk Score 
# 
# O programa vai prever se um cliente tem um risco de crédito bom ou ruim

# <center>
# <img src="img/Collection.png" alt="drawing",  width="100" height="200">
# </center>

# In[6]:


#loading the dataset

dfcredit = pd.read_csv(path + '\\CitiDSPython\\data\\CreditScoreRisk.csv')


# In[7]:


dfcredit.head()


# In[8]:


dfcredit.shape


# In[9]:


dfcredit.info()


# In[10]:


#The correlation betweeen features, In this case we can see only numeric features
dheatmap(dfcredit.corr(), 12, 8, 9) #(dfCorr, sWidth, sHeight, sYlim):


# # Recurso de destino

# In[11]:



#(0 = Good,  1 = Bad)


#the target is what we would like to predict
dfcredit['BadCredit'][:3]


# # Classes desequilibradas
# Vemos que o número de recursos **bad_credit** é mais bom do que ruim

# In[12]:


import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,4))
dfcredit.groupby('BadCredit').customerID.count().plot.bar(ylim=0, color=[ '#1f78b4', 'orange'])
plt.show()

#(0 = Good,  1 = Bad)


# # Representação de texto/categórica
# Usei ***One Hot Encoding*** para converter recursos de texto/categóricos.

# In[13]:


dfcredit.info()


# In[14]:


dfcredit.columns.values


# In[15]:


cols_to_modify = ['chkgAccountStatus', 'CreditHistory', 'Purpose', 'SavingsAcctBalance', 'TimeEmployed', 'GenderStatus', 'OtherSignators', 'Property', 'OtherCredit', 'HomeOwner', 'JobCategory', ' Telephone','ForeignWorker' ]

cols_to_keep = list(set(dfcredit.columns.values).difference(cols_to_modify))


# In[16]:


cols_to_modify


# In[17]:


cols_to_keep


# In[18]:


dfcreditNew = pd.get_dummies(dfcredit, columns=cols_to_modify)


# **Verificando o tamanho do conjunto de dados após a função get_dummies**

# In[19]:


dfcreditNew.shape


# In[20]:


dfcreditNew[:10]


# In[21]:


#correlation
dheatmap(dfcreditNew.corr(), 25, 20, 60) #(dfCorr, sWidth, sHeight, sYlim):


# **Obtaing the best correlactions**

# In[22]:


correlations = dfcreditNew.corr()['BadCredit'].sort_values()


# In[23]:


print('Most Positive Correlations: \n\n' , correlations.tail(7)*100)


# In[24]:


print('Most Negative Correlations: \n\n', correlations.head(7)*100)


# In[25]:


#Checking the correlation again with specific columns
dfcreditNew =dfcreditNew[[ 'LoanAmount', 
                             'LoanDuration', 
                             'SavingsAcctBalance_< 100 DM', 
                             'SavingsAcctBalance_unknown/none',
                             'CreditHistory_all loans at bank paid',
                             'CreditHistory_no credit - paid',
                             'CreditHistory_critical account - other non-bank loans',
                             'chkgAccountStatus_< 0 DM',
                             'chkgAccountStatus_none', 
                              'HomeOwner_own', 
                              'Property_real estate',
                              'OtherCredit_none',
                              'Purpose_radio/television',
                              'BadCredit']]

 
dheatmap(dfcreditNew.corr(), 20, 15, 14)


# # Obtendo o conjunto de dados para árvores de decisão

# In[26]:


#Predictors feature
#selecting feature over 12% correlated with BadCredit

X = dfcreditNew[['LoanAmount', 
                 'LoanDuration', 
                 'SavingsAcctBalance_< 100 DM', 
                 'SavingsAcctBalance_unknown/none',
                 'CreditHistory_all loans at bank paid',
                 'CreditHistory_no credit - paid',
                 'CreditHistory_critical account - other non-bank loans',
                 'chkgAccountStatus_< 0 DM',
                 'chkgAccountStatus_none', 
                  'HomeOwner_own', 
                  'Property_real estate']]

#target feature
y = dfcreditNew['BadCredit']

print (X.shape)
print (y.shape)


# In[27]:


X.head()


# In[28]:


y.head()


# # Divisão do conjunto de dados

# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


#In the following code, test_size is 0.3,  which means 70% of the data should be split into the training dataset and the remaining 30% should be in the testing dataset.

#Random_state is seed in this process of generating pseudo-random numbers, which makes the results reproducible by splitting the exact same observations while running every time:
    
X_wtrain, X_wtest, y_wtrain, y_wtest = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
 


# In[32]:


print (X_wtrain.shape, y_wtrain.shape)
print (X_wtest.shape, y_wtest.shape)


# In[33]:


X_wtrain[:3]


# In[34]:


y_wtrain[:3]


# In[35]:


X_wtest[:3]


# In[36]:


y_wtest[:3]


# # Parâmetros:
# 
# 
# Os parâmetros com os quais o algoritmo deve construir a árvore, pois segue uma abordagem recursiva para construir a árvore, então estabeleci alguns limites para criar
# 
# 
# **X_wtrain:** conjunto de dados de recursos de preditores de treinamento
# 
# **y_wtrain:** conjunto de dados de recursos de destino de treinamento
# 
# **X_wtest:** Conjunto de dados de variáveis de previsão de teste: y_wtrain
# 
# **y_wtest:** Conjunto de dados da variável de destino de teste: y_wtest
# 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_leaf=15)
dtc

#**dtc** object i also the **model**


# <img>

# ***criterion:*** ajustar a precisão do modelo, também o algoritmo para construir a árvore, deixar de avaliar os galhos em que nenhuma melhoria é obtida de acordo com a função de perda.
# 
# ***max_depth:*** Número máximo de níveis que a árvore terá.
# 
# ***min_samples_leaf:*** Este parâmetro é otimizável e indica o número mínimo de amostras que queremos ter nas folhas.

# In[ ]:


#fit is the responsible by training
dtc = dtc.fit(X_wtrain, y_wtrain)
dtc


# # Predição

# In[ ]:





# In[ ]:


# Predict method returns the predictions regarding the new objects/features
y_wpred = dtc.predict(X_wtest)


# In[ ]:


#Predictors
X_wtest[:3]


# In[ ]:


#true values
y_wtest[:3]


# In[ ]:


#Predictions
y_wpred[:3]


# In[ ]:



#table Comparing True and Prediction values
dfPred  = X_wtest[:]
dfPred['True Values'] =y_wtest[:]
dfPred['Predicted Values'] =y_wpred[:]
dfPred['Prediction Error'] = dfPred['True Values'].sub(dfPred['Predicted Values'] )
dfPred.tail(10)


# # Análise de performance

# In[ ]:


from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

#y_wpred = dtc.predict(X_wtest)
print("Accuracy:", metrics.accuracy_score(y_wtest, y_wpred))


# In[ ]:


from sklearn.metrics import classification_report
report = classification_report(y_wtest, y_wpred)
print(report)


# In[ ]:





# Usando esse conjunto de dados:
# 
# * Treinei vários modelos variando o valor do parâmetro 'max_depth' entre 1 e 20
# * Tracei um gráfico com duas curvas, 'max_depth vs. precisão' no conjunto de treinamento, 'max_depth vs. precisão' no conjunto de teste.
# 
# Para plotagens, usei a função `plot_accs(values, accs_train, accs_test, param_name)`.

# In[ ]:


## splitting the base - 20% for test
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=1,stratify=y)

train_accs = []
test_accs = []

depths = list(range(1, 20))


# In[ ]:


depths


# In[ ]:



for depth in depths:
 
    dt = DecisionTreeClassifier(random_state=1, max_depth=depth, min_samples_leaf=15)  
    
    dt.fit(xtrain, ytrain)    
    
    ytest_pred = dt.predict(xtest) 
    
    ytrain_pred = dt.predict(xtrain) 
    
    test_accs.append(metrics.accuracy_score(ytest, ytest_pred))
    
    train_accs.append(metrics.accuracy_score(ytrain, ytrain_pred))
    
#     ...


plot_accs(depths, train_accs, test_accs)


# In[ ]:




