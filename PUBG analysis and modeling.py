#!/usr/bin/env python
# coding: utf-8

# # Project: EDA PUBG
# ## Table of Contents
# <ul>
# <li><a href="#Dictionary">Data Dictionary</a></li>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# <li><a href="#model">Modeling </a></li>
# </ul> 

# In[1]:


# import the important libraries
import pandas as pd     # for dataframe
import numpy as np      # for arraies
import matplotlib.pyplot as plt  # for visualization 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns           # for visualization 


# In[2]:


# read the data
df = pd.read_csv('train_V2.csv')
df.head()


# <a id='Dictionary'></a>
# ## Data Dictionary

# - **groupId** - Integer ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
# - **matchId** - Integer ID to identify match. There are no matches that are in both the training and testing set.
# - **assists** - Number of enemy players this player damaged that were killed by teammates.
# - **boosts** - Number of boost items used.
# - **damageDealt** - Total damage dealt. Note: Self inflicted damage is subtracted.
# - **DBNOs** - Number of enemy players knocked.
# - **headshotKills** - Number of enemy players killed with headshots.
# - **heals** - Number of healing items used.
# - **killPlace** - Ranking in match of number of enemy players killed.
# - **killPoints** - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.)
# - **kills** - Number of enemy players killed.
# - **killStreaks** - Max number of enemy players killed in a short amount of time.
# - **longestKill** - Longest distance between player and player killed at time of death. This may be misleading, as downing a - player and driving away may lead to a large longestKill stat.
# - **maxPlace** - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
# - **numGroups** - Number of groups we have data for in the match.
# - **revives** - Number of times this player revived teammates.
# - **rideDistance** - Total distance traveled in vehicles measured in meters.
# - **roadKills** - Number of kills while in a vehicle.
# - **swimDistance** - Total distance traveled by swimming measured in meters.
# - **teamKills** - Number of times this player killed a teammate.
# - **vehicleDestroys** - Number of vehicles destroyed.
# - **walkDistance** - Total distance traveled on foot measured in meters.
# - **weaponsAcquired** - Number of weapons picked up.
# - **winPoints** - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.)
# - **winPlacePerc** - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.

# <a id='intro'></a>
# ## Introduction
# 
# 

# - In this project what we want is analyzing the data and find out what affects the player's win by answering some questions such as:
#     -  **Who are the highest win (Solos, Duos or Squads)?**
#     -  **What is the impact of damage on the number of kills?**
#     -  **What is affects of the player's win?**
#     -  **What is the amount of work as a team between players?**
# 
# 
# - **What's the best strategy to win in PUBG? Should you sit in one spot and hide your way into victory, or do you need to be the top shot? Let's let the data do the talking!**

# 
# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# 

# **We can divide the data into two parts to see all the fetures**

# In[13]:


df.iloc[:,:14].head(10)


# In[14]:


df.iloc[:,14:].head(10)


# In[9]:


df.info(null_counts=True)


# In[15]:


# for knowing statistical information
df.describe().T


# In[23]:


df.winPlacePerc.value_counts().head()


# In[29]:


df.winPlacePerc.plot('hist',bins=100)
plt.title('number of winer in each place')
plt.show()


# In[25]:


df.weaponsAcquired.value_counts().head()


# **Detect unique data in columns**

# In[16]:


df.nunique()


# In[18]:


# how many duplicats in the data
df.duplicated().sum()


# **What has happened so far is an attempt to explore and understand the data further, and find out which problems are in it and need cleaning**

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 

# In[34]:


# firstly let's drop the nan in the winPlacePerc feture 
df.dropna(axis=0,inplace=True)


# In[36]:


df.info(null_counts=1)


# ### now we are ready to Explor!

# # Who are the highest win (Solos, Duos or Squads)?

# In[38]:


df.matchType.value_counts()


# In[52]:


df.matchType.value_counts()


# In[55]:


plt.figure(figsize=(15,6))
plt.xticks(rotation=45)
ax = sns.barplot(df.matchType.value_counts().index, df.matchType.value_counts().values, alpha=0.8)
ax.set_title("Number of players in the same match type")
plt.show()


# ### now we will make it only 3 types [solo,duo,squad]

# In[76]:


df.matchType.replace(['squad-fpp','squad','normal-squad-fpp','normal-squad'],'squad',inplace=True)


# In[79]:


df.matchType.replace(['duo-fpp','normal-duo-fpp','normal-duo'],'duo',inplace=True)


# In[80]:


df.matchType.replace(['solo-fpp','normal-solo-fpp','normal-solo'],'solo',inplace=True)


# In[82]:


df.matchType.replace(['crashfpp','flaretpp','flarefpp','crashtpp'],'others',inplace=True)


# In[83]:


df.matchType.value_counts()


# In[86]:


sns.barplot(df.matchType.value_counts().index, df.matchType.value_counts().values)
plt.title('squad vs duo vs solo')
plt.show()


# **we can say that more than 50% of players play as a (squad)**

# In[113]:


plt.figure(figsize=(30,10))
plt.xticks(rotation=90)
sns.countplot(df.DBNOs,hue=df.matchType)
plt.title("DBNOs/match types")
plt.show()


# In[112]:


plt.figure(figsize=(30,6))
plt.xticks(rotation=90)
sns.countplot(df.revives,hue=df.matchType)
plt.title("revives/match types")
plt.show()


# In[111]:


plt.figure(figsize=(30,10))
plt.xticks(rotation=90)
sns.countplot(df.kills,hue=df.matchType)
plt.title("kills/match types")
plt.show()


# **so working as a team give you the chance to revived or make more damage and kills**

# # let's work on a war

# # What is the impact of damage on the number of kills?

# In[118]:


sns.scatterplot(df.kills,df.damageDealt)
plt.title('relation between kills and damage')
plt.show()


# In[127]:


kills = df[['headshotKills','killPlace','killPoints','kills'
            ,'killStreaks','longestKill','roadKills','teamKills'
            ,'weaponsAcquired','winPlacePerc']]


# In[128]:


kills.head()


# In[129]:


kills.nunique()


# In[149]:


kills_features = ['headshotKills','killPlace','killPoints','kills'
                  ,'killStreaks','longestKill','roadKills','teamKills','weaponsAcquired']
def scatter (feature_name):
        sns.scatterplot(x=feature_name,y='winPlacePerc',data=kills)


# In[152]:


plt.figure(figsize=(15,10))
plt.subplot(3,3,1)
scatter(kills_features[0])
plt.subplot(3,3,2)
scatter(kills_features[1])
plt.subplot(3,3,3)
scatter(kills_features[2])
plt.subplot(3,3,4)
scatter(kills_features[3])
plt.subplot(3,3,5)
scatter(kills_features[4])
plt.subplot(3,3,6)
scatter(kills_features[5])
plt.subplot(3,3,7)
scatter(kills_features[6])
plt.subplot(3,3,8)
scatter(kills_features[7])
plt.subplot(3,3,9)
scatter(kills_features[8])
plt.show()


# ## let's see the correlation

# In[161]:


kills.corr()


# In[155]:


plt.figure(figsize=(8,8))
sns.heatmap(kills.corr(), annot=True, linewidths=.5)


# ### Now we can say that the more weapons you carry, and the more kills you have killed, the greater your chance of getting an advanced center.
# 
# ### Notice that the more dead your enemy kills. Much lower your chance of getting the top position

# # Mobility in the match

# - Running
# - Swimming
# - Driving

# In[166]:


mobility=df[['rideDistance','roadKills','swimDistance','vehicleDestroys','walkDistance','winPlacePerc']]


# In[168]:


mobility.head()


# In[170]:


# Then you map to the grid
g = sns.PairGrid(mobility)
g.map(plt.scatter)


# In[172]:


sns.heatmap(mobility.corr(),annot=True, linewidths=.5)


# **We can see that all these features have positive relationships with a  , but I struggled is that there is a very strong positive relationship between the distance that the player walks and win place Followed Ride distance**

#  <a id='conclusions'></a>
# # Conclusions

# **After analyzing and understanding the data is time to answer a question:**
# # What's the best strategy to win in PUBG?

# ### Whenever you played in a team increased your chance to win and find aid to have other opportunities, In addition to increasing the number of killers you kill them , Taking into account that if the competitor kills more than you, you are an exhibition of loss so you must develop your skills in murder from distances and focus when shooting on the head, Must move so much until they do not give an opportunity to hit you and always try to collect weapons and find vehicles

# **------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------**

# <a id='model'></a>
# # Modeling  
# **it's time to build a model that predicts the win place for players**

# ## Table of Contents
# <ul>
# <li><a href="#feature">Feature Selection </a></li>
# <li><a href="#prepro">Data Preprocessing</a></li>
# <li><a href="#Train">Select and Train the Model</a></li>
# </ul> 

# In[3]:


data_train = df.copy()


# <a id='feature'></a>
# # Feature Selection

# In[7]:


data_train.head()


# In[9]:


plt.figure(figsize=(20,20))
sns.heatmap(data_train.corr(),annot=True, linewidths=.5)


# **We can see that there are features between them, very strong relationships**

# **there are some features Because there is a strong relationship between them and other variables, or because they do not affect winning** 
# ### [killPoints,matchDuration,maxPlace,numGroups,rankPoints,roadKills,teamKills,winPoints,killStreaks,longestKill,killPoints]

# In[31]:


f_data_train=data_train.drop(['Id','groupId','matchId','killPoints','matchDuration','maxPlace',
                 'numGroups','rankPoints','roadKills','teamKills'
                 ,'winPoints','killStreaks','longestKill','killPoints'],axis=1)


# In[32]:


f_data_train.head()


# In[ ]:





# <a id='prepro'></a>
# # Data Preprocessing
# 
# **let's make a pipelines to automate our work**

# In[33]:


f_data_train.info()


# In[93]:


X= f_data_train.drop('winPlacePerc',axis=1)
X.head()


# In[109]:


y= data_train[['winPlacePerc']]
y.head()


# In[94]:


num_data = X.drop('matchType',axis=1)


# In[95]:


num_data.shape


# In[96]:


cat_data = f_data_train['matchType']


# In[97]:


cat_data = pd.DataFrame(cat_data)


# ## standardize numerical data

# In[98]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('std_scaler', StandardScaler())])

data_num_tr = num_pipeline.fit_transform(num_data)


# In[99]:


data_num_tr= pd.DataFrame(data_num_tr,columns=num_data.columns)
data_num_tr.head()


# ### Encod Categorical feature 

# In[100]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder(sparse=False)
data_cat_1hot = cat_encoder.fit_transform(cat_data)
data_cat_1hot


# In[101]:


cat_encoder.categories_


# Now let's build a pipeline for preprocessing all attributes:

# In[102]:


from sklearn.compose import ColumnTransformer

num_attribs = list(num_data)
cat_attribs = ["matchType"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

data_prepared = full_pipeline.fit_transform(f_data_train)


# In[103]:


data_prepared=pd.DataFrame(data_prepared)


# In[107]:


data_prepared.drop(17,axis=1,inplace=True)


# ### What we did is we made pipelines to make the work easier and turned the numerical data and categorical data to be ready for the model

# <a id='Train'></a>
# 
# # Select and Train the Model

# In[112]:


X = data_prepared.copy()


# In[115]:


y.head()


# In[117]:


import xgboost as xgb

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)

D_train = xgb.DMatrix(X_train, label=Y_train)
D_test = xgb.DMatrix(X_test, label=Y_test)


# In[118]:


param = {
    'eta': 0.15, 
    'max_depth': 5,  
    'num_class': 2} 

steps = 20  # The number of training iterations
model = xgb.train(param, D_train, steps)


# In[119]:


from sklearn.metrics import mean_squared_error

preds = model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])

print("MSE = {}".format(mean_squared_error(Y_test, best_preds)))


# # Apply on  Test data

# In[121]:


test_data = pd.read_csv('test_V2.csv')


# In[122]:


test_data.head()


# In[124]:


list(f_data_train.columns)


# In[148]:


test_set = test_data[['assists','boosts','damageDealt','DBNOs','headshotKills',
 'heals','killPlace','kills','matchType','revives','rideDistance',
 'swimDistance','vehicleDestroys','walkDistance','weaponsAcquired']]


# In[167]:


test_set.matchType.replace(['squad-fpp','squad','normal-squad-fpp','normal-squad'],'squad',inplace=True)
test_set.matchType.replace(['duo-fpp','normal-duo-fpp','normal-duo'],'duo',inplace=True)
test_set.matchType.replace(['solo-fpp','normal-solo-fpp','normal-solo'],'solo',inplace=True)
test_set.matchType.replace(['crashfpp','flaretpp','flarefpp','crashtpp'],'others',inplace=True)


# In[168]:


test_set.head()


# In[173]:


test_model = full_pipeline.fit_transform(test_set)


# In[171]:


test_model


# In[174]:


test_model = pd.DataFrame(test_model)


# In[175]:


test_model.head()


# In[176]:


test_model.drop(17,axis=1,inplace=True)


# # Training The models

# In[184]:


# X is dependant features
# y independant label
# test model for prediction


# In[189]:


X.head()


# In[188]:


y.head()


# In[190]:


test_model.head()


# In[192]:


X.to_csv('independant_feture_train.csv')
y.to_csv('dependant_feture_train.csv')
test_model.to_csv('test_model.csv')


# In[2]:


X = pd.read_csv('independant_feture_train.csv',)
y = pd.read_csv('dependant_feture_train.csv')
test_model = pd.read_csv('test_model.csv')


# In[10]:


X.drop('Unnamed: 0',axis=1,inplace=True)
y.drop('Unnamed: 0',axis=1,inplace=True)
test_model.drop('Unnamed: 0',axis=1,inplace=True)


# ## Random Forest

# In[22]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(X_train, y_train)


# In[26]:


from sklearn.metrics import mean_absolute_error
forest_predictions = forest_reg.predict(X_test)
forest_mae = mean_absolute_error(y_test, forest_predictions)
forest_mae


# In[ ]:





# # xgboost

# In[66]:


'''
for tuning parameters
parameters_for_testing = {
    'colsample_bytree':[0.4,0.6,0.8],
    'gamma':[0,0.03,0.1,0.3],
    'min_child_weight':[1.5,6,10],
    'learning_rate':[0.1,0.07],
    'max_depth':[3,5],
    'n_estimators':[10000],
    'reg_alpha':[1e-5, 1e-2,  0.75],
    'reg_lambda':[1e-5, 1e-2, 0.45],
    'subsample':[0.6,0.95]  
}

                   
xgb_model = xgboost.XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,
     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)

gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, n_jobs=6,iid=False, verbose=10,scoring='neg_mean_squared_error')
gsearch1.fit(train_x,train_y)
print (gsearch1.grid_scores_)
print('best params')
print (gsearch1.best_params_)
print('best score')
print (gsearch1.best_score_)
'''


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)


# In[63]:


import xgboost 

xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=1000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)
xgb_model.fit(X_train,y_train)


# In[64]:


xgb_pred = xgb_model.predict(X_test)


# In[65]:


mean_absolute_error(y_test, xgb_pred)


# In[71]:


test_pred = xgb_model.predict(test_model)


# In[68]:


submit = pd.read_csv('sample_submission_V2.csv')


# In[72]:


submit.winPlacePerc = test_pred


# In[75]:


submit.to_csv('xgboost_prediction.csv')


# # SGD

# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,)


# In[45]:


from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=10000, tol=1e-3, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X_train, y_train)


# In[46]:


sgd_pred=sgd_reg.predict(X_test)


# In[47]:


sqd_mae = mean_absolute_error(y_test,sgd_pred)


# In[48]:


sqd_mae


# # Linear Regression

# In[49]:


from sklearn.linear_model import LinearRegression


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# In[51]:


lm = LinearRegression()
lm.fit(X_train,y_train)


# In[52]:


lm_pred = lm.predict(X_test)


# In[53]:


mean_absolute_error(y_test,lm_pred)


# In[ ]:





# # so let's fine tune our model to se the best parameter

# In[ ]:




