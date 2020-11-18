import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split
from DLtools.Data import instant_data,intersection,del_less_col,station_sel
from DLtools.feature_sel import call_mar

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

loading = instant_data()
# df,mode = loading.hourly_instant(),'hour'
df,mode = loading.daily_instant(),'day'

if mode =='hour': n_past,n_future = 24*7,24
elif mode =='day': n_past,n_future = 30,14
else: n_future=None; print('incorrect input')

st = 'CPY012'
target,start_p,stop_p,host_path=station_sel(st,mode)
save_path =host_path+'/SVR/'

########### DATA PREPROCES #####################################
data = df[start_p:stop_p]
data = del_less_col(data,ratio=.80).interpolate(limit=3000000000,limit_direction='both').astype('float32')
data['Day'] = data.index.dayofyear #add day
data_mar = call_mar(data,target,mode,sel_t=n_future)

X = data_mar.drop(columns=[target])
Y = data_mar[target]


X = data_mar.drop(columns=[target]).values
Y = data_mar[target].values

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_y.fit_transform(Y.reshape(-1,1))

########## EXP SETUP ###################################

# kernals = ['Polynomial', 'RBF','Linear']#A function which returns the corresponding SVC model
# def Yhat_series(yhat):
#     return yhat #pd.Series(yhat,index=test.index)
    
# def getClassifier(ktype):
#     if ktype == 0:
#         # Polynomial kernal
#         return svm.SVR(kernel='poly', degree=8, gamma="auto")
#     elif ktype == 1:
#         # Radial Basis Function kernal
#         return svm.SVR(kernel='rbf', gamma="auto")
#     elif ktype == 2:
#         # Linear kernal
#         return svm.SVR(kernel='linear', gamma="auto")


trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3, shuffle=False)
param_grid = {  'C': [1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10,10**2,10**3,10**4,10**5,10**6], 
                'gamma': [1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10,10**2,10**3,10**4,10**5,10**6],
                'kernel': ['rbf', 'linear']}
grid = GridSearchCV(svm.SVR(),param_grid,refit=True,verbose=2)
grid.fit(trainX,trainY)
print('*'*50)
print(grid.best_estimator_)
print('*'*50)
grid_predictions = grid.predict(testX)

plt.figure(figsize=(15,5))
plt.plot(sc_y.inverse_transform(testY),label='Y')
plt.plot(sc_y.inverse_transform(grid_predictions),label='Yhat')
plt.legend()
plt.savefig(save_path+'best_fig.png')