#SVM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm,trange
import random
from sklearn import svm
from sklearn.preprocessing import StandardScaler


def scatter_loss_plot():
    plt.subplot(1,2,1)
    plt.ylim(-1,1)
    plt.xlim(-1,1)
    plt.plot(y_test[colnames[0]],y_test_pred[0],'.')

    
    plt.subplot(1,2,2)
    plt.ylim(-1,1)
    plt.xlim(-1,1)
    plt.plot(y_train[colnames[0]],y_train_pred[0],'.')
  
def rmse(obs,pre):
    return np.sqrt(mean_squared_error(obs, pre))
    
def caculate_cor():
    global r_test,r_train,y_test_pred,y_train_pred,rmse_test,rmse_train
    y_test_pred=pd.DataFrame(model.predict(x_test).reshape(y_test.shape),index=test_index)
    r_test=np.corrcoef(y_test_pred[0],y_test[colnames[0]])
    
    y_train_pred=pd.DataFrame(model.predict(x_train).reshape(y_train.shape),index=train_index)
    r_train=np.corrcoef(y_train_pred[0],y_train[colnames[0]])
    
    rmse_test=rmse(y_test[colnames[0]],y_test_pred[0])
    rmse_train=rmse(y_train[colnames[0]],y_train_pred[0])
xl = pd.ExcelFile('Standard_Genome.xlsx')
namelist=xl.sheet_names[0]

#predf=pd.read_excel('experiment materials.xlsx',sheet_name=1).iloc[0:5,0:31]

writer1=pd.ExcelWriter('svm-op_no_Genome.xlsx')
writer2=pd.ExcelWriter('svm-cor_no_Genome.xlsx')
#writer4=pd.ExcelWriter('all-val.xlsx')
#writer5=pd.ExcelWriter('all-pre.xlsx')
#writer6=pd.ExcelWriter('ss-index-all.xlsx')

frame=pd.read_excel('Standard_Genome.xlsx',sheet_name=0)
random.seed(1)
val=random.sample(range(0,len(frame)),5)
model_index=list(frame.index)
for j in val:
        model_index.remove(j)
    #x_data=frame[globals()['colindex'+str(i)]]
valdata=frame.loc[val,:]

val_x=valdata.iloc[:,0:18]
val_y=valdata.iloc[:,68:]
frame=frame.loc[model_index,:]
stdsc=StandardScaler()
x_data=frame.iloc[:,0:18]
x_data=pd.DataFrame(stdsc.fit_transform(x_data))
y_data=frame.iloc[:,68:]
x_names=x_data.columns.values.tolist()
colnames=y_data.columns.values.tolist()

    
ss=ShuffleSplit(n_splits=10, test_size=0.1,random_state=0)
    #stdsc=StandardScaler()
prelist=[]
vallist=[]
corlist_train=[]
corlist_test=[]
rmsel_train=[]
rmsel_test=[]
o=[]
model=svm.SVR(kernel='rbf')
with tqdm(total=10) as pbar:
        for train_index , test_index in ss.split(x_data,y_data):
            x_train=x_data.loc[train_index,:]
            x_train.columns=x_names
            y_train=y_data.iloc[train_index,:]        
            x_test=x_data.loc[test_index,:]
            x_test.columns=x_names        
            y_test=y_data.iloc[test_index,:]            
            model.fit(x_train,y_train)
            
            val_one=model.predict(val_x)
            vallist.append(val_one.T)
            #pre_one=model.predict(predf)
            #prelist.append(pre_one.T)
            
            caculate_cor()
            corlist_train.append(r_train[1,0])
            corlist_test.append(r_test[1,0])
            rmsel_train.append(rmse_train)
            rmsel_test.append(rmse_test)
            scatter_loss_plot()
            o.append(y_train[colnames[0]])
            o.append(y_train_pred[0])
            o.append(y_test[colnames[0]])
            o.append(y_test_pred[0])
            pbar.update()           

plt.show()        
cordf=pd.DataFrame({'train':corlist_train,'test':corlist_test,
                        'rmse_train':rmsel_train,'rmse_test':rmsel_test})
obs_pre_df=pd.DataFrame([y_data[colnames[0]],o[1],o[5],o[9],o[13],o[17],o[21],o[25],o[29],o[33],o[37],
                            o[3],o[7],o[11],o[15],o[19],o[23],o[27],o[31],o[35],o[39]]).T
obs_pre_df.columns=(colnames[0],'train1','train2','train3','train4','train5',
                        'train6','train7','train8','train9','train10',
                        'test1','test2','test3','test4','test5',
                        'test6','test7','test8','test9','test10')
    
presult=pd.DataFrame(prelist,columns=['T','C','S','M','L']).T
vresult=pd.DataFrame(vallist,columns=val).T
vresult['observe']=val_y
   
obs_pre_df.to_excel(writer1,sheet_name=colnames[0])
cordf.to_excel(writer2,sheet_name=colnames[0])
    #presult.to_excel(writer4,sheet_name=colnames[0])
    #vresult.to_excel(writer5,sheet_name=colnames[0])
       
writer1.save()
writer2.save()
#writer3.save()
#writer4.save()
#writer5.save()
