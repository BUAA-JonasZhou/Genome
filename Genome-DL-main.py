from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import lime
import lime.lime_tabular
import shap
import lime_stability 
from lime_stability.stability import LimeTabularExplainerOvr
from tensorflow.keras import layers, models
from keras import losses 
from tensorflow.keras.models import clone_model
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm,trange
import random
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from itertools import combinations



def rmse(obs,pre):
        return np.sqrt(mean_squared_error(obs, pre))
def caculate_cor():
        global r_test,r_train,y_test_pred,y_train_pred,rmse_test,rmse_train
        y_test_pred=pd.DataFrame(model.predict(x_test),index=test_index)
        r_test=np.corrcoef(y_test_pred[0],y_test[colnames[0]])
        y_train_pred=pd.DataFrame(model.predict(x_train),index=train_index)
        r_train=np.corrcoef(y_train_pred[0],y_train[colnames[0]])
        rmse_test=rmse(y_test[colnames[0]],y_test_pred[0])
        rmse_train=rmse(y_train[colnames[0]],y_train_pred[0])

writer1=pd.ExcelWriter('dl-op_genome.xlsx')
writer2=pd.ExcelWriter('dl-cor_enome.xlsx')
writer3=pd.ExcelWriter('dl-evaluation_genome.xlsx')
writer4=pd.ExcelWriter('dl-explanation_genome.xlsx')
#writer5=pd.ExcelWriter('dl-val_genome_test.xlsx')
#writer6=pd.ExcelWriter('all-dl-shap_explanation_genome.xlsx')

prelist=[]
vallist=[]
corlist_train=[]
corlist_test=[]
rmsel_train=[]
rmsel_test=[]
o=[]
lossl=[]
mael=[]

explanation=[]
shapl=[]

frame=pd.read_excel('Standard_Genome.xlsx')
print(frame.isnull().any())
random.seed(3)

#import data
xl = pd.ExcelFile('Standard_Genome.xlsx')
namelist=xl.sheet_names[0]
val=random.sample(range(0,len(frame)),5)
model_index=list(frame.index)

for j in val:
        model_index.remove(j)
valdata=frame.loc[val,:]
val_x=valdata.iloc[:,0:68]
val_y=valdata.iloc[:,68:]
frame=frame.loc[model_index,:]
x_data=frame.iloc[:,0:68]
x_data.index=range(len(x_data))
y_data=frame.iloc[:,68:]
y_data.index=range(len(y_data))
x_names=x_data.columns.values.tolist()
colnames=y_data.columns.values.tolist()

input_dim=68
output_dim=1

#build DL models
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(input_dim,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(output_dim)  
])

#model compiling
model.compile(optimizer='adam', 
              loss='mse',      
              #loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['mse']) 

model.summary()

#model training and validation
ss=ShuffleSplit(n_splits=10, test_size=0.3,random_state=0)


for train_index , test_index in ss.split(x_data,y_data):
        x_train=x_data.iloc[train_index,:]
        x_train.columns=x_names    
        y_train=y_data.iloc[train_index,:]    
        x_test=x_data.iloc[test_index,:]
        x_test.columns=x_names
        y_test=y_data.iloc[test_index,:]
        
        loss, mae = model.evaluate(x_train, y_train)
        model.fit(x_train,np.array(y_train).ravel(),epochs=10, batch_size=32,validation_split=0.3)
        val_one=model.predict(val_x)
        vallist.append(val_one.T)
        
        #LIME explainer
        #explainer=lime.lime_tabular.LimeTabularExplainer(np.array(x_train),feature_names=x_train.columns,mode='regression') 
        explainer = LimeTabularExplainerOvr(np.array(x_train), feature_names=x_train.columns, verbose=True, mode='regression')
        
        sample_idx=0
        sample=x_test.iloc[sample_idx]
               
        exp=explainer.explain_instance(sample,model.predict,num_features=20)
        
        weights=exp.as_list()
        features=[x[0] for x in weights]
        weights=[x[1] for x in weights]
        
        csi,vsi=explainer.check_stability(n_calls=20, data_row=x_test.iloc[sample_idx],predict_fn=model, index_verbose=False )
        
        caculate_cor()
        
        corlist_train.append(r_train[1,0])
        corlist_test.append(r_test[1,0])
        rmsel_train.append(rmse_train)
        rmsel_test.append(rmse_test)
        lossl.append(loss)
        mael.append(mae)
        
        
        o.append(y_train[colnames[0]])
        o.append(y_train_pred[0])
        o.append(y_test[colnames[0]])
        o.append(y_test_pred[0])
        
        explanation.append({'Fold':len(explanation)+1, 'csi':csi, 'vsi':vsi,'Feature':features,'Weights':weights,})
       


#SHAP explainer
shap_explainer=shap.Explainer(model, x_train)


# model prediction
predictions = model.predict(x_train[:5])

print("Predictions:", predictions)
  
cordf=pd.DataFrame({'train':corlist_train,'test':corlist_test,
                        'rmse_train':rmsel_train,'rmse_test':rmsel_test})
obs_pre_df=pd.DataFrame([y_data[colnames[0]],o[1],o[5],o[9],o[13],o[17],o[21],o[25],o[29],o[33],o[37],
                            o[3],o[7],o[11],o[15],o[19],o[23],o[27],o[31],o[35],o[39]]).T
model_evaluation=pd.DataFrame({'loss':lossl,'mae':mael})

obs_pre_df.columns=(colnames[0],'train1','train2','train3','train4','train5',
                        'train6','train7','train8','train9','train10',
                        'test1','test2','test3','test4','test5',
                        'test6','test7','test8','test9','test10')

#vresult=pd.DataFrame(vallist,columns=val).T
    
#print(np.corrcoef(np.array(np.mean(vresult.T)).ravel(),
#                          np.array(val_y).ravel())[0,1]) 

#vresult['predict']=np.array(np.mean(vresult.T)).ravel()
#vresult['observe']=val_y
#vresult['error']=vresult['predict']-vresult['observe']

explanation=pd.DataFrame(explanation)

#shap_df=pd.DataFrame(shap_values, columns=shap_explainer.feature_names)

#write result into excel files
obs_pre_df.to_excel(writer1,sheet_name=colnames[0]) #training and validation result
cordf.to_excel(writer2,sheet_name=colnames[0])#R2 and RMSE
model_evaluation.to_excel(writer3,sheet_name=colnames[0])#loss and mae
explanation.to_excel(writer4,sheet_name=colnames[0])
#vresult.to_excel(writer5,sheet_name=colnames[0])
#shap_df.to_excel(writer6, sheet_name=colnames[0]) 

#dave data
writer1.save()
writer2.save()
writer3.save()
writer4.save()
#writer5.save()
#writer6.save()


"""

"""
xl = pd.ExcelFile('Standard_Genome.xlsx')
namelist=xl.sheet_names[0:68]
writer6=pd.ExcelWriter('ss-index-all_genome_test.xlsx')
l_train_list=[]
l_test_list=[]
frame=pd.read_excel('Standard_Genome.xlsx')#

corresult=pd.read_excel('external-dl-cor_genome_test.xlsx',sheet_name=0)
max_index=np.argmax(corresult['train'],axis=0)  
    
random.seed(1)
val=random.sample(range(0,len(frame)),62)#
model_index=list(frame.index)
for j in val:
    model_index.remove(j)
    #x_data=frame[globals()['colindex'+str(i)]]
frame=frame.iloc[model_index,:] 
x_data=frame.iloc[:,0:68]
ss=ShuffleSplit(n_splits=10, test_size=0.1,random_state=0)
index_list=[]
for train_index in ss.split(x_data):
        index_list.append(train_index)
index_train_list=index_list[max_index][0]
index_df=pd.DataFrame(index_train_list)
index_df.to_excel(writer6,sheet_name=namelist[0])

l_train=len(index_list[0][0]) 
l_test=len(index_list[0][1])
l_train_list.append(l_train)
l_test_list.append(l_test) 

len_df=pd.DataFrame({'train':l_train_list,'test':l_test_list})
len_df.to_excel('ss_dl_len_genome_test.xlsx')

"""
#ss-length output
"""
""""""

l_train_list=[]
l_test_list=[]
frame=pd.read_excel('Standard_Genome.xlsx')

globals()['colindex'+str(0)]=[]
    
x_data=frame[globals()['colindex'+str(0)]]
y_data=frame.iloc[:,68:]
x_names=x_data.columns.values.tolist()
colnames=y_data.columns.values.tolist()

ss=ShuffleSplit(n_splits=10, test_size=0.1,random_state=0)

index_list=[]
for train_index in ss.split(x_data):
        index_list.append(train_index)
l_train=len(index_list[0][0]) 
l_test=len(index_list[0][1])
l_train_list.append(l_train)
l_test_list.append(l_test)


"""
#colindex output
"""
writer7=pd.ExcelWriter('colindex_dl-genome_test.xlsx')
colindex=pd.DataFrame(globals()['colindex'+str(0)])
colindex.to_excel(writer7,sheet_name=namelist[0])
writer7.save()
    
