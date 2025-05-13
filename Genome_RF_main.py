from enum import auto
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm,trange
import random
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from itertools import combinations
import openpyxl


def rmse(obs,pre):
        return np.sqrt(mean_squared_error(obs, pre))
def caculate_cor():
        global r_test,r_train,y_test_pred,y_train_pred,rmse_test,rmse_train
        y_test_pred=pd.DataFrame(model.predict(x_test).reshape(y_test.shape),index=test_index)#测试集预测结果
        r_test=np.corrcoef(y_test_pred[0],y_test[colnames[0]])#计算测试集结果相关性（相关系数）
        y_train_pred=pd.DataFrame(model.predict(x_train).reshape(y_train.shape),index=train_index)#训练集预测结果
        r_train=np.corrcoef(y_train_pred[0],y_train[colnames[0]])#计算训练集结果相关性（相关系数）
        rmse_test=rmse(y_test[colnames[0]],y_test_pred[0])#计算测试集均方根误差
        rmse_train=rmse(y_train[colnames[0]],y_train_pred[0])#计算训练集均方根误差
#导入数据
xl = pd.ExcelFile('Standard_Genome.xlsx')
namelist=xl.sheet_names[0]

#建立随机森林回归模型
model=RandomForestRegressor(n_estimators=600, 
                            #max_depth=7, 
                            max_features=9, 
                            oob_score=True,random_state=0,
                            #min_samples_leaf=10 #,min_samples_leaf=2
                                   )
#predf=pd.read_excel('experiment materials.xlsx',sheet_name=1).iloc[0:5,0:31]

writer1=pd.ExcelWriter('rf-hpo-op_Genome6-4.xlsx')
writer2=pd.ExcelWriter('rf-hpo-cor_Genome6-4.xlsx')
writer3=pd.ExcelWriter('rf-hpo-imp_Genome6-4.xlsx')
writer4=pd.ExcelWriter('rf-hpo-pre_Genome6-4.xlsx')
writer5=pd.ExcelWriter('rf-hpo-val_Genome6-4.xlsx')
#writer6=pd.ExcelWriter('ss-index-all.xlsx')
frame=pd.read_excel('Standard_Genome.xlsx')
test=pd.read_excel('validation_strugeon.xlsx')
model_index=list(frame.index)#将frame转变为列表
valdata=pd.DataFrame(test)
random.seed(3)#定义随机数种子

#val=random.sample(range(0,len(frame)),5)#从数据中选出五行
model_index=list(frame.index)#将frame转变为列表

valdata=pd.DataFrame(test)
val_x=valdata.iloc[:,0:68]
val_y=valdata.iloc[:,68:]
frame=frame.loc[model_index,:]
x_data=frame.iloc[:,0:68]
    #x_data=frame[globals()['colindex'+str(i)]]
x_data.index=range(len(x_data))
y_data=frame.iloc[:,68:]
y_data.index=range(len(y_data))
x_names=x_data.columns.values.tolist()
colnames=y_data.columns.values.tolist()
    
ss=ShuffleSplit(n_splits=10, test_size=0.1,random_state=0)#定义交叉验证
    #stdsc=StandardScaler()
prelist=[]
vallist=[]
corlist_train=[]
corlist_test=[]
rmsel_train=[]
rmsel_test=[]
o=[]
imp=[]
model=RandomForestRegressor(n_estimators=600, 
                            #max_depth=7, 
                            max_features=9, 
                            oob_score=True,random_state=0,
                            #min_samples_leaf=10 #,min_samples_leaf=2
                            )
#不考虑max_depth和min_samples_leaf的模型结果反而更好

for train_index , test_index in ss.split(x_data,y_data):
        x_train=x_data.iloc[train_index,:]
        x_train.columns=x_names
        
        y_train=y_data.iloc[train_index,:]
        
        x_test=x_data.iloc[test_index,:]
        x_test.columns=x_names
    
        y_test=y_data.iloc[test_index,:]

        model.fit(x_train,np.array(y_train).ravel())
        val_one=model.predict(val_x)#基于val_x进行预测
        vallist.append(val_one.T)#将预测结果添加进vallist
        #pre_one=model.predict(predf)
        #prelist.append(pre_one.T)


        caculate_cor()#计算模型预测效果
        #将结果添加至对应的数据集中
        corlist_train.append(r_train[1,0])
        corlist_test.append(r_test[1,0])
        rmsel_train.append(rmse_train)
        rmsel_test.append(rmse_test)
       # scatter_loss_plot()
        o.append(y_train[colnames[0]])
        o.append(y_train_pred[0])
        o.append(y_test[colnames[0]])
        o.append(y_test_pred[0])
        #pbar.update()           
        imp.append(model.feature_importances_)#计算特征重要性（Gini importance)
        


    #plt.show()        
cordf=pd.DataFrame({'train':corlist_train,'test':corlist_test,
                        'rmse_train':rmsel_train,'rmse_test':rmsel_test})#定义表格的列名
obs_pre_df=pd.DataFrame([y_data[colnames[0]],o[1],o[5],o[9],o[13],o[17],o[21],o[25],o[29],o[33],o[37],
                            o[3],o[7],o[11],o[15],o[19],o[23],o[27],o[31],o[35],o[39]]).T

obs_pre_df.columns=(colnames[0],'train1','train2','train3','train4','train5',
                        'train6','train7','train8','train9','train10',
                        'test1','test2','test3','test4','test5',
                        'test6','test7','test8','test9','test10')#第一列为RF原表中第32列数据，即实验结果
presult=pd.DataFrame(prelist,columns=['T','C','S','M','L']).T #没用，参见前面隐去的代码
#vresult=pd.DataFrame(vallist,columns=test_index).T#预测结果
print(vallist)    
print(val_y)
print(np.corrcoef(np.array(np.mean(vresult.T)).ravel(),np.array(val_y).ravel())[0,1]) #计算预测结果与实际结果的相关性

#vresult['predict']=np.array(np.mean(vresult.T)).ravel()#预测值
#vresult['observe']=val_y #真实值
#vresult['error']=vresult['predict']-vresult['observe']#两者之间的误差

    #imp_df=pd.DataFrame(imp,columns=globals()['colindex'+str(i)])
imp_df=pd.DataFrame(imp,columns=x_names)
    #将结果写进excel表格
obs_pre_df.to_excel(writer1,sheet_name=colnames[0]) #写入训练—验证结果
cordf.to_excel(writer2,sheet_name=colnames[0])#写入10次交叉验证后，训练-验证集的相关系数和rmse结果
imp_df.to_excel(writer3,sheet_name=colnames[0])#写入10次交叉验证后，每次每个特征的重要性
presult.to_excel(writer4,sheet_name=colnames[0])
#vresult.to_excel(writer5,sheet_name=colnames[0])#10次交叉验证+预测误差
    
writer1.save()
writer2.save()#writer2=pd.ExcelWriter('all-rf-cor1.xlsx')
writer3.save()
writer4.save()
#writer5._save()

#Succes


#ss-index output

xl = pd.ExcelFile('Standard_Genome.xlsx')
namelist=xl.sheet_names[0:68]
writer6=pd.ExcelWriter('ss-index-all_Genome_hpo.xlsx')
l_train_list=[]
l_test_list=[]
frame=pd.read_excel('Standard_Genome.xlsx')#导入原始数据

corresult=pd.read_excel('rf-hpo-cor_Genome6-4.xlsx',sheet_name=0)#导入相关系数和rmse结果
max_index=np.argmax(corresult['train'],axis=0)   #在train中寻找最大值
    
random.seed(1)
val=random.sample(range(0,len(frame)),5)#从原始数据中随机选出5行
model_index=list(frame.index)
for j in val:
    model_index.remove(j)
    #x_data=frame[globals()['colindex'+str(i)]]
frame=frame.iloc[model_index,:] #用选出的5行数据组成新表
x_data=frame.iloc[:,0:18]
ss=ShuffleSplit(n_splits=10, test_size=0.1,random_state=0)#交叉验证
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

writer6.save()

len_df=pd.DataFrame({'train':l_train_list,'test':l_test_list})
len_df.to_excel('ss_len_Genome.xlsx')


    #index_train_list=index_list[max_index][0]
    
    #index_df=pd.DataFrame(index_train_list)
    #index_df=index_df.stack()
    #index_df=index_df.unstack(0)

    #index_df.to_excel(writer3,sheet_name=colnames[0])
"""
#ss-length output
"""
""""""

l_train_list=[]
l_test_list=[]
frame=pd.read_excel('Standard_Genome.xlsx')

globals()['colindex'+str(0)]=[]
    
#corresult=pd.read_excel('sbs-rf-cor.xlsx',sheet_name=i-1)
#max_index=np.argmax(corresult['test'],axis=0)
    
x_data=frame[globals()['colindex'+str(0)]]
y_data=frame.iloc[:,68:]
x_names=x_data.columns.values.tolist()
colnames=y_data.columns.values.tolist()

ss=ShuffleSplit(n_splits=10, test_size=0.4,random_state=0)

index_list=[]
for train_index in ss.split(x_data):
        index_list.append(train_index)
l_train=len(index_list[0][0]) 
l_test=len(index_list[0][1])
l_train_list.append(l_train)
l_test_list.append(l_test)



    #index_train_list=index_list[max_index][0]
    
    #index_df=pd.DataFrame(index_train_list)
    #index_df=index_df.stack()
    #index_df=index_df.unstack(0)

    #index_df.to_excel(writer3,sheet_name=colnames[0])

"""
#colindex output
"""
writer5=pd.ExcelWriter('colindex-hpo_Genome.xlsx')
colindex=pd.DataFrame(globals()['colindex'+str(0)])
colindex.to_excel(writer5,sheet_name=namelist[0])
writer5.save()

#n_estimators=750: 0.9804647126241955
#n_estimators=700: 0.980419140546526
#n_estimators=650: 0.9805926211130133
#n_estimators=625: 0.9806900109942338
#n_estimators=600: 0.9807268402260877 (最佳)
#n_estimators=550: 0.9804152487943313
    