from keras import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras import optimizers
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm,trange
import xlrd
import openpyxl

def write_result(t):#定义写入结果函数
    pd.DataFrame(train.history).to_excel(writer,startcol=5*t,sheet_name='loss&mae')
    pd.DataFrame(globals()['r_test_'+colnames[0]]).to_excel(writer,startcol=68*t,sheet_name='cor&score')
    pd.DataFrame(globals()['r_test_'+colnames[1]]).to_excel(writer,startcol=68*t+3,sheet_name='cor&score')
    pd.DataFrame(globals()['r_train_'+colnames[0]]).to_excel(writer,startcol=68*t+6,sheet_name='cor&score')
    pd.DataFrame(globals()['r_train_'+colnames[1]]).to_excel(writer,startcol=68*t+9,sheet_name='cor&score')
    pd.DataFrame(score).to_excel(writer,startcol=68*t,sheet_name='score')
    
    pred_obv=pd.DataFrame({'y_test_'+colnames[0]:y_test[colnames[0]].reset_index(drop=True),
                       'y_test_'+colnames[1]:y_test[colnames[1]].reset_index(drop=True),
                       #'y_pred_'+colnames[0]:y_pred[0],
                       #'y_pred_'+colnames[1]:y_pred[1],
                       'y_train_'+colnames[0]:y_train[colnames[0]].reset_index(drop=True),
                       'y_train_'+colnames[1]:y_train[colnames[1]].reset_index(drop=True),
                       'y_train_pred_'+colnames[0]:y_train_pred[0],
                       'y_train_pred_'+colnames[1]:y_train_pred[1]     })

    pred_obv.to_excel(writer,sheet_name='obv&pre',startcol=t*9)

def scatter_loss_plot():
    plt.subplot(2,3,1)
    plt.ylim(-1,1)
    plt.xlim(-1,1)
    plt.plot(y_test[colnames[0]],y_test_pred[0],'.')

    
    plt.subplot(2,3,2)
    plt.ylim(-1,1)
    plt.xlim(-1,1)
    plt.plot(y_train[colnames[0]],y_train_pred[0],'.')

    
    plt.subplot(2,3,4)
    #plt.ylim()
    plt.plot(train.history['loss'],'-')    
    plt.subplot(2,3,5)
    #plt.ylim()
    #plt.plot(train.history['mean_absolute_error'],'-')    
    plt.subplot(2,3,6)
    plt.plot(train.history['val_loss'],'-')    
    
def Network_train(opt,Setlr,dlcs,sjsl,nepochs):#神经网络训练，自定义：优化器选择opt，学习率setlr，动量dlcs，学习率衰减值sjsl
    global train,score#全局变量
    Adam=optimizers.Adam(lr=Setlr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=sjsl, amsgrad=True)#定义Adam优化器
    sgd=optimizers.SGD(lr=Setlr, momentum=dlcs, decay=sjsl, nesterov=False)#定义SGD（随机梯度下降）优化器
    Adagrad=optimizers.Adagrad(lr=Setlr, epsilon=1e-06)#定义Adagrad优化器
    model.compile(optimizer=opt,loss='mean_squared_error', metrics=['mae'])#配置训练模型：定义优化器，损失函数和模型评估标准
    #train=model.fit(x_data.iloc[train_index,:],y_data.iloc[train_index,:],
                    #validation_data=(x_val,y_val),epochs=nepochs,batch_size=16)
    train=model.fit(x_train,y_train,validation_split=0.11,epochs=nepochs,batch_size=16,verbose=1)#以给定数量的轮次训练模型，训练模型迭代轮次nepochs
    score=model.evaluate(x_test,y_test,batch_size=16)#在测试模式下返回模型的误差值和评估标准值，batch_size：每次评估的样本数

def Set_network(n_hide,n_input):
    global model
    model=Sequential()#创建一个sequential模型
    model.add(Dense(input_dim=n_input,units=n_hide,kernel_initializer='normal',activation='relu'))#添加全连接层
    #input_dim：输入尺寸，units：输出尺寸，kernel_initializer normal：正态分布随机初始化激活，函数relu：整流线性单元，返回逐元素的max（x，0）
    model.add(Dropout(0.2))#Dropout 包括在训练中每次更新时， 将输入单元的按比率随机设置为 0， 这有助于防止过拟合。此处rate=0.2 
    #model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))
    #model.add(Dense(input_dim=n_hide,units=1,kernel_initializer='normal'))
    #model.add(LeakyReLU(alpha=1))

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

#################################################################################################3
frame=pd.read_excel('Standard_Genome.xlsx',sheet_name=0)

x_data=frame.iloc[:,0:18]
y_data=frame.iloc[:,68:]

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2,random_state=1234)
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.25,random_state=1234)
colnames=y_data.columns.values.tolist()

#将predict表中的数据写入predf中
file1='Standard_Genome.xlsx'
wb1=xlrd.open_workbook(filename=file1)
ws1=wb1.sheet_by_name('Sheet1')#定义ws1为wb1的sheet1
predictdata=[]
for i in range(ws1.nrows):
    col=[]
    for j in range(ws1.ncols):
        col.append(ws1.cell(i,j).value)
    predictdata.append(col)#添加预测数据
predf=pd.DataFrame(predictdata[1:],columns=predictdata[0],dtype='float64')

ss=ShuffleSplit(n_splits=10, test_size=0.1,random_state=0)
kf=KFold(n_splits=10,shuffle=False)
prelist=[]
corlist_train=[]
corlist_test=[]
rmsel_train=[]
rmsel_test=[]
o=[]
with tqdm(total=10) as pbar:
    for train_index , test_index in ss.split(x_data,y_data):
        #global x_train,y_train,x_test,y_test
        x_train=x_data.iloc[train_index,:]
        y_train=y_data.iloc[train_index,:]
        x_test=x_data.iloc[test_index,:]
        y_test=y_data.iloc[test_index,:]
        Set_network(36,18)
        Network_train('sgd',0.1,0.9,0.0001,2000)
        #pre=model.predict(predf)
        #prelist.append(pre.T)
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
        pbar.update(1)
    
    
#presult=pd.DataFrame(np.array(prelist),columns=['T','C','S','M','L'])
cordf=pd.DataFrame({'tarin':corlist_train,'test':corlist_test,'rmse_train':rmsel_train,'rmse_test':rmsel_test})
obs_pre_df=pd.DataFrame([y_data[colnames[0]],o[1],o[5],o[9],o[13],o[17],o[21],o[25],o[29],o[33],o[37],
                        o[3],o[7],o[11],o[15],o[19],o[23],o[27],o[31],o[35],o[39]]).T
obs_pre_df.columns=(colnames[0],'train1','train2','train3','train4','train5',
                    'train6','train7','train8','train9','train10',
                    'test1','test2','test3','test4','test5',
                    'test6','test7','test8','test9','test10')

writer=pd.ExcelWriter('anti-relu-noGenome.xlsx')
#write_result(3)
#writer.save()
weight=model.get_weights()
model.save('anti-relu-model.h5')
model.save_weights("anti-relu-weights.h5")



writer1=pd.ExcelWriter('ann-op-no_Genome.xlsx')
writer2=pd.ExcelWriter('ann-cor-no_Genome.xlsx')

frame=pd.read_excel('Standard_Genome.xlsx',sheet_name=0)
x_data=frame.iloc[:,0:18]
x_data.index=range(len(x_data))
y_data=frame.iloc[:,68:]
y_data.index=range(len(y_data))
x_names=x_data.columns.values.tolist()
colnames=y_data.columns.values.tolist()


ss=ShuffleSplit(n_splits=10, test_size=0.1,random_state=0)
stdsc=StandardScaler()
x_data=pd.DataFrame(stdsc.fit_transform(x_data))
x_data.columns=x_names
    
prelist=[]
corlist_train=[]
corlist_test=[]
o=[]
    
with tqdm(total=10) as pbar:
        for train_index , test_index in ss.split(x_data,y_data):
            #global x_train,y_train,x_test,y_test
            x_train=x_data.iloc[train_index,:]
            #x_train=pd.DataFrame(stdsc.fit_transform(x_train))    
            y_train=y_data.iloc[train_index,:]    
            x_test=x_data.iloc[test_index,:]
            #x_test=pd.DataFrame(stdsc.fit_transform(x_test))
            #x_test.columns=x_names      
            y_test=y_data.iloc[test_index,:]
            Set_network(36,18)
            Network_train('Adam',0.1,0.9,0,200)
            caculate_cor()
            corlist_train.append(r_train[1,0])
            corlist_test.append(r_test[1,0])
            #scatter_loss_plot()
            o.append(y_train[colnames[0]])
            o.append(y_train_pred[0])
            o.append(y_test[colnames[0]])
            o.append(y_test_pred[0])
            pbar.update(1)

cordf=pd.DataFrame({'tarin':corlist_train,'test':corlist_test,'rmse_train':rmsel_train,'rmse_test':rmsel_test})
obs_pre_df=pd.DataFrame([y_data[colnames[0]],o[1],o[5],o[9],o[13],o[17],o[21],o[25],o[29],o[33],o[37],
                            o[3],o[7],o[11],o[15],o[19],o[23],o[27],o[31],o[35],o[39]]).T
obs_pre_df.columns=(colnames[0],'train1','train2','train3','train4','train5',
                        'train6','train7','train8','train9','train10',
                        'test1','test2','test3','test4','test5',
                        'test6','test7','test8','test9','test10')
    
obs_pre_df.to_excel(writer1,sheet_name=colnames[0])
cordf.to_excel(writer2,sheet_name=colnames[0])

writer1.save()
writer2.save()