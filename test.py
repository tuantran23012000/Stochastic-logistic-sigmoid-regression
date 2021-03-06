from model import S_L_R
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns
parser = argparse.ArgumentParser()
parser.add_argument("--state",type=str,default=False,required=False)
parser.add_argument("--path_data", type = str, default = False,required=False)
parser.add_argument("--n_clus", type = int, default = False,required=False)
parser.add_argument("--alpha", type = float, default = False,required=False)
parser.add_argument("--option", type = str, default = False,required=False)
parser.add_argument("--threshold", type = float, default = False,required=False)
parser.add_argument("--min_iters", type = int, default = False,required=False)
parser.add_argument("--max_iters",type=int,default=False,required=False)
parser.add_argument("--max_loop",type=int,default=False,required=False)
args = parser.parse_args()

path=os.getcwd()+args.path_data
print("Data path: ",path)
df = pd.read_csv(path)
df.head()
n_v=len(df.columns[:-1])+1 # nunber of variables
df.dropna(inplace=True)
df.shape
X = df[df.columns[:-1]]
y = df[df.columns[-1]]

# Split the X, y data into training/validation/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 ,random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0) # 0.25 x 0.8 = 0.2
# #Standard data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
X_val = sc_X.transform(X_val)
X_train = pd.DataFrame(data=X_train, columns=X.columns)
X_test = pd.DataFrame(data=X_test, columns=X.columns)
X_val = pd.DataFrame(data=X_val, columns=X.columns)

def run_train(X_train,X_test,X_val,y_train,y_test,y_val,n_c,n_v,alpha,option,max_loop,threshold=None):

    slr=S_L_R()
    weight_all,list_k,acc_s,f1_s,p_s,r_s,g_w,epsi,mA,ret=slr.train(X_train,X_val,X_val,y_train,y_test,y_val,n_c,n_v,alpha,option,max_loop,threshold)
    if ret:
        index= np.argmax(acc_s)
        g_w=np.reshape(np.array(g_w),(len(g_w),1))
        w_x=[]
        for i in range(0,n_c):
            w_x.append(np.dot(mA[i,0:n_v],np.array(weight_all[index])).tolist())
        c=w_x-epsi[index]
        acc_g = slr.accuracy(slr.predict(weight_all[index],X_val),y_val) 
        f1_g = slr.f1_score(slr.predict(g_w,X_val),y_val)
        p_g = slr.precision_score(slr.predict(g_w,X_val),y_val)
        r_g = slr.recall_score(slr.predict(g_w,X_val),y_val)
        print("beta : ",list_k[index])
        print("accuracy of SLR on testset: ",slr.accuracy(slr.predict(weight_all[index],X_test),y_test))
        print("accuracy of SLR on valset: ",slr.accuracy(slr.predict(weight_all[index],X_val),y_val))
        print("accuracy of GLR on valset: ",slr.accuracy(slr.predict(g_w,X_val),y_val))
        print("accuracy of GLR on testset: ",slr.accuracy(slr.predict(g_w,X_test),y_test))
        print("f1_score of SLR on testset: ",slr.f1_score(slr.predict(weight_all[index],X_test),y_test))
        print("f1_score of SLR on valset: ",slr.f1_score(slr.predict(weight_all[index],X_val),y_val))
        print("f1_score of GLR on valset: ",slr.f1_score(slr.predict(g_w,X_val),y_val))
        print("f1_score of GLR on testset: ",slr.f1_score(slr.predict(g_w,X_test),y_test))
        print("precision of SLR on testset: ",slr.precision_score(slr.predict(weight_all[index],X_test),y_test))
        print("precision of SLR on valset: ",slr.precision_score(slr.predict(weight_all[index],X_val),y_val))
        print("precision of GLR on valset: ",slr.precision_score(slr.predict(g_w,X_val),y_val))
        print("precision of GLR on testset: ",slr.precision_score(slr.predict(g_w,X_test),y_test))
        print("recall of SLR on testset: ",slr.recall_score(slr.predict(weight_all[index],X_test),y_test))
        print("recall of SLR on valset: ",slr.recall_score(slr.predict(weight_all[index],X_val),y_val))
        print("recall of GLR on valset: ",slr.recall_score(slr.predict(g_w,X_val),y_val))
        print("recall of GLR on testset: ",slr.recall_score(slr.predict(g_w,X_test),y_test))
        plt.figure(figsize=(10,5))
        plt.rcParams.update({'font.size': 16})
        plt.plot(list_k, acc_g*np.ones(len(acc_s)), label = "ACC_GLR",linewidth=4,ls="--")
        plt.plot(list_k, acc_s, label = "ACC_SLR",linewidth=4)
        plt.xlabel("Beta")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()
        plt.show()
        plt.figure(figsize=(10,5))
        plt.rcParams.update({'font.size': 16})
        plt.plot(list_k, f1_g*np.ones(len(f1_s)), label = "F1Score_GLR",linewidth=4,ls="--")
        plt.plot(list_k, f1_s, label = "F1Score_SLR",linewidth=4)
        plt.xlabel("Beta")
        plt.ylabel("F1-score")
        plt.legend()
        plt.grid()
        plt.show()
        plt.figure(figsize=(10,5))
        plt.rcParams.update({'font.size': 16})
        plt.plot(list_k, p_g*np.ones(len(p_s)), label = "Pre_GLR",linewidth=4,ls="--")
        plt.plot(list_k, p_s, label = "Pre_SLR",linewidth=4)
        plt.xlabel("Beta")
        plt.ylabel("Precision-score")
        plt.legend()
        plt.grid()
        plt.show()
        plt.figure(figsize=(10,5))
        plt.rcParams.update({'font.size': 16})
        plt.plot(list_k, r_g*np.ones(len(r_s)), label = "Re_GLR",linewidth=4,ls="--")
        plt.plot(list_k, r_s, label = "Re_SLR",linewidth=4)
        plt.xlabel("Beta")
        plt.ylabel("Recall-score")
        plt.legend()
        plt.grid()
        plt.show()


def run_eval(X_train,X_test,X_val,y_train,y_test,y_val,n_v,alpha,min_iters,max_iters):

    slr=S_L_R()
    acc_s,f1_s,p_s,r_s,acc_g,f1_g,p_g,r_g,num,time_run=slr.eval(X_train,X_test,X_val,y_train,y_test,y_val,n_v,alpha,"Kmean",min_iters,max_iters)
    print(num)
    X_val=X_val.drop('bias',axis=1)
    acc_s_q,f1_s_q,p_s_q,r_s_q,_,_,_,_,_,time_run_q=slr.eval(X_train,X_test,X_val,y_train,y_test,y_val,n_v,alpha,"Quantile",min_iters,num)
    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 16})
    plt.plot(num, acc_g*np.ones(len(acc_s)), label = "ACC_GLR",linewidth=4)
    plt.plot(num, acc_s, label = "ACC_SLR_kmean",linewidth=4,ls="--")
    plt.plot(num, acc_s_q, label = "ACC_SLR_quantile",linewidth=4,ls="-.")
    plt.xlabel("num_cluters(level_quantile)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()
    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 16})
    plt.plot(num, f1_g*np.ones(len(f1_s)), label = "F1Score_GLR",linewidth=4)
    plt.plot(num, f1_s, label = "F1Score_SLR_kmean",linewidth=4,ls="--")
    plt.plot(num, f1_s_q, label = "F1Score_SLR_quantile",linewidth=4,ls="-.")
    plt.xlabel("num_cluters(level_quantile)")
    plt.ylabel("F1-score")
    plt.legend()
    plt.grid()
    plt.show()
    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 16})
    plt.plot(num, p_g*np.ones(len(p_s)), label = "Pre_GLR",linewidth=4)
    plt.plot(num, p_s, label = "Pre_SLR_kmean",linewidth=4,ls="--")
    plt.plot(num, p_s_q, label = "Pre_SLR_quantile",linewidth=4,ls="-.")
    plt.xlabel("num_cluters(level_quantile)")
    plt.ylabel("Precision-score")
    plt.legend()
    plt.grid()
    plt.show()
    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 16})
    plt.plot(num, r_g*np.ones(len(r_s)), label = "Re_GLR",linewidth=4)
    plt.plot(num, r_s, label = "Re_SLR_kmean",linewidth=4,ls="--")
    plt.plot(num, r_s_q, label = "Re_SLR_quantile",linewidth=4,ls="-.")
    plt.xlabel("num_cluters(level_quantile)")
    plt.ylabel("Recall-score")
    plt.legend()
    plt.grid()
    plt.show()
    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 16})
    plt.plot(num, time_run, label = "Time_training_kmean",linewidth=4,ls="--")
    plt.plot(num, time_run_q, label = "Time_training_quantile",linewidth=4,ls="-.")
    plt.xlabel("num_cluters(level_quantile)")
    plt.ylabel("Time training")
    plt.legend()
    plt.grid()
    plt.show()

if __name__=='__main__':

    if args.state == "train":
        run_train(X_train,X_test,X_val,y_train,y_test,y_val,args.n_clus,n_v,args.alpha,args.option,args.max_loop,args.threshold)
    elif args.state == "eval":
        run_eval(X_train,X_test,X_val,y_train,y_test,y_val,n_v,args.alpha,args.min_iters,args.max_iters)
    
