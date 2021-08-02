import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix , auc , classification_report,roc_curve,log_loss,roc_auc_score,f1_score,precision_score,recall_score
from scipy.special import erf
from sklearn.cluster import KMeans
from cvxopt import matrix , solvers , spdiag,mul , log, div ,exp
import scipy.linalg as la
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", np.ComplexWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from tqdm import tqdm
import time
from data_scoring import scoring_data

class S_L_R():

    def __init__(self):

        self.key_weight=[]
        self.pro=0
        self.weight=[]
        self.list_k=[]

    def laplace_inv(self,k):

        arr=[]
        Phi = lambda x: (erf(x/2**0.5))/2
        for i in range (-500,500,1):
            if np.abs(Phi(i/100.0)-(k-0.5)) < 0.01:
                arr.append(i)
        return sum(arr)/(100*len(arr))

    def model(self,meanA,varA,beta,m,b,y_mu,alpha): 

        k1 = 1.0/self.laplace_inv((beta+1)/2) 
        k2 = 1.0/self.laplace_inv((1-beta)/2) 
        check1=[] 
        check2=[]
        check3=[]
        h=[]
        n=m
        l=len(meanA)
        m_A=np.array(meanA)
        G=[matrix(m_A[0:int(l/2),:])]
        G+=[matrix((-1)*m_A[0:int(l/2),:])]

        meanA1 = k1*np.array(meanA[0:int(l/2),:])
        meanA2 = k2*np.array(meanA[int(l/2):l,:])
        meanA = np.concatenate((meanA1,meanA2))
        
        varA=np.array(varA)
        # Dùng list để lưu lại các ma trận hệ số sau khi đưa về dạng chính tắc của các ràng buộc căn 
        for i in range(0,l):
            eigvals,eigvecs=la.eig(np.reshape(varA[i,:],(n,n)))
            g=-np.multiply(eigvecs,np.sqrt(eigvals)).T
            check1.append(g)
        Z=np.zeros((1,n))
        # Dùng list để ghép ma trận hệ số với ma trận 0 của hệ số tự do 
        for i in range(0,l):
            check2.append(np.vstack((check1[i],Z)))
        
        # Dùng list để ghép ma trận hệ số của căn với hệ số của vế phải 
        for i in range(0,l):
            mt=np.vstack((np.reshape(meanA[i,:],(1,n)),check2[i])).T
            check3.append(np.array(mt,dtype=np.float32))
        
        # Khởi tạo ma trận G chứa các hệ số của ma trận hệ số vừa ghép với list hệ số của các ràng buộc tuyến tính (nếu có)
        #G=[ matrix( check3[0].tolist()) ]
        G+=[matrix( check3[0].tolist()) ]
        for i in range(1,l):
            G+=[ matrix( check3[i].tolist()) ]
        # h là ma trận chứa hệ số tự do 
        for i in range(0,int(l/2)):
            h.append(matrix([[k1*alpha[i]]+[0.0]*n +[0.0]]))
        for i in range(0,int(l/2)):
            h.append(matrix([[(-1)*k2*alpha[i]]+[0.0]*n +[0.0]]))
        h=[0]*l+h
        y_mu=matrix([0]*b+y_mu)
        y_hs=matrix([0]*b+[1]*(n-b))
        def F(x = None, z = None):
            if x is None:  return 0, matrix(0.0, (n,1))
            #if max(abs(x)) >= 1.0:  return None
            u = 1+exp(x)
            val = -sum(mul(y_mu,x)-mul(y_hs,log(u)))
            Df = (-1*y_mu+mul(y_hs,div(exp(x),u))).T
            
            if z is None:  return val, Df
            H = spdiag( z[0] * mul(y_hs,div(exp(x),u**2)))
            #H = z[0] * (y_hs*exp(x))/(u**2)
            return val, Df, H
        # Ghép ma trận hệ số tự do với các hệ số vế phải của ràng buộc tuyến tính 
        # q,_=matrix(h).size
        dims = {'l': l, 'q': ((n+2)*np.ones(l).astype(int)).tolist(), 's': [0]}
        solvers.options['show_progress'] = False
        # Hàm tìm nghiệm tối ưu 
        x = solvers.cp(F, matrix(G), matrix(h),dims)['x']
        return x

    def weight_train(self,X_train,X_val,y_train,y_val,n_c,n_v,alpha,option,max_loop,threshold=None):

        if threshold is False:
            max=0
        else:
            max=threshold
        pro = None
        flag=True
        tmp=0
        best_w=0
        count=-1

        ret = True
        while flag:
            try:
                self.weight=[]
                self.list_k=[]
                count+=1
                print("Loop : ",count)
                if count== max_loop:
                    ret = False
                    break
                if option == "Kmean":
                    meanA,varA,y_mu = self.S_L_R_Kmean(X_train,y_train,n_c,n_v)
                if option =="Quantile":
                    meanA,varA,y_mu = self.S_L_R_Quantile(X_train,y_train,n_c,n_v)
                m=np.array(meanA).shape[1] 
                for k in tqdm(range(1,100,1)):
                    self.list_k.append(k/100)
                    x=self.model(meanA,varA,k/100,m,n_v,y_mu,alpha)
                    W=np.array(x)
                    W=W[0:n_v,:]
                    self.weight.append(W)
                    X_val["bias"]=1
                    check=np.dot(X_val.to_numpy(),W).reshape(-1)
                    check1=1/(1+np.exp(-check))
                    for i in range(0,len(check1)):
                        if check1[i]>0.5:
                            check1[i]=1
                        else:
                            check1[i]=0
                    tmp=accuracy_score(np.array(y_val), check1)
                    #print("Accuracy at present : ", tmp)
                    if tmp>max:
                        pro=k/100
                        self.key_weight=W
                        flag = False
                    if tmp>best_w:
                        best_w=tmp
                print("Best accuracy: ",best_w)
                if pro is None:
                    flag = True
            except:
                continue
        self.pro=pro
        return self.weight,self.list_k,ret

    def weight_eval(self,X_train,y_train,n_c,n_v,alpha,option):

        num=0
        try:
            self.weight=[]
            if option == "Kmean":
                start=time.time()
                meanA,varA,y_mu = self.S_L_R_Kmean(X_train,y_train,n_c,n_v)
                
            if option =="Quantile":
                start=time.time()
                meanA,varA,y_mu = self.S_L_R_Quantile(X_train,y_train,n_c,n_v)
                
            m=np.array(meanA).shape[1]
            for k in range(1,100,1):
                x=self.model(meanA,varA,k/100,m,n_v,y_mu,alpha)
                W=np.array(x)
                W=W[0:n_v,:]
                self.weight.append(W)
            num=n_c
            end=time.time()
        except:
            end=time.time()
            num=0
        run_time=end-start
        return self.weight,num,run_time

    def weight_eval_alpha(self,X_train,y_train,n_c,n_v,alpha,beta,option):

        ret=False
        try:
            self.weight=[]
            if option == "Kmean":
                start=time.time()
                meanA,varA,y_mu = self.S_L_R_Kmean(X_train,y_train,n_c,n_v)
                
            if option =="Quantile":
                start=time.time()
                meanA,varA,y_mu = self.S_L_R_Quantile(X_train,y_train,n_c,n_v)
                
            m=np.array(meanA).shape[1] 
            for k in alpha:
                alpha_ = (k)*np.ones(n_c)
                x=self.model(meanA,varA,beta,m,n_v,y_mu,alpha_)
                W=np.array(x)
                W=W[0:n_v,:]
                self.weight.append(W)
            end=time.time()
            ret=True
        except:
            ret=False
            end=time.time()
        run_time=end-start
        return self.weight,run_time,ret

    def predict(self,w,X_val):

        X_val["bias"]=1
        c_=np.dot(X_val.to_numpy(),w).reshape(-1)
        c_1=1/(1+np.exp(-c_))
        for i in range(0,len(c_1)):
            if c_1[i]>0.5:
                c_1[i]=1
            else:
                c_1[i]=0
        return c_1

    def accuracy(self,y_pred,y_test):

        return accuracy_score(np.array(y_test), y_pred)

    def confusion_matrix(self,y_pred,y_test):

        return confusion_matrix(np.array(y_test), y_pred)

    def classification_report(self,y_pred,y_test):

        return classification_report(np.array(y_test), y_pred)

    def roc_curve(self,y_pred,y_test):

        fpr, tpr, thresholds=roc_curve(np.array(y_test), y_pred)
        return fpr, tpr, thresholds

    def log_loss(self,y_pred,y_test):

        return log_loss(np.array(y_test), y_pred)

    def roc_auc_score(self,y_pred,y_test):

        return roc_auc_score(np.array(y_test), y_pred)

    def f1_score(self,y_pred,y_test):

        return f1_score(np.array(y_test), y_pred)

    def precision_score(self,y_pred,y_test):

        return precision_score(np.array(y_test), y_pred)

    def recall_score(self,y_pred,y_test):

        return recall_score(np.array(y_test), y_pred)

    def S_L_R_Kmean(self,X_train,y_train,n_c,n_v):

        K=X_train.copy()
        kmeans = KMeans(n_clusters=n_c)
        kmeans.fit(K)
        kmeans.cluster_centers_
        kmeans.labels_
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        dict_data = dict(zip(unique, counts))
        dict_data
        K["cluster"] = kmeans.labels_
        kmeans.score
        var_X=[]
        M=K.copy()
        N=K.copy()
        N['y']=y_train
        M['bias']=1
        M=pd.concat([M, pd.DataFrame(data=np.zeros((len(M),n_c)))], ignore_index=True,axis=1)
        test=(M.groupby(by=[n_v-1])).cov() #group by col "b-1"
        j=0
        for i in range(0,n_c):
            var_X.append((np.reshape(test.iloc[j:(j+n_v+n_c),:].to_numpy(),(-1))).tolist())
            j=j+n_v+n_c
        test1=N.groupby(by=['cluster']).mean()
        test1['bias']=1
        var_X=np.vstack((var_X,var_X))
        Z_1=test1
        Z_2=test1
        test3=pd.concat([Z_1, Z_2], ignore_index=True,axis=0)
        y_=test1['y'].to_numpy()
        y_mu=y_.tolist()
        test3=test3.drop(['y'],axis=1)
        E=pd.DataFrame(data=np.vstack((-np.identity(n_c),-np.identity(n_c))))
        mean_X=pd.concat([test3, E], ignore_index=True,axis=1).to_numpy()
        return mean_X,var_X,y_mu

    def S_L_R_Quantile(self,X_train,y_train,n_q,n_v):

        quan=[0]
        for alpha in range(1,(n_q+1)):
           quan.append(y_train.quantile(alpha/int(n_q)))
        var_X=[]
        M=X_train.copy()
        N=X_train.copy()
        y_train_c=y_train.copy()
        M['bias']=1
        M=pd.concat([M, pd.DataFrame(data=np.zeros((len(M),n_q)))], ignore_index=True,axis=1)
        test=(M.groupby(pd.cut(y_train,quan))).cov()
        j=0
        for i in range(0,n_q):
            var_X.append((np.reshape(test.iloc[j:(j+n_v+n_q),:].to_numpy(),(-1))).tolist())
            j=j+n_v+n_q
        test1=(N.groupby(pd.cut(y_train_c,quan))).mean()
        test1['y']=quan[1:]
        test1['bias']=1
        var_X=np.vstack((var_X,var_X))
        test3=pd.concat([test1, test1], ignore_index=True,axis=0)
        y_=test1['y'].to_numpy()
        y_mu=y_.tolist()
        test3=test3.drop(['y'],axis=1)
        E=pd.DataFrame(data=np.vstack((-np.identity(n_q),-np.identity(n_q))))
        mean_X=pd.concat([test3, E], ignore_index=True,axis=1).to_numpy()
        return mean_X,var_X,y_mu

    def train(self,X_train,X_test,X_val,y_train,y_test,y_val,n_c,n_v,alpha,option,max_loop,threshold=None):

        print("Start training !")
        acc_s=[]
        f1_s=[]
        p_s=[]
        r_s=[]
        list_k=[]
        weight_all=[]
        y_scoring, y_pred , g_w = scoring_data(X_train,X_val,y_train)
        alpha_ = (alpha)*np.ones(n_c)
        if threshold is False:
            flag=False
            print("Accuracy for original Logistic Regression Model on testset: ",self.accuracy(self.predict(g_w,X_test),y_test))
            print("Accuracy for original Logistic Regression Model on validationset: ",self.accuracy(y_pred,y_val))

        else:
            flag=True
            weight_all,list_k,ret=self.weight_train(X_train,X_val,y_scoring,y_val,n_c,n_v,alpha_,option,max_loop,threshold)
            if ret:
                for item in weight_all:
                    g=self.predict(item,X_val)
                    acc_s.append(self.accuracy(g,y_val))
                    f1_s.append(self.f1_score(g,y_val))
                    p_s.append(self.precision_score(g,y_val))
                    r_s.append(self.recall_score(g,y_val))
            elif not ret:
                flag=False
                print("Try lower threshold !")
        print("Complete training !")
        return weight_all,list_k,acc_s,f1_s,p_s,r_s,g_w,flag

    def eval(self,X_train,X_test,X_val,y_train,y_test,y_val,n_v,alpha,option,min_iters,max_iters):

        print("Start evaluating !")
        acc_s=[]
        f1_s=[]
        p_s=[]
        r_s=[]
        acc_g=[]
        f1_g=[]
        p_g=[]
        r_g=[]
        num=[]
        time_run=[]
        y_scoring, _ , g_w = scoring_data(X_train,X_val,y_train)
        if option=="Kmean":
            print("Evaluate by K-Means clustering !")
            for i in tqdm(range(min_iters,max_iters)):
                acc_s_=[]
                f1_s_=[]
                p_s_=[]
                r_s_=[]
                alpha_=alpha*np.ones(i)
                weight_all,n,run_time=self.weight_eval(X_train,y_scoring,i,n_v,alpha_,option)

                if n!=0:
                    num.append(n)
                    time_run.append(run_time)
                    for item in weight_all:
                        g=self.predict(item,X_val)
                        acc_s_.append(self.accuracy(g,y_val))
                        f1_s_.append(self.f1_score(g,y_val))
                        p_s_.append(self.precision_score(g,y_val))
                        r_s_.append(self.recall_score(g,y_val))
                    index1= np.argmax(acc_s_)
                    index2= np.argmax(f1_s_)
                    index3= np.argmax(p_s_)
                    index4= np.argmax(r_s_)
                    g_w=np.reshape(np.array(g_w),(len(g_w),1))
                    acc_s.append(self.accuracy(self.predict(weight_all[index1],X_test),y_test))
                    f1_s.append(self.f1_score(self.predict(weight_all[index2],X_test),y_test))
                    p_s.append(self.precision_score(self.predict(weight_all[index3],X_test),y_test))
                    r_s.append(self.recall_score(self.predict(weight_all[index4],X_test),y_test))
                    acc_g.append(self.accuracy(self.predict(g_w,X_test),y_test))
                    f1_g.append(self.f1_score(self.predict(g_w,X_test),y_test))
                    p_g.append(self.precision_score(self.predict(g_w,X_test),y_test))
                    r_g.append(self.recall_score(self.predict(g_w,X_test),y_test))
                    X_val=X_val.drop('bias',axis=1)
                    X_test=X_test.drop('bias',axis=1)
        else:
            print("Evaluate by Q-Quantiles !")
            for i in tqdm(max_iters):
                acc_s_=[]
                f1_s_=[]
                p_s_=[]
                r_s_=[]
                alpha_=alpha*np.ones(i)
                weight_all,n,run_time=self.weight_eval(X_train,y_scoring,i,n_v,alpha_,option)
                
                if n!=0:
                    num.append(n)
                    time_run.append(run_time)
                    for item in weight_all:
                        g=self.predict(item,X_val)
                        acc_s_.append(self.accuracy(g,y_val))
                        f1_s_.append(self.f1_score(g,y_val))
                        p_s_.append(self.precision_score(g,y_val))
                        r_s_.append(self.recall_score(g,y_val))
                    index1= np.argmax(acc_s_)
                    index2= np.argmax(f1_s_)
                    index3= np.argmax(p_s_)
                    index4= np.argmax(r_s_)
                    g_w=np.reshape(np.array(g_w),(len(g_w),1))
                    acc_s.append(self.accuracy(self.predict(weight_all[index1],X_test),y_test))
                    f1_s.append(self.f1_score(self.predict(weight_all[index2],X_test),y_test))
                    p_s.append(self.precision_score(self.predict(weight_all[index3],X_test),y_test))
                    r_s.append(self.recall_score(self.predict(weight_all[index4],X_test),y_test))
                    acc_g.append(self.accuracy(self.predict(g_w,X_test),y_test))
                    f1_g.append(self.f1_score(self.predict(g_w,X_test),y_test))
                    p_g.append(self.precision_score(self.predict(g_w,X_test),y_test))
                    r_g.append(self.recall_score(self.predict(g_w,X_test),y_test))
                    X_val=X_val.drop('bias',axis=1)
                    X_test=X_test.drop('bias',axis=1)
        print("Complete evaluating !")       
        return acc_s,f1_s,p_s,r_s,acc_g,f1_g,p_g,r_g,num,time_run
            
