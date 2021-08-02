from sklearn.linear_model import LogisticRegression
import numpy as np

def scoring_data(X_train,X_val,y_train):

    # Logistic Regression
    classifier = LogisticRegression(random_state = 0)
    reg=classifier.fit(X_train, y_train)
    # Predicting the test set results
    Y_Pred = classifier.predict(X_val)
    #X_train=X_train.drop('bias',axis=1)
    #X_val=X_val.drop('bias',axis=1)
    # print(X_train)
    w__=reg.coef_[0]@X_train.T+reg.intercept_
    y_train=1/(1+np.exp(-w__))
    return y_train, Y_Pred, reg.coef_[0].tolist()+reg.intercept_.tolist()