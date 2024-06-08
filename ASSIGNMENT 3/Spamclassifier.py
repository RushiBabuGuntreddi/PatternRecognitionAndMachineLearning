# %%
import pandas as pd
import os
import numpy as np
import re
# import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
import chardet
import numpy as np
from sklearn.model_selection import train_test_split



# Load the dataset
with open('emails.csv', 'rb') as f:
    rawdata = f.read()
enc = chardet.detect(rawdata)
enc = enc['encoding']
data = pd.read_csv('emails.csv',encoding=enc, usecols=['text', 'spam'])
data.rename(columns={'text': 'email', 'spam': 'label'}, inplace=True)

def preprocess(text) :
    
        
        text=re.sub(r'[^a-zA-Z\s]','',text)
        
        text=text.lower()
        text = text.replace("subject",'')
        return text

data["email"]=data["email"].apply(preprocess)
emails=data["email"]
emails=np.array(emails)
labels=data["label"]
labels=np.array(labels)


#Naive Bayes
vectorizer = CountVectorizer(binary=True,stop_words="english")
vectorizer.fit(emails)
train_vectors=vectorizer.transform(emails)
train_vectors=train_vectors.toarray()
train_vectors=np.array(train_vectors)





# %%
number_of_spam_train=np.sum(labels==1) 
number_of_nonspam_train=np.sum(labels==0)
total_number_train=emails.shape[0]
prior=(number_of_spam_train+1)/(total_number_train+2)#Laplace Smoothing add vector with all onesto both spam and ham class.
train_spam_vectors=train_vectors[labels==1]
train_nonspam_vectors=train_vectors[labels==0]
prob_spam=(np.sum(train_spam_vectors,axis=0)+1)/(number_of_spam_train+1)
prob_nonspam=(np.sum(train_nonspam_vectors,axis=0)+1)/(number_of_nonspam_train+1)
w_NB=np.log(prob_spam*(1-prob_nonspam)/(prob_nonspam*(1-prob_spam)),dtype="float64")
bias_NB=np.sum(np.log((1-prob_spam)/(1-prob_nonspam),dtype="float64"))+np.log(prior/(1-prior),dtype="float64")
NB_train_predicted_label=np.full(total_number_train,0)
for i in range(total_number_train):

    sum=np.dot(train_vectors[i],w_NB)+bias_NB
    
    
    if sum >= 0 :
       
        NB_train_predicted_label[i]=1


   

# %%
def sig(a) :
    return 1/(1+np.exp(-a))
vectorizer2=TfidfVectorizer(stop_words="english")
vectorizer2.fit(emails)
LR_train_vectors=vectorizer2.transform(emails)
LR_train_vectors=LR_train_vectors.toarray()
LR_train_vectors=np.array(LR_train_vectors)
W_LR=np.zeros(LR_train_vectors.shape[1])
N=0.1
for itr in range(100) :
    labels_pred=sig(np.dot(LR_train_vectors,W_LR))
    W_LR+=N*np.dot(LR_train_vectors.T,(labels-labels_pred))




# %%

LR_train_predicted_label=np.full(total_number_train,0)
for i in range(total_number_train):

    sum=np.dot(LR_train_vectors[i],W_LR)
    
    
    if sum >= 0 :
       
        LR_train_predicted_label[i]=1


    

# %%

vectorizer3=CountVectorizer(binary=False,stop_words="english")
vectorizer3.fit(emails)
SVM_train_vectors=vectorizer3.transform(emails)
SVM_train_vectors=SVM_train_vectors.toarray()
SVM_train_vectors=np.array(SVM_train_vectors)
svm_classifier=svm.LinearSVC(C=1,max_iter=10000,dual=False)
svm_classifier.fit(SVM_train_vectors,labels)
SVM_train_predictions=svm_classifier.predict(SVM_train_vectors)



# %%
acc=np.sum(NB_train_predicted_label==labels)/total_number_train
print("Naive Bayes train_accuracy  ",acc*100) 
LR_acc=np.sum(LR_train_predicted_label==labels)/total_number_train
print("Logistic Regression train_accuaracy  ",LR_acc*100)
SVM_acc=np.sum(SVM_train_predictions==labels)/total_number_train
print("SVM train_accuaracy  ",SVM_acc*100)

# %%
files=[f for f in os.listdir('test1') if f.endswith('.txt')]
predicted_label_NB=np.zeros(len(files))
predicted_label_LR=np.zeros(len(files))
predicted_label_SVM=np.zeros(len(files))
for i, file in enumerate(files):
    with open(os.path.join('test1', file), 'r') as f:
        email = f.read()

    # Preprocess the email
    email = preprocess(email)

    # Transform the email into features
    test_vector_NB = vectorizer.transform([email]).toarray()
    test_vector_LR = vectorizer2.transform([email]).toarray()
    test_vector_SVM = vectorizer3.transform([email]).toarray()
    sum1=np.dot(test_vector_NB,w_NB)+bias_NB
    sum2 = np.dot(test_vector_LR, W_LR)

    if sum1 >= 0:
        predicted_label_NB[i] = 1
    if sum2 >= 0:
        predicted_label_LR[i] = 1
    predicted_label_SVM[i]=svm_classifier.predict(test_vector_SVM)[0]




# %%
print("LR : ",predicted_label_LR.astype(int))
print("NB : ",predicted_label_NB.astype(int))
print("SVM : ",predicted_label_SVM.astype(int))
Final_predicted_label_for_test_emails=predicted_label_NB+predicted_label_LR+predicted_label_SVM
Final_predicted_label_for_test_emails=np.where(Final_predicted_label_for_test_emails>=2,1,0)
print("Final predictions for test emails : ",Final_predicted_label_for_test_emails)




