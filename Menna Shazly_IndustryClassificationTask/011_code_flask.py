import nltk
import string
import pandas as pd
import numpy as np
import sklearn
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask
from flask import render_template



def load_file_glove():
    dict = {}
    with open('glove.6B.300d.txt', 'r', encoding="utf8") as f:
        for line in f:
            dict[line.split()[0]] = list(map(float,line.split()[1:]))
        f.close()
    return  dict



def search_dict(data, dict):
    total_list = []
    for list in data:
        res_list = []
        for word in list:
            if word in dict.keys():
                res_list.append(dict[word])
            else:
                res_list.append([0]*300)
        total_list.append(res_list)
    return total_list



def sum_method(data):
    sum_list = []
    temp_sum_list = []
    for x in data:
        temp_sum_list = sum(np.array(x))
        sum_list.append(temp_sum_list.tolist())
    return sum_list



def get_label(size_data,label):
    label_list =[]
    for i in range(size_data):
        label_list.append(label)
    return label_list



def concat_data(list1,list2):
    list_all = []
    for l1 in list1 :
        list_all.append(l1)
    for l2 in list2:
         list_all.append(l2)
    return list_all



def Train_Test_Split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state = 42)
    return X_train, X_test, y_train, y_test



def classify_KNN(X_train, X_test, y_train, y_test):
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return neigh,acc



def predict_KNN(KNN_model, test , dict ):
    test_list= []
    tokens =nltk.word_tokenize(test)
    test_list.append(tokens)
    x = search_dict(test_list,dict)
    sum_list  =  sum_method(x)
    y_pred = KNN_model.predict(sum_list)
    return y_pred

def output(prediction):
    if prediction ==0 :
        return ('Accountancy')
    elif prediction == 1:
        return ('Education')
    elif prediction == 2:
        return ('IT')
    else :
        return ('Marketing')

def final(input_test):
    data = pd.read_csv("Job titles and industries.csv")
    data.sort_values("industry", inplace = True)
    data = data.drop_duplicates(subset ="job title",keep = 'first')
    data['cleaned'] = data['job title'].apply(lambda x:''.join([i for i in x if i not in string.punctuation]))
    data['tokenized_sents'] = data.apply(lambda row: nltk.word_tokenize(row['cleaned']), axis=1)
    glove_dict = load_file_glove()
    all_glove_data = search_dict(data['tokenized_sents'],glove_dict)
    all_sum = sum_method(all_glove_data)
    acc_label = get_label(263,0)
    ed_label = get_label(972,1)
    it_label = get_label(1514,2)
    mark_label = get_label(1141,3)
    list1 = concat_data(acc_label,ed_label)
    list2 = concat_data(list1,it_label)
    all_label = concat_data(list2,mark_label)
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(all_sum, all_label)
    X_train, X_test, y_train, y_test = Train_Test_Split(X_resampled, y_resampled)
    model_knn,acc_knn = classify_KNN(X_train, X_test, y_train, y_test)
    model_knn,acc_knn = classify_KNN(X_train, X_test, y_train, y_test)
    prediction_KNN = predict_KNN(model_knn,input_test,glove_dict)
    pred = output(prediction_KNN)
    return pred

#ay7aga = final('developer')
#print(ay7aga)