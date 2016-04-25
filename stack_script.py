'''
Created on 14.04.2016

@author: Tobias
'''
import sklearn_helpers as skhelper
from stacking.stacking_model import *

import pandas as pd
import numpy as np
from time import strptime
import datetime
from collections import Counter

from sklearn import preprocessing, naive_bayes
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.ensemble.forest import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score, StratifiedKFold,\
    train_test_split
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
        
def create_sub(y_pred):
    print("generating submission")
    sample_sub = pd.read_csv("data/sample_submission.csv")
    n = sample_sub.shape[0]
    names = ["Adoption","Died","Euthanasia","Return_to_owner","Transfer"]
    for i in range(n):
        for col in range(5):
            sample_sub.loc[i,names[col]] = y_pred[i,col]
    sample_sub.to_csv("n_submission.csv", sep=",", index=False)

def preproc(df):
    le = preprocessing.LabelEncoder()
    print("preprocessing")
    n = df.shape[0]
    df = df.fillna("NaN")

    # name == nan
    name_vec = np.zeros(n)
    name_length = np.zeros(n)
    name_list = []
    i = 0
    for name in df["Name"]:
        if name == "NaN":
            name_vec[i] = 0
            name_length[i] = 0
        else:
            name_list.append(name)
            name_vec[i] = 1
            name_length[i] = len(name)
        i += 1
    df["hasName"] = name_vec
    df["name_length"] = name_length
    
    c = Counter(name_list)
    names_vec = np.zeros(n)
    i = 0
    for name in df["Name"].values:
        if c[name] != 0:
            names_vec[i] = 1 - c[name]/n
        else:
            names_vec[i] = 0
        i += 1
    df["name_weirdness"] = names_vec
    df = df.drop("Name",axis=1)
    
    # map animal
    mapping = {'Dog': 0, 'Cat': 1}
    df = df.replace({'AnimalType': mapping})
    
    # color
    i = 0
    color1_vec = []
    color2_vec = []
    number_colors = np.zeros(n)
    texture_vec = []
    texture2_vec = []
    for color in df["Color"].values:
        if '/' in color:
            number_colors[i] = 2
            color1 = color.split("/")[0]
            color2 = color.split("/")[1]
            color1 = color1.split(" ")
            color1_vec.append(color1[0])
            color2 = color2.split(" ")
            color2_vec.append(color2[0])
            if len(color1) > 1 and len(color2) == 1:
                texture_vec.append(color1[1])
                texture2_vec.append("0")
            if len(color2) > 1 and len(color1) == 1:
                texture_vec.append(color2[1])
                texture2_vec.append("0")
            if len(color1) == 1 and len(color2) == 1:
                texture_vec.append("0")
                texture2_vec.append("0") 
            if len(color1) > 1 and len(color2) > 1:
                texture_vec.append(color1[1])
                texture2_vec.append(color2[1])
        else:
            color2_vec.append("0")
            texture2_vec.append("0") 
            number_colors[i] = 1
            color = color.split(" ")
            if len(color) > 1:
                texture_vec.append(color[1])
                color1_vec.append(color[0])
            else:
                texture_vec.append("0")
                color1_vec.append(color[0])      
        i += 1
        
    
    color1_vec = le.fit_transform(color1_vec)
    color2_vec = le.fit_transform(color2_vec)    
    texture_vec = le.fit_transform(texture_vec)
    texture2_vec = le.fit_transform(texture2_vec)
    df["color1"] = color1_vec
    df["color2"] = color2_vec
    df["number_of_colors"] = number_colors
    df["texture"] = texture_vec
    df["texture2"] = texture2_vec
    
    # sex to male/female/unknown
    sex_vec = np.zeros(n)
    new_vec = np.zeros(n)
    i = 0
    for sex in df["SexuponOutcome"].values:
        if sex == "Unknown" or sex == "NaN":
            sex_vec[i] = 0
            new_vec[i] = 0
        else:
            if sex.split(" ")[1] == "Male":
                sex_vec[i] = 2
            elif sex.split(" ")[1] == "Female":
                sex_vec[i] = 1
                
            if sex.split(" ")[0] == "Intact":
                new_vec[i] = 2
            elif sex.split(" ")[0] == "Spayed":
                new_vec[i] = 1
            elif sex.split(" ")[0] == "Neutered":
                new_vec[i] = 3
        i += 1
    df["Sex"] = sex_vec
    df["Sex_stat"] = new_vec
    df = df.drop("SexuponOutcome",axis=1)
    
    # mix
    mix_vec = np.zeros(n)
    i = 0
    breed_list = []
    for breed in df["Breed"].values:
        if breed.split(" ")[-1] == "Mix":
            mix_vec[i] = 1
            breed_list.append(breed[:-4])
        else:
            breed_list.append(breed)
            mix_vec[i] = 0
        i += 1
    df["Mix"] = mix_vec
    c = Counter(breed_list)
    breed_weird_vec = np.zeros(n)
    i = 0
    for breed in breed_list:
        if c[breed] != 0:
            breed_weird_vec[i] = 1 - c[breed]/n
        else:
            breed_weird_vec[i] = 0
        i += 1
    df["breed_weirdness"] = breed_weird_vec
    
    # Age
    age_week = np.zeros(n)
    age_month = np.zeros(n)
    age_year = np.zeros(n)
    i = 0
    for age in df["AgeuponOutcome"]:
        age = age.split(" ")
        if age[-1][0:4] == "week":
            age_week[i] = int(age[0])
            age_month[i] = 0 
            age_year[i] = 0
        elif age[-1][0:5] == "month":
            age_week[i] = 0
            age_month[i] = int(age[0])
            age_year[i] = 0
        elif age[-1][0:4] == "year":
            age_week[i] = 0
            age_month[i] = 0
            age_year[i] = int(age[0])
        i += 1
    df = df.drop("AgeuponOutcome",axis=1)
    AgeuponOutcome_vec = np.zeros(n)
    for i in range(n):
        AgeuponOutcome_vec[i] = age_week[i]*7 + age_month[i]*30 + age_year[i]*365
    df["AgeuponOutcome"] = AgeuponOutcome_vec
    
    # use the time variable
    hour_vec = np.zeros(n)
    week_vec = np.zeros(n)
    month_vec = np.zeros(n)
    day_of_month_vec = np.zeros(n)
    i = 0
    for date in df["DateTime"].values:
        date_ = date.split(" ")[0]
        time_month = date_.split("-")[1]
        day_of_month_vec[i] = date_.split("-")[2]
        time_ = date.split(" ")[1]
        time_hour = time_.split(":")[0]
        hour_vec[i] = time_hour
        month_vec[i] = time_month
        # week
        date = datetime.datetime(*(strptime(date, '%Y-%m-%d %H:%M:%S')[0:6]))
        week_vec[i] = date.weekday()
        i += 1
    df["Hour"] = hour_vec
    df["Weekday"] = week_vec
    df["Month"] = month_vec
    df["Day_of_month"] = day_of_month_vec
    
    # drop what should be encoded
    df = df.drop(["DateTime"], axis=1)
    df =df.drop(["Breed","Color"],axis=1)
    return df

if __name__ == "__main__":
    print("load data")
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")

    # map the classes
    mapping = {'Adoption': 0, 'Died': 1, 'Euthanasia': 2,
               'Return_to_owner': 3, 'Transfer': 4}
    df_train = df_train.replace({'OutcomeType': mapping})
    y = df_train["OutcomeType"].values
    df_train = df_train.drop(["OutcomeType","OutcomeSubtype","AnimalID"],axis=1)

    n_train = df_train.shape[0]
    df_all = df_train.append(df_test, ignore_index=True)
    df_all = preproc(df_all)
    df_all = df_all.drop("ID",axis=1)
    df_train = df_all.iloc[:n_train]
    df_test = df_all.iloc[n_train:]

    X = df_train.values
    X_test = df_test.values
    feature_names = df_all.columns.values.tolist()
    
    print(X.shape)
    
    print("build the model")
    clf1 = RandomForestClassifier(n_estimators=100,random_state=571,max_features=8,max_depth=13,n_jobs=1)
    clf2 = KNeighborsClassifier(n_neighbors=250, p=1, weights="distance")
    clf3 = ExtraTreesClassifier(n_estimators=100,max_depth=14, max_features=12,random_state=571,n_jobs=1)
    nb = GaussianNB()
    rft = RandomForestClassifier(n_estimators=100,random_state=571,max_features=8,max_depth=13,n_jobs=1)
    clf4 = Pipeline([('rft', rft), ('ng', nb)])
    clf5 = GradientBoostingClassifier(n_estimators=100,random_state=571,max_depth=6, max_features=7)
    
    clf6 = RandomForestClassifier(n_estimators=1000,max_features=10,max_depth=14,n_jobs=1) # feats = 10
    clf7 = GradientBoostingClassifier(n_estimators=100,max_depth=9, max_features=7)  # feats = 7
    
    first_stage = [
                   ("rf",clf1),
                   ("knn",clf2),
                   ("et",clf3),
                   ("rf_gnb",clf4),
                   ("gbm",clf5)
                   ]
    second_stage = [
                    ("gbm",clf7),
                    ("rf",clf6)
                     ]
    
    weights = [3,1]
    stack = StackingClassifier(stage_one_clfs=first_stage,stage_two_clfs=second_stage,weights=weights, n_runs=10, use_append=False)
    
    skf = StratifiedKFold(y, n_folds=5,random_state=571)
    
#     print("Training")
#     stack.fit(X,y)
#     print("Predict")
#     y_pred = stack.predict_proba(X_test)
#     create_sub(y_pred)
    
#     print("CV")
#     scores = cross_val_score(stack,X,y,scoring="log_loss",cv=skf)
#     print(scores)
#     print("CV-Score: %.3f" % -scores.mean())
    # with append:        Score: 0.783
    # without append:     CV-Score: 0.843
    
    # gridsearch
    params1 = {
               "max_depth": [3,4],
               "max_features": [3,4,5]
               }
    
    params2 = {
               "max_depth": [7,8],
               "max_features": [4,5]
               }
    paramsset = [params1, params2]
    
    stack = StackingClassifier(stage_one_clfs=first_stage,stage_two_clfs=second_stage,weights=weights, n_runs=10, use_append=False,
                               do_gridsearch=True, params=paramsset, cv=skf, scoring="log_loss")
    stack.fit(X,y)