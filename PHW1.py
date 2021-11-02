import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import sklearn.svm as svm

# Define a function that performs overall process
def best_combination_model(dataframe, target, scalers = None, encoders = None, models = None) :

    # Split X, y (In this dataset, target is 'class' column)
    X = dataframe.drop(target, axis = 1)
    y = dataframe[target]

    # Set the columns to be scaled / encoded
    scale_col = ['clump_thickness','size_uniformity','shape_uniformity','marginal_adhesion','epithelial_size','bare_nucleoli','bland_chromatin','normal_nucleoli','mitoses']
    encode_col = ''

    # Split the train set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

    # Settings before creating model
    # Use 3 Encoders > OrdinalEncoder(), OneHotEncoder(), LabelEncoder()
    # Use 4 Sclaers > StandardSclaer(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()
    # Use 4 Models > DecisionTreeClassifier (Entropy), DecisionTreeClassifier (gini), LogisticRegression, SVC
    
    if encoders == None:
        encode = [OrdinalEncoder(), OneHotEncoder(), LabelEncoder()]
    else: 
        encode = encoders

    if scalers == None:
        scale = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
    else: 
        scale = scalers

    if models == None:
        DecisionTreeEntropy = DecisionTreeClassifier()
        DecisionTreeGini = DecisionTreeClassifier()
        LR = LogisticRegression()
        SVCs = svm.SVC()
        classifier = [DecisionTreeEntropy, DecisionTreeGini, LR, SVCs]
    else : 
        classifier = models



    """
    each variables means :
    (model name)_bestscore : model's best score
    (model name)_bestparams : model's best parameters
    (model name)_best_enc_scale : model's best encoder & scaler
    (model name)_grid_params : models's parameters for GridSearchCV
    """
    DecisionTreeEntropy_bestscore = 0
    DecisionTreeEntropy_bestparams = []
    DecisionTreeEntropy_best_enc_scale = []
    DecisionTreeEntropy_grid_params = {
        'criterion' : ['entropy'],
        'min_samples_split' : [3,5,7],
        'max_features' : [3,5,7],
        'max_depth' : [3,4,5],
        'max_leaf_nodes' : [30,50,70,90,100]
    }

    DecisionTreeGini_bestscore = 0
    DecisionTreeGini_bestparams = []
    DecisionTreeGini_best_enc_scale = []
    DecisionTreeGini_grid_params = {
        'criterion' : ['gini'],
        'min_samples_split' : [3,5,7],
        'max_features' : [3,5,7],
        'max_depth' : [3,4,5],
        'max_leaf_nodes' : [30,50,70,90,100]
    }

    LogisticRegression_bestscore = 0
    LogisticRegression_bestparams = []
    LogisticRegression_best_enc_scale = []
    LogisticRegression_grid_params = {
        'C':[0.1, 1, 10],
        'penalty':['l2'],
        'solver':['lbfgs','liblinear']
    }

    SVC_bestscore = 0
    SVC_bestparams = []
    SVC_best_enc_scale = []
    SVC_grid_params = {
        'C' : [0.001, 0.01, 0.1],
        'gamma' : [0.001,0.01,0.1,1,10,100],
        'kernel' : ['linear','poly','rbf']
    }


    # Performing the process
    for i in scale :
        for j in encode :
            # 1. Scaling
            scaler = i
            scaler = pd.DataFrame(scaler.fit_transform(X[scale_col]))

            # 2. Encoding (But, if there's no columns to encode, skip this step)
            if encode_col != '' :
                if j == OrdinalEncoder() :
                    enc = j
                    enc = enc.fit_transform(X[encode_col])
                    new_df = pd.concat([scaler, enc], axis = 1)
                elif j == LabelEncoder() :
                    enc = j
                    enc = enc.fit_transform(X[encode_col])
                    new_df = pd.concat([scaler, enc], axis = 1)
                else :
                    dum = pd.DataFrame(pd.get_dummies(X[encode_col]))
                    new_df = pd.concat([scaler, dum], axis = 1)
            
            # 3. Modeling
            for model in classifier :

                # For each model, perform GridSearchCV and fit the model
                # And then, get the best score and best scaler / encoder

                if model == DecisionTreeEntropy :
                    DecisionEntropy = GridSearchCV(model, param_grid=DecisionTreeEntropy_grid_params, cv=3)
                    DecisionEntropy.fit(X_train, y_train)
                    score = DecisionEntropy.score(X_test, y_test)

                    if score > DecisionTreeEntropy_bestscore :
                        DecisionTreeEntropy_bestscore = score
                        DecisionTreeEntropy_bestparams = DecisionEntropy.best_params_
                        DecisionTreeEntropy_best_enc_scale = [i,j]
                
                elif model == DecisionTreeGini :
                    DecisionGini = GridSearchCV(model, param_grid=DecisionTreeGini_grid_params, cv=3)
                    DecisionGini.fit(X_train, y_train)
                    score = DecisionGini.score(X_test, y_test)

                    if score > DecisionTreeGini_bestscore :
                        DecisionTreeGini_bestscore = score
                        DecisionTreeGini_bestparams = DecisionGini.best_params_
                        DecisionTreeGini_best_enc_scale = [i,j]

                elif model == LR :
                    LReg = GridSearchCV(model, param_grid=LogisticRegression_grid_params, cv=3)
                    LReg.fit(X_train, y_train)
                    score = LReg.score(X_test, y_test)

                    if score > LogisticRegression_bestscore :
                        LogisticRegression_bestscore = score
                        LogisticRegression_bestparams = LReg.best_params_
                        LogisticRegression_best_enc_scale = [i,j]

                else :
                    SupVC = GridSearchCV(model, param_grid=SVC_grid_params, cv=3)
                    SupVC.fit(X_train, y_train)
                    score = SupVC.score(X_test, y_test)

                    if score > SVC_bestscore :
                        SVC_bestscore = score
                        SVC_bestparams = SupVC.best_params_
                        SVC_best_enc_scale = [i,j]
    
    # Print the results
    print("[Decision Tree - Entropy]")
    print('Best Score : ',DecisionTreeEntropy_bestscore)
    print('Best parameters : ', DecisionTreeEntropy_bestparams)
    print('Best Encoder / Scaler :', DecisionTreeEntropy_best_enc_scale)
    print()

    print("[Decision Tree - Entropy]")
    print('Best Score : ',DecisionTreeGini_bestscore)
    print('Best parameters : ', DecisionTreeGini_bestparams)
    print('Best Encoder / Scaler :', DecisionTreeGini_best_enc_scale)
    print()

    print("[Logistic Regression]")
    print('Best Score : ', LogisticRegression_bestscore)
    print('Best parameters : ', LogisticRegression_bestparams)
    print('Best Encoder / Scaler :', LogisticRegression_best_enc_scale)
    print()

    print("[Support Vector Machine]")
    print('Best Score : ',SVC_bestscore)
    print('Best parameters : ', SVC_bestparams)
    print('Best Encoder / Scaler :', SVC_best_enc_scale)
    print()

    return


# Load the dataset
df = pd.read_csv("/Users/jaeuk/Desktop/Python/ML/breast-cancer-wisconsin.data")

df.columns = ['id','clump_thickness','size_uniformity','shape_uniformity','marginal_adhesion','epithelial_size','bare_nucleoli','bland_chromatin','normal_nucleoli','mitoses','class']

##########################
# Preprocessing

# drop 'id' column
df = df.drop(['id'], axis = 1)

# drop the row that has '?' value
df2 = df[df['bare_nucleoli'] == '?'].index
df = df.drop(df2)

# Reset the index and print the dataframe
df = df.reset_index(drop=True)

# Call the function for the modeling and print
best_combination_model(df, 'class')