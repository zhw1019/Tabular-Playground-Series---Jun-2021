import sys
path = sys.path[0]
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from deepforest import CascadeForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# data process
data = pd.read_csv(path + "/data/train.csv")
value_map = {'Class_1': 0, 'Class_2': 1, 'Class_3': 2, 'Class_4': 3, 'Class_5': 4, 'Class_6': 5, 'Class_7': 6, 'Class_8': 7, 'Class_9': 8}
data = data.replace({'target':value_map})
data = data.drop(columns=['id'])
x_train = data.iloc[:, :-1]
y_train = data['target']

test = pd.read_csv(path + '/data/test.csv')
x_test = test.iloc[:, 1:] # keep the id column for output

# model train
MODEL = 1
if (MODEL == 1):
    model = CascadeForestClassifier(n_jobs=2, n_estimators=4, n_trees=100)
    model.fit(x_train.values, y_train.values)
    proba = model.predict_proba(x_test.values)
    output = pd.DataFrame({'id': test['id'], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3], 'Class_5': proba[:,4], 'Class_6': proba[:,5], 'Class_7': proba[:,6], 'Class_8': proba[:,7],'Class_9': proba[:,8]})
    output.to_csv(path + '/result/my sub/my_submission_df.csv', index=False)

    model = RandomForestClassifier(n_jobs=2, n_estimators=500)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)
    output = pd.DataFrame({'id': test['id'], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3], 'Class_5': proba[:,4], 'Class_6': proba[:,5], 'Class_7': proba[:,6], 'Class_8': proba[:,7],'Class_9': proba[:,8]})
    output.to_csv(path + '/result/my sub/my_submission_rf.csv', index=False)

    model = HistGradientBoostingClassifier(max_iter=250,
                                           validation_fraction=None, 
                                           learning_rate=0.01, 
                                           max_depth=10, 
                                           min_samples_leaf=24, 
                                           max_leaf_nodes=60,
                                           random_state=123,
                                           verbose=1)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)
    output = pd.DataFrame({'id': test['id'], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3], 'Class_5': proba[:,4], 'Class_6': proba[:,5], 'Class_7': proba[:,6], 'Class_8': proba[:,7],'Class_9': proba[:,8]})
    output.to_csv(path + '/result/my sub/my_submission_HGBDT.csv', index=False)

    parms = {'learning_rate': 0.03817329673009776, 'gamma': 0.3993428240049768, 'reg_alpha': 3,
         'reg_lambda': 1, 'n_estimators': 334, 'colsample_bynode': 0.2695766080178446,
         'colsample_bylevel': 0.6832712495239914, 'subsample': 0.6999062848890633,
         'min_child_weight': 100, 'colsample_bytree': 0.34663755614898173}

    model = XGBClassifier(objective='multi:softprob',
                          eval_metric = "mlogloss",
                          num_class = 9,
                          tree_method = 'gpu_hist',
                          max_depth = 14,
                          use_label_encoder=False, **parms)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)
    output = pd.DataFrame({'id': test['id'], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3], 'Class_5': proba[:,4], 'Class_6': proba[:,5], 'Class_7': proba[:,6], 'Class_8': proba[:,7],'Class_9': proba[:,8]})
    output.to_csv(path + '/result/my sub/my_submission_xgboost.csv', index=False)  # 1.74868

    model = CatBoostClassifier(depth=8,
                               iterations=1000,
                               learning_rate=0.02,                            
                               eval_metric='MultiClass',
                               loss_function='MultiClass', 
                               bootstrap_type= 'Bernoulli',
                               leaf_estimation_method='Gradient',
                               random_state=2021,
                               task_type='GPU')
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)
    output = pd.DataFrame({'id': test['id'], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3], 'Class_5': proba[:,4], 'Class_6': proba[:,5], 'Class_7': proba[:,6], 'Class_8': proba[:,7],'Class_9': proba[:,8]})
    output.to_csv(path + '/result/my sub/my_submission_catboost.csv', index=False)

    model = LGBMClassifier(objective = 'multiclass',
                           reg_lambda = 10,
                           learning_rate = 0.1,
                           max_depth = 4,
                           seed = 2021,
                           colsample_bytree = 0.5,
                           subsample = 0.9,
                           is_unbalance = True)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_test)
    output = pd.DataFrame({'id': test['id'], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3], 'Class_5': proba[:,4], 'Class_6': proba[:,5], 'Class_7': proba[:,6], 'Class_8': proba[:,7],'Class_9': proba[:,8]})
    output.to_csv(path + '/result/my sub/my_submission_lgbm.csv', index=False)

#post process
def ensemble(main, support, coeff, believe_main): 
    sub1  = support.copy()
    sub1v = sub1.values    
    sub2  = main.copy() 
    sub2v = sub2.values
    imp  = main.copy()    
    impv = imp.values
    number = 0
    
    for i in range (len(main)):               
        
        row1 = sub1v[i,1:]
        row2 = sub2v[i,1:]
        row = (row2 * coeff) + (row1 * (1.0 - coeff))

        if (believe_main == 1):
            row1_argsort = np.argsort(row1)
            row2_argsort = np.argsort(row2) 
            if (row1_argsort[8] != row2_argsort[8]):
                row = row2
                number += 1
        impv[i, 1:] = row
    
    if (believe_main == 1):
        per = round(number / len(main) * 100, 2)
        print(f"Percentage of unchanged: {per}%")
    imp.iloc[:, 1:] = impv[:, 1:]                                   
    return imp
    
def improve(sub1, sub2, sub3, sub4, sub5, sub6, res, sort_same, class_same):  
    sub1v = sub1.values
    sub2v = sub2.values
    sub3v = sub3.values
    sub4v = sub4.values
    sub5v = sub5.values 
    sub6v = sub6.values
    imp = res.copy()
    impv = imp.values
    NCLASS = 9
    number = 0

    for i in range (len(res)):
        class_num = 0  
        row = impv[i,1:]
        row1 = sub1v[i,1:]
        row2 = sub2v[i,1:]
        row3 = sub3v[i,1:]
        row4 = sub4v[i,1:]
        row5 = sub5v[i,1:]
        row6 = sub6v[i,1:]
        row_argsort = np.argsort(row)        
        row1_argsort = np.argsort(row1)
        row2_argsort = np.argsort(row2)
        row3_argsort = np.argsort(row3)
        row4_argsort = np.argsort(row4)
        row5_argsort = np.argsort(row5)
        row6_argsort = np.argsort(row6)

        for j in range(NCLASS):
            count = 0
            if (row6_argsort[j] == row1_argsort[j]):
                count += 1
            if (row6_argsort[j] == row2_argsort[j]):
                count += 1
            if (row6_argsort[j] == row3_argsort[j]):
                count += 1
            if (row6_argsort[j] == row4_argsort[j]):
                count += 1
            if (row6_argsort[j] == row5_argsort[j]):
                count += 1
            if (count >= sort_same):
                class_num = class_num + 1
        
        if ((class_num >= class_same) and (row6[row6_argsort[NCLASS - 1]] >= row[row_argsort[NCLASS - 1]])): 
            impv[i, 1:] = row6            
            number = number + 1                        
                    
    imp.iloc[:, 1:] = impv[:, 1:]
    p_number = round(((number / 100000) * 100),2)
    print('=' * 38)
    print(f'Improve Percentage of changes: {p_number} %')
    print('=' * 38)
    return imp

def Reinforce(sub, take, get, ratio, sub1, sub2, sub3):
    reinforce = sub.copy()
    impv = reinforce.values
    sub1v = sub1.values
    sub2v = sub2.values
    sub3v = sub3.values
    count = 0
    for i in range (len(sub)):
        row = impv[i,1:]
        row1 = sub1v[i,1:]
        row2 = sub2v[i,1:]
        row3 = sub3v[i,1:]
        row_sort = np.sort(row)
        row_sort_temp = row_sort
        arg_sort = np.argsort(row)
        arg_sort1 = np.argsort(row1)
        arg_sort2 = np.argsort(row2)
        arg_sort3 = np.argsort(row3)
        
        bre = 0
        for j in range(take):
            if arg_sort[j] != arg_sort1[j] or arg_sort[j] != arg_sort2[j] or arg_sort[j] != arg_sort3[j]:
                bre = 1
        if bre == 1:
            continue
        bre = 0
        for j in range(get):
            if arg_sort[8 - j] != arg_sort1[8 - j] or arg_sort[8 - j] != arg_sort2[8 - j] or arg_sort[8 - j] != arg_sort3[8 - j]:
                bre = 1
        if bre == 1:
            continue

        count += 1
        deprive = 0
        sum1 = 0
        sum2 = 0
        for j in range(take):
            sum1 += row_sort[j]
        for j in range(get):
            sum2 += row_sort[8 - j]
        for j in range(take):
            deprive += row_sort[j] * ratio

        for j in range(9):
            if j < take:
                row_sort[j] -= deprive * row_sort_temp[take - j - 1] / sum1
                if row_sort[j] <= 0:
                    row_sort[j] = 0
            elif j > 8 - get:
                row_sort[j] += deprive * row_sort_temp[j] / sum2
        for j in range(9):
            row[arg_sort[j]] = row_sort[j]
        impv[i,1:] = row

    count_r = round((count / len(sub) * 100), 2)
    print('=' * 30)
    print(f"\nReinforce Percentage of changes: {count_r} %\n")
    print('=' * 30)
    reinforce.iloc[:, 1:] = impv[:, 1:]
    return reinforce


sub0 = pd.read_csv(path + '/result/my sub/my_submission_xgboost.csv')    # 1.74868
sub1 = pd.read_csv(path + '/result/other sub/sub_1.74456.csv')  # TPS JUNE 21 EDA + Models
sub2 = pd.read_csv(path + '/result/other sub/sub_1.74442.csv')  # Residual network for tabular data
sub3 = pd.read_csv(path + '/result/other sub/sub_1.74427.csv')  # 1dCNN + 2dCNN + residual network
sub4 = pd.read_csv(path + '/result/my sub/my_submission_NN.csv')    # 1.74419
sub5 = pd.read_csv(path + '/result/other sub/NN_1.74400.csv')   # NNs+GBTs+Optimization
sub6 = pd.read_csv(path + '/result/ensembling/comparative_1.74372.csv')

sub = ensemble(sub1, sub0, 0.85, 1)
sub = ensemble(sub2, sub1, 0.65, 0)
sub = ensemble(sub3, sub, 0.65, 0)
sub = ensemble(sub4, sub, 0.65, 0)
sub = ensemble(sub5, sub, 0.75, 1)
sub = ensemble(sub6, sub, 0.85, 1)

sub_imp = improve(sub1, sub2, sub3, sub4, sub5, sub6, sub, 5, 7)

sub.to_csv(path + '/result/ensembling/my_submission_ensembling.csv', index=False)

sub_imp.to_csv(path + '/result/ensembling/my_submission_comparative.csv', index=False)

sub_imp.to_csv(path + '/result/my_submission.csv', index=False)