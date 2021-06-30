import sys
path = sys.path[0]
import numpy as np
import pandas as pd

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
sub4 = pd.read_csv(path + '/result/other sub/NN_1.74400.csv')   # NNs+GBTs+Optimization
sub5 = pd.read_csv(path + '/result/ensembling/ensembling_1.74373.csv')
sub6 = pd.read_csv(path + '/result/ensembling/comparative_1.74373.csv')

sub = ensemble(sub1, sub0, 0.85, 1)
sub = ensemble(sub2, sub1, 0.65, 0)
sub = ensemble(sub3, sub, 0.65, 0)
sub = ensemble(sub4, sub, 0.65, 1)
sub = ensemble(sub5, sub, 0.75, 1)
sub = ensemble(sub6, sub, 0.85, 1)

sub_imp = improve(sub1, sub2, sub3, sub4, sub5, sub6, sub, 5, 7)

sub.to_csv(path + '/result/ensembling/my_submission_ensembling.csv', index=False)

sub_imp.to_csv(path + '/result/ensembling/my_submission_comparative.csv', index=False)

sub_imp.to_csv(path + '/result/my_submission.csv', index=False)