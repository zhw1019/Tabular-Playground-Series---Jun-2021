import sys
path = sys.path[0]
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from scipy.optimize import minimize

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations,callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import initializers
from tensorflow.keras.utils import plot_model

from tensorflow.keras.models import Model
from keras_tuner import BayesianOptimization

import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv(path + "/data/train.csv", index_col = 'id')
test = pd.read_csv(path + "/data/test.csv", index_col = 'id')
submission = pd.read_csv(path + "/data/sample_submission.csv")
value_map = {'Class_1': 0, 'Class_2': 1, 'Class_3': 2, 'Class_4': 3, 'Class_5': 4, 'Class_6': 5, 'Class_7': 6, 'Class_8': 7, 'Class_9': 8}
target = train.target
targets = pd.get_dummies(train['target'])
target_optim = train.replace({'target':value_map})['target']

train_knn = np.load(path + "/add_feat_train.npy")
test_knn = np.load(path + "/add_feat_test.npy")

scaler = MinMaxScaler()
train_knn = scaler.fit_transform(train_knn)
test_knn = scaler.transform(test_knn)
knn_list = ['knn_1', 'knn_2', 'knn_3', 'knn_4', 'knn_5', 'knn_6', 'knn_7', 'knn_8', 'knn_9']
train = pd.concat([train.drop('target', axis = 1), 
                   pd.DataFrame(train_knn, columns = knn_list)], axis = 1)
test = pd.concat([test.reset_index().drop('id', axis = 1), 
                   pd.DataFrame(test_knn, columns = knn_list)], axis = 1, ignore_index=False)
train['target'] = target

X_train, X_test, y_train, y_test = train_test_split(
    train.drop('target', axis =1), targets, 
    test_size = 0.2, stratify = targets, random_state = 2021)

def custom_metric(y_true, y_pred):
    y_pred = K.clip(y_pred, 1e-15, 1-1e-15)
    loss = K.mean(cce(y_true, y_pred))
    return loss

cce = tf.keras.losses.CategoricalCrossentropy()

es = tf.keras.callbacks.EarlyStopping(
    monitor='val_custom_metric', min_delta=0.00001, patience=6, verbose=0,
    mode='min', baseline=None, restore_best_weights=True)

plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_custom_metric', factor=0.04, patience=5, verbose=0, mode='min')


def model_builder(hp):

    #--------- List of hyperparameters --------
    # This is example to illustrate how it works. 
    # Feel free to use list of parameters as you want. Be aware .... the more parameters you specify the more resources (time) it will take
    
    emb_units = hp.Int('emb_units', min_value = 7, max_value = 8, step = 1)
    conv1d_filters = hp.Int('conv1d_units', min_value = 1, max_value = 2, step = 1)
    
    dropout_rates = [0.2, 0.4] #[0.2, 0.3, 0.4]
    dropout1 = hp.Choice("drop_out1", values = dropout_rates)
    dropout2 = hp.Choice("drop_out2", values = dropout_rates)
    dropout3 = hp.Float("drop_out3", min_value = 0.0, 
                        max_value = 0.5, 
                        default = 0.25, 
                        step = 0.05,)
    
    lin_nodes = [16, 64] #[16, 32, 64]
    l1_nodes = hp.Choice("l1_units", values = lin_nodes)
    l2_nodes = hp.Choice("l2_units", values = lin_nodes)
    l3_nodes = hp.Choice("l3_units", values = lin_nodes)
    
    learning_rates = hp.Choice("learning_rate", [1e-2]) #[1e-2, 1e-3]
    
    non_linears = ['relu', 'elu'] #['relu', 'selu', 'elu']
    act1 = hp.Choice('dense_act1', values = non_linears, default='relu')
    act2 = hp.Choice('dense_act2', values = non_linears, default='relu')
    act3 = hp.Choice('dense_act3', values = non_linears, default='relu')
    
    ker_inits = ['lecun_normal', 'he_uniform']
    ker_init1 = hp.Choice('kern_init1', values = ker_inits, default = 'lecun_normal')
    ker_init2 = hp.Choice('kern_init2', values = ker_inits, default = 'lecun_normal')
    ker_init3 = hp.Choice('kern_init3', values = ker_inits, default = 'lecun_normal')
    ker_init4 = hp.Choice('kern_init4', values = ker_inits, default = 'lecun_normal')
    
    conv_kernel = hp.Int('conv_kernel', min_value = 5, max_value = 20, step = 1)
    #--------------------------------------
    
    conv_inputs = layers.Input(shape = (75))
    knn_inputs = layers.Input(shape = (9))
        
    #----------- Embedding layers ----------------------
    embed = layers.Embedding (input_dim = 353, 
                              output_dim = emb_units,
                              embeddings_regularizer='l2')(conv_inputs)
    
    #----------- Convolution layers ----------------------
    
    embed = layers.Conv1D(conv_kernel, 1, activation = 'relu')(embed) 
    embed = layers.Flatten()(embed)
    hidden = layers.Dropout(dropout1)(embed)
    
    #----------- Residual blocks layers ----------------------
    hidden = tfa.layers.WeightNormalization(
                layers.Dense(units = l1_nodes,
                             activation = act1, #selu
                             kernel_initializer = ker_init1))(hidden)
   
    
    output = layers.Dropout(dropout2)(layers.Concatenate()([embed, hidden, knn_inputs]))
   
    output = tfa.layers.WeightNormalization(
    layers.Dense(units = l2_nodes,
                 activation = act2,
                 kernel_initializer = ker_init2))(output) 
    

    output = layers.Dropout(dropout3)(layers.Concatenate()([embed, hidden, output]))
    output = tfa.layers.WeightNormalization(
    layers.Dense(units = l3_nodes, 
                 activation = act3, #elu
                 kernel_initializer = ker_init3))(output)
    
    #----------- Final layer -----------------------
    
    conv_outputs = layers.Dense(units = 9, 
                                activation = 'softmax',
                                kernel_initializer = ker_init4)(output)
    
    #----------- Model instantiation  ---------------
    model = Model([conv_inputs, knn_inputs],conv_outputs)
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = keras.optimizers.RMSprop(learning_rates), 
                  metrics = custom_metric)
    
    return model

tuner = BayesianOptimization(
    model_builder,
    objective = "val_loss",
    max_trials = 100,
    executions_per_trial = 2,
    overwrite = True,
    seed = 2021,
)
# tuner.search_space_summary()

Parameter_Search = 0
if (Parameter_Search == 1):
    tuner.search([X_train.iloc[:, :75], X_train.iloc[:, 75:]],  y_train, epochs = 3, 
                 validation_data = ([X_test.iloc[:, :75], X_test.iloc[:, 75:]], y_test))
    tuner.results_summary(num_trials = 5)
    best_hp = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(best_hp)
    model.summary()
    plot_model(model)


net_params = [{'emb_units': 8, 'conv1d_units': 1, 
               'drop_out1': 0.3, 'drop_out2': 0.4, 'drop_out3': 0.2, 
               'l1_units': 16, 'l2_units': 64, 'l3_units': 16, 
               'learning_rate': 0.001, 
               'dense_act1': 'elu', 'dense_act2': 'relu', 'dense_act3': 'relu',
              'kern_init1': 'he_uniform', 'kern_init2': 'he_uniform', 'kern_init3': 'he_uniform', 'kern_init4': 'lecun_normal'},
              {'emb_units': 8, 'conv1d_units': 1, 
               'drop_out1': 0.3, 'drop_out2': 0.4, 'drop_out3': 0.2, 
               'l1_units': 16, 'l2_units': 64, 'l3_units': 16, 
               'learning_rate': 0.001, 
               'dense_act1': 'elu', 'dense_act2': 'relu', 'dense_act3': 'relu',
              'kern_init1': 'he_uniform', 'kern_init2': 'he_uniform', 'kern_init3': 'he_uniform', 'kern_init4': 'lecun_normal'},
              {'emb_units': 7, 'conv1d_units': 1, 
               'drop_out1': 0.3, 'drop_out2': 0.2, 'drop_out3': 0.2, 
               'l1_units': 16, 'l2_units': 128, 'l3_units': 32, 
               'learning_rate': 0.001, 
               'dense_act1': 'elu', 'dense_act2': 'relu', 'dense_act3': 'relu',
              'kern_init1': 'he_uniform', 'kern_init2': 'he_uniform', 'kern_init3': 'he_uniform', 'kern_init4': 'lecun_normal'}
              ]

def model_builder_optimized(net_config):
 
    emb_units = net_config['emb_units']
    conv1d_filters = net_config['conv1d_units']
    
    dropout1 = net_config["drop_out1"]
    dropout2 = net_config["drop_out2"]
    dropout3 = net_config["drop_out3"]

    l1_nodes = net_config["l1_units"]
    l2_nodes = net_config["l2_units"]
    l3_nodes = net_config["l3_units"]
    
    learning_rates = net_config["learning_rate"]

    act1 = net_config['dense_act1']
    act2 = net_config['dense_act2']
    act3 = net_config['dense_act3']
    

    ker_init1 = net_config['kern_init1']
    ker_init2 = net_config['kern_init2']
    ker_init3 = net_config['kern_init3']
    ker_init4 = net_config['kern_init4']
    #--------------------------------------
    
    
    
    conv_inputs = layers.Input(shape = (75))
    knn_inputs = layers.Input(shape = (9))
    #----------- Embedding layers ----------------------
    embed = layers.Embedding (input_dim = 353, 
                              output_dim = emb_units,
                              embeddings_regularizer='l2')(conv_inputs)
    
    #----------- Convolution layers ----------------------
    
    embed = layers.Conv1D(10, conv1d_filters, activation = 'relu')(embed) 
    embed = layers.Flatten()(embed)
    hidden = layers.Dropout(dropout1)(embed)
    
    #----------- Residual blocks layers ----------------------
    hidden = tfa.layers.WeightNormalization(
                layers.Dense(
                units = l1_nodes,
                activation = act1, #selu
                kernel_initializer = ker_init1))(hidden)
    
   
    output = layers.Dropout(dropout2)(layers.Concatenate()([embed, hidden, knn_inputs]))
   
    output = tfa.layers.WeightNormalization(
    layers.Dense(
                units = l2_nodes,
                activation = act2,
                kernel_initializer = ker_init2))(output) 
    

    output = layers.Dropout(dropout3)(layers.Concatenate()([embed, hidden, output]))
    output = tfa.layers.WeightNormalization(
    layers.Dense(
                units = l3_nodes, 
                activation = act3, #elu
                kernel_initializer = ker_init3))(output)
    
    
    #----------- Final layer -----------------------
    
    conv_outputs = layers.Dense(
                units = 9, 
                activation = 'softmax',
                kernel_initializer = ker_init4)(output)
    
    #----------- Model instantiation  ---------------
    model = Model([conv_inputs, knn_inputs], conv_outputs)
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = keras.optimizers.RMSprop(learning_rates), 
                  metrics = custom_metric)
    
    return model

def inter_class_optimizer(weights, a0, a1, a2, a3, a4, a5, a6, a7, a8):
    oof = np.array([weights[0]*a0, 
                    weights[1]*a1, 
                    weights[2]*a2, 
                    weights[3]*a3, 
                    weights[4]*a4, 
                    weights[5]*a5, 
                    weights[6]*a6, 
                    weights[7]*a7, 
                    weights[8]*a8]).transpose()
    
    oof = oof / np.sum(oof, axis=1).reshape(-1, 1)
    return log_loss(y_val, oof)

def pred_fold_optimizer(oof_preds, test_preds):
    
    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    res = minimize(fun=inter_class_optimizer,
                   x0=[1/9 for _ in range(9)],
                   args=tuple(oof_preds[ :, i] for i in range(9)),
                   method= 'Nelder-Mead',
                   options={'maxiter': 300},
                   bounds=[(0.0, 1.0)] * len(oof_class_preds),
                   constraints=cons)

    oof_preds = np.array([res.x[i]*oof_preds[ :, i] for i in range(9)]).transpose()
    oof_preds = oof_preds / np.sum(oof_preds, axis=1).reshape(-1, 1)
    
    test_preds = np.array([res.x[i]*test_preds[:, i] for i in range(9)]).transpose()
    test_preds = test_preds / np.sum(test_preds, axis=1).reshape(-1, 1)

    return res["fun"], test_preds, oof_preds

def inter_model_optimizer(weights):
    final_prediction = 0
    
    for weight, prediction in zip(weights, oof_class_preds):
        final_prediction += weight * prediction
    
    return log_loss(y_val, final_prediction)

def pred_model_optimizer(oof_class_preds, test_class_preds):
    optmized_oof_nn_preds = 0
    optmized_test_nn_preds = 0
    
    starting_values = [1/len(oof_class_preds)] * len(oof_class_preds)
    
    cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
    res = minimize(inter_model_optimizer, 
                   starting_values,
                   method='Nelder-Mead',
                   bounds=[(0.0, 1.0)] * len(oof_class_preds),
                   constraints=cons)
    
    print(f'--- Inter model optimized logloss: {(res["fun"]):.5f} using {res["x"]} weights (sum:{np.sum(res["x"])}) ---\n')

    for weight, prediction in zip(res["x"], oof_class_preds):
        optmized_oof_nn_preds += weight * prediction
    
    for weight, prediction in zip(res["x"], test_class_preds):
        optmized_test_nn_preds += weight * prediction

        
    return optmized_oof_nn_preds, optmized_test_nn_preds

EPOCH = 70
SEED = 2021
N_FOLDS = 20
RANDOM_STATES_NUM = 3
NUM_TOP_MODELS = 2

y_val = []
pred_NN_a = np.zeros((test.shape[0],9))
pred_NN_a_optimized = np.zeros((test.shape[0],9))

tuners = tuner.get_best_hyperparameters(num_trials = NUM_TOP_MODELS)

for rs_n in range(RANDOM_STATES_NUM):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state = (SEED + rs_n))

    oof_NN_a = np.zeros((train.shape[0],9))
    oof_NN_a_optim = np.zeros((train.shape[0],9))
    oof_NN_fold_optimized = np.zeros((train.shape[0],9))
       
    for fold, (tr_idx, ts_idx) in enumerate(skf.split(train, train.target)):

        X_train = train.iloc[:, :75].iloc[tr_idx]
        X_train_knn = train.iloc[:, 75:-1].iloc[tr_idx]
        y_train = targets.iloc[tr_idx]
            
        X_test = train.iloc[:, :75].iloc[ts_idx]
        X_test_knn = train.iloc[:, 75:-1].iloc[ts_idx]
        y_test = targets.iloc[ts_idx]
            
        oof_class_preds = []
        test_class_preds = []

        for n_models in range(NUM_TOP_MODELS):

            K.clear_session()  

            model_conv = model_builder_optimized(net_params[n_models])
            l_rate = net_params[n_models]["learning_rate"]

            model_conv.compile(loss='categorical_crossentropy', 
                               optimizer = keras.optimizers.RMSprop(learning_rate = l_rate), 
                               metrics=custom_metric)

            model_conv.fit([X_train, X_train_knn], y_train,
                           batch_size = 128, epochs = EPOCH,
                           validation_data=([X_test, X_test_knn], y_test),
                           callbacks=[es, plateau],
                           verbose = 0)

            pred_a = model_conv.predict([X_test, X_test_knn]) 
            score_NN_a = log_loss(y_test, pred_a)  
                
            test_NN_preds = model_conv.predict([test.iloc[:, :75], test.iloc[:, 75:]]) 
                
            y_val = target_optim.iloc[ts_idx]
            optim_score, test_preds_optim, oof_preds_optim = pred_fold_optimizer(pred_a, test_NN_preds)
                 
            print(f"  * FOLD {fold + 1} -> MODEL {n_models + 1} -> SCORE: {(score_NN_a):.5f} -> OPTIMIZED SCORE: {optim_score:.5f} (GAIN: {(optim_score-score_NN_a):.5f})")
                
            pred_NN_a += test_preds_optim
            oof_NN_a[ts_idx] += pred_a 
            oof_NN_a_optim[ts_idx] += oof_preds_optim 
            
            oof_class_preds.append(oof_preds_optim)
            test_class_preds.append(test_preds_optim)
        
        oof_NN_fold_optimized[ts_idx], pred_NN_optimized = pred_model_optimizer(oof_class_preds, test_class_preds)
        pred_NN_a_optimized += pred_NN_optimized

    score_a = log_loss(targets, (oof_NN_a / NUM_TOP_MODELS))
    score_o = log_loss(targets, oof_NN_fold_optimized)
    print(f"- FINAL SCORE FOR {n_models + 1} MODELS IN RANDOM STATE {SEED + rs_n}: {score_a:.5f} - OPTIMIZED (inter class and model): {score_o:.5f} (GAIN: {(score_o-score_a):.5f})")

pred_NN_a = pred_NN_a / (N_FOLDS * RANDOM_STATES_NUM * NUM_TOP_MODELS)
pred_NN_a_optimized = pred_NN_a_optimized /  (N_FOLDS * RANDOM_STATES_NUM)

proba = pred_NN_a_optimized
output = pd.DataFrame({'id': submission['id'], 'Class_1': proba[:,0], 'Class_2':proba[:,1], 'Class_3':proba[:,2], 'Class_4':proba[:,3], 'Class_5': proba[:,4], 'Class_6': proba[:,5], 'Class_7': proba[:,6], 'Class_8': proba[:,7],'Class_9': proba[:,8]})
output.to_csv(path + '/result/my sub/my_submission_NN.csv', index=False)