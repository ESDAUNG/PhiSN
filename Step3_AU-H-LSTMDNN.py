import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd,numpy as np, tensorflow as tf
import argparse, importlib, pickle, dataLoader,text2Vect,NLPFeatureExtractor,TokenAndPositionEmbedding, preprocessor

from sklearn.utils import class_weight
from tensorflow.keras.layers.experimental import preprocessing
from pathlib import Path
from tensorflow.keras import layers, metrics, Model, regularizers
from time import time

parser = argparse.ArgumentParser(description="Hybrid-DNN S-LSTM N-DNN Training Main - All")

##############################################################################################################################################
###### Disclaimer : Re-Naming of Files 
###### XX [Dataset] >>> YY [Impelmentation]
###### nlp-dnn >>> n-dnn/N-DNN
###### url-lstm >>> s-lstm/S-LSTM
##############################################################################################################################################

### Initializing Model Directory >>>
parser.add_argument('--model.model_dir', type=str, default="/path-to-main-folder/", metavar="MODELDIR", help="location of model")

### Initializing Datasets >>>
parser.add_argument('--data.cur_directory', type=str, default="path-to-data-folder/",metavar="CurrentDir",help="location of current directory")
parser.add_argument('--data.path_to_df', type=str, default="data.csv",metavar="DataFrameDF",help="DataFrame DF file")

### Initializing arguments >>>
parser.add_argument('--log.kfold', type=int, default=10, metavar="Strafified KFold", help="number of k-fold")
parser.add_argument('--model.max_len', type=int, default=200, metavar="MaxLength", help="model max. number of word")
parser.add_argument('--model.vocab_size', type=int, default=5000, metavar="VocabSize", help="model max. vocabulary size")

### Parsing Arguments >>>
FLAGS = vars(parser.parse_args())
n_fold = FLAGS["log.kfold"]
maxlen = FLAGS["model.max_len"]  # Only consider the first 200 words
vocab_size = FLAGS["model.vocab_size"] # Only consider the top 5000 words to be stored in Dictionary

##############################################################################################################################################
############################################## Directory Construction ########################################################################
##############################################################################################################################################

### Creating Parent Directory >>>
parent_directory = FLAGS["model.model_dir"]
cur_directory = FLAGS["data.cur_directory"]
path = os.path.join(parent_directory, cur_directory)
if not os.path.exists(path):
    os.mkdir(path)
else:
    path = path
print('Created path ', path)

### Creating Result Directory >>>
cur_folder_path = 'Hybrid_DNNLSTMDNN_All_ModelTrainTest/'
result_folder_path = os.path.join(path, cur_folder_path)
if not os.path.exists(result_folder_path):
    os.mkdir(result_folder_path)
else:
    result_folder_path = result_folder_path
print('Created result folder path ', result_folder_path)

tmp_filepath = os.path.join(result_folder_path,'tmp/')

if not os.path.exists(tmp_filepath):
    os.mkdir(tmp_filepath)
else:
    tmp_filepath = tmp_filepath
print('Created tmp_filepath ', tmp_filepath)

checkpoint_filepath = os.path.join(tmp_filepath,'checkpoint/')
checkpoint_filepath4Domain = os.path.join(tmp_filepath,'checkpoint4domain/')
print('os checkpoint ',checkpoint_filepath)

if not os.path.exists(checkpoint_filepath):
    os.mkdir(checkpoint_filepath)
else:
    checkpoint_filepath = checkpoint_filepath
print('Created checkpoint_filepath ', checkpoint_filepath)

if not os.path.exists(checkpoint_filepath4Domain):
    os.mkdir(checkpoint_filepath4Domain)
else:
    checkpoint_filepath4Domain = checkpoint_filepath4Domain
print('Created checkpoint_filepath4Domain ', checkpoint_filepath4Domain)

indx_folder_path = os.path.join(path,'all_models_cv/')
##############################################################################################################################################
############################################## End of Directory Construction #################################################################
##############################################################################################################################################

##############################################################################################################################################
############################################## Functions Definition ##########################################################################
##############################################################################################################################################

### Function: Constructing Model >>>
def Hybrid_DNNLSTMDNN_Model(param_dict):
    ### Hybrid Parameters >>>
    metrics=param_dict['metrics']
    num_layers=param_dict['num_layers']
    num_units=param_dict['num_units']
    #dropout_rate=param_dict['dropout_rate']
    activation=param_dict['activation']
    activation_out=param_dict['activation_out']
    loss=param_dict['loss']
    initializer=param_dict['initializer']
    optimizer=param_dict['optimizer']
    learning_rate=param_dict['learning_rate']

    ### N-DNN Parameters >>>
    nlp_featlen=param_dict['nlp_featlen']
    nlp_dnn_num_layers=param_dict['nlp_dnn_num_layers']
    nlp_dnn_num_units=param_dict['nlp_dnn_num_units']
    nlp_dnn_dropout_rate=param_dict['nlp_dnn_dropout_rate']
    nlp_dnn_activation=param_dict['nlp_dnn_activation']
    nlp_dnn_loss=param_dict['nlp_dnn_loss']
    nlp_dnn_initializer=param_dict['nlp_dnn_initializer']
    #nlp_dnn_optimizer=param_dict['nlp_dnn_optimizer']
    #nlp_dnn_learning_rate=param_dict['nlp_dnn_learning_rate']

    ### S-LSTM Parameters >>>
    url_max_len=param_dict['url_max_len']
    url_vocab_size=param_dict['url_vocab_size']
    url_lstm_embed_dim=param_dict['url_lstm_embed_dim']
    url_lstm_num_layers=param_dict['url_lstm_num_layers']
    url_lstm_num_units=param_dict['url_lstm_num_units']
    url_lstm_dropout_rate=param_dict['url_lstm_dropout_rate']
    url_lstm_activation=param_dict['url_lstm_activation']
    url_lstm_loss=param_dict['url_lstm_loss']
    url_lstm_initializer=param_dict['url_lstm_initializer']
    #url_lstm_optimizer=param_dict['url_lstm_optimizer']
    #url_lstm_learning_rate=param_dict['url_lstm_learning_rate']
    
    ### Initializing N-DNN Model >>>
    if nlp_dnn_num_layers != len([nlp_dnn_num_units]):
        nlp_dnn_num_nodes = [nlp_dnn_num_units][0]
        nlp_dnn_num_units = [i for i in range(nlp_dnn_num_layers) for i in [nlp_dnn_num_nodes]]
    nlp_inputs = layers.Input(shape=(nlp_featlen,), name='NLP_DNN_Input_Layer')
    for i in range(nlp_dnn_num_layers):
        if i == 0:
            nlp_dnn_x = layers.Dense(units=nlp_dnn_num_units[i],input_shape=(nlp_featlen,), activation=nlp_dnn_activation,kernel_initializer=nlp_dnn_initializer, kernel_regularizer=regularizers.L1L2())(nlp_inputs)
            nlp_dnn_x = layers.Dropout(rate=nlp_dnn_dropout_rate)(nlp_dnn_x)
        else:
            nlp_dnn_x = layers.Dense(units=nlp_dnn_num_units[i], activation=nlp_dnn_activation,kernel_initializer=nlp_dnn_initializer, kernel_regularizer=regularizers.L1L2())(nlp_dnn_x)
            nlp_dnn_x = layers.Dropout(rate=nlp_dnn_dropout_rate)(nlp_dnn_x)
    nlp_dnn_outputs = layers.Dense(units=1, name='nlp_dnn_outputs', activation=activation_out,kernel_initializer=nlp_dnn_initializer, kernel_regularizer=regularizers.L1L2())(nlp_dnn_x)

    ### Initializing S-LSTM Model >>>
    if url_lstm_num_layers != len([url_lstm_num_units]):
        url_lstm_num_nodes = [url_lstm_num_units][0]
        url_lstm_num_units = [i for i in range(url_lstm_num_layers) for i in [url_lstm_num_nodes]]
    url_inputs = layers.Input(shape=(url_max_len,), name='word_LSTM_Input_Layer')
    url_lstm_x = TokenAndPositionEmbedding.TokenAndPositionEmbedding(url_max_len, url_vocab_size, url_lstm_embed_dim)(url_inputs)
    url_lstm_x = layers.BatchNormalization()(url_lstm_x)
    for i in range(url_lstm_num_layers):
        url_lstm_x = layers.LSTM(units=url_lstm_num_units[i], return_sequences=True,kernel_initializer=url_lstm_initializer, dropout=url_lstm_dropout_rate)(url_lstm_x)
    url_lstm_x = layers.Flatten()(url_lstm_x)
    url_lstm_x = layers.Dense(units=8, activation=url_lstm_activation)(url_lstm_x)
    url_lstm_outputs = layers.Dense( units=1, name='url_lstm_outputs', activation=activation_out)(url_lstm_x)

    ### Initializing H-DNN Model >>>
    if num_layers != len([num_units]):
        num_nodes = [num_units][0]
        num_units = [i for i in range(num_layers) for i in [num_nodes]]
    
    ### H-DNN: Concatenate Outputs from S-LSTM and N-DNN >>>
    dnn_lstm_dnn_x = layers.Concatenate(axis=1)([nlp_dnn_x,url_lstm_x])
    ### H-DNN: Normalize Layer >>>
    dnn_lstm_dnn_x = layers.BatchNormalization()(dnn_lstm_dnn_x)
    ### H-DNN: Dense Layer >>>
    for i in range(num_layers):
        dnn_lstm_dnn_x = layers.Dense(units=num_units[i], activation=activation,kernel_initializer=initializer, kernel_regularizer=regularizers.L1L2())(dnn_lstm_dnn_x)
    dnn_lstm_dnn_outputs = layers.Dense(units=1, name='dnn_lstm_dnn_outputs', activation=activation_out, kernel_initializer=initializer, kernel_regularizer=regularizers.L1L2())(dnn_lstm_dnn_x)
    
    ###### Initializing H-DNN_S-LSTM_N-DNN Model #################
    dnn_lstm_dnn_model = Model(inputs=[nlp_inputs, url_inputs], outputs=[nlp_dnn_outputs,url_lstm_outputs,dnn_lstm_dnn_outputs])
    ###### H-DNN Optimizer #################
    optimizer_class = getattr(importlib.import_module(
        'tensorflow.keras.optimizers'), optimizer)
    optimizer = optimizer_class(lr=learning_rate)

    ###### Compiling H-DNN #################
    dnn_lstm_dnn_model.compile(loss={'nlp_dnn_outputs': nlp_dnn_loss, 'url_lstm_outputs': url_lstm_loss, 'dnn_lstm_dnn_outputs': loss}, optimizer=optimizer, metrics=metrics)
    dnn_lstm_dnn_model.summary()

    return dnn_lstm_dnn_model

##### Function: Scheduling Learning Rate  #################
def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

##### Function: Calling Model Builder with Parameters  #################
def build_model(param_dict:dict):
    model = Hybrid_DNNLSTMDNN_Model(param_dict)
    return model

##### Function: Training and Testing >>>
def build_eval(X_train_data,X_train_data4Domain,y_train,X_valid_data4url,y_valid_url,X_valid_data4domain,y_valid_domain,X_valid_data4full,y_valid,param_dict,param_dict4domain,result_folder_path,model_id):

    ##### Slicing Concatenated data into S-LSTM and N-DNN data
    X_train_url=tf.convert_to_tensor(X_train_data.iloc[:,:200])
    X_train_domain=tf.convert_to_tensor(X_train_data4Domain.iloc[:,:200])
    X_valid_url=tf.convert_to_tensor(X_valid_data4url.iloc[:,:200])
    X_valid_domain=tf.convert_to_tensor(X_valid_data4domain.iloc[:,:200])
    X_valid_full=tf.convert_to_tensor(X_valid_data4full.iloc[:,:200])

    X_train_nlp=tf.convert_to_tensor(X_train_data.iloc[:,200:])
    X_train_nlp_domain=tf.convert_to_tensor(X_train_data4Domain.iloc[:,200:])
    X_valid_nlp_url=tf.convert_to_tensor(X_valid_data4url.iloc[:,200:])
    X_valid_nlp_domain=tf.convert_to_tensor(X_valid_data4domain.iloc[:,200:])
    X_valid_nlp_full=tf.convert_to_tensor(X_valid_data4full.iloc[:,200:])

    time2train, time2train4domain = 0,0

    ##### Training for URL-Level Model: to be trained with entire URL regardless of the presence of PATH
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='dnn_lstm_dnn_outputs_loss', patience=50,restore_best_weights=True)
    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    tmp_model_id = model_id+'/'
    checkpoint_filepath_model_id = os.path.join(checkpoint_filepath,tmp_model_id)
    print('os checkpoint ',checkpoint_filepath_model_id)

    model_file = result_folder_path+'url-n-dnn_s-lstm_{}.h5'.format(model_id)
    model_file = Path(model_file)
    print('Model file ',model_file)
    if model_file.is_file():
        print('URL Model existed!!!')
        model = tf.keras.models.load_model(result_folder_path+'url-n-dnn_s-lstm_{}.h5'.format(model_id), custom_objects={'TokenAndPositionEmbedding': TokenAndPositionEmbedding.TokenAndPositionEmbedding})
    else:
        if os.path.exists(checkpoint_filepath_model_id):
            latest = tf.train.latest_checkpoint(checkpoint_filepath_model_id)
            print('Latest checkpoint ',latest)
            model = build_model(param_dict)
            # Load the previously saved weights
            model.load_weights(latest)
        else:
            os.mkdir(checkpoint_filepath_model_id)
            # Create a new model instance
            model = build_model(param_dict)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath_model_id+'/checkpoint',save_weights_only=True,monitor='dnn_lstm_dnn_outputs_accuracy',mode='max',save_best_only=True)    
        t1 = time()
        history = model.fit(x=[X_train_nlp,X_train_url],y=y_train,epochs=param_dict['epoch_size'],batch_size=param_dict['batch_size'],class_weight=param_dict['class_weights'], callbacks=[model_checkpoint_callback,callback_lr,stop_early],verbose=1)
        t2 = time()
        time2train= t2-t1
    
        print('Model was saved to {}url-n-dnn_s-lstm_{}.h5'.format(result_folder_path,model_id))
        model.save(result_folder_path+'url-n-dnn_s-lstm_{}.h5'.format(model_id))

        val_acc_per_epoch=history.history['dnn_lstm_dnn_outputs_accuracy']
        best_epoch=val_acc_per_epoch.index(max(val_acc_per_epoch))+1
        model.save(result_folder_path+'url-n-dnn_s-lstm_{}@BestEpoch_{}'.format(model_id,best_epoch),save_format='tf')

    tf.keras.backend.clear_session()

    ##### Training for Domain-Level Model: to be trained with URLs w/o Domain
    stop_early_domain = tf.keras.callbacks.EarlyStopping(monitor='dnn_lstm_dnn_outputs_loss', patience=50,restore_best_weights=True)
    callback_lr_domain = tf.keras.callbacks.LearningRateScheduler(scheduler)
    checkpoint_filepath4Domain_model_id = os.path.join(checkpoint_filepath4Domain,tmp_model_id)
    print('os checkpoint ',checkpoint_filepath4Domain_model_id)
    
    model_file = result_folder_path+'domain-n-dnn_s-lstm_{}.h5'.format(model_id)
    model_file = Path(model_file)
    if model_file.is_file():
        print('Domain Model existed!!!')
        model4Domain = tf.keras.models.load_model(result_folder_path+'domain-n-dnn_s-lstm_{}.h5'.format(model_id), custom_objects={'TokenAndPositionEmbedding': TokenAndPositionEmbedding.TokenAndPositionEmbedding})
    else:
        if os.path.exists(checkpoint_filepath4Domain_model_id):
            latest = tf.train.latest_checkpoint(checkpoint_filepath4Domain_model_id)
            print('Latest checkpoint ',latest)
            model4Domain = build_model(param_dict4domain)
            model4Domain.load_weights(latest)
        else:
            os.mkdir(checkpoint_filepath4Domain_model_id)
            # Create a new model instance
            model4Domain = build_model(param_dict4domain)
        
        model_checkpoint_callback4domain = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath4Domain_model_id+'/checkpoint',save_weights_only=True,monitor='dnn_lstm_dnn_outputs_accuracy',mode='max',save_best_only=True)
        t1 = time()
        history4domain = model4Domain.fit(x=[X_train_nlp_domain,X_train_domain],y=y_train,epochs=param_dict4domain['epoch_size'],batch_size=param_dict4domain['batch_size'],class_weight=param_dict4domain['class_weights'], callbacks=[model_checkpoint_callback4domain,callback_lr_domain,stop_early_domain],verbose=1)
        t2 = time()
        time2train4domain= t2-t1

        print('Model was saved to {}domain-n-dnn_s-lstm_{}.h5'.format(result_folder_path,model_id))
        model4Domain.save(result_folder_path+'domain-n-dnn_s-lstm_{}.h5'.format(model_id))

        val_acc_per_epoch=history4domain.history['dnn_lstm_dnn_outputs_accuracy']
        best_epoch=val_acc_per_epoch.index(max(val_acc_per_epoch))+1
        model4Domain.save(result_folder_path+'domain-n-dnn_s-lstm_{}@BestEpoch_{}'.format(model_id,best_epoch),save_format='tf')

    ##### Testing on URL-Level (URLs w/ Path) H-DNN Model
    tf.keras.backend.clear_session()
    print('Model evaluating URL [URLs w/ Path] from {}url-n-dnn_s-lstm_{}.h5'.format(result_folder_path,model_id))
    t3 = time()
    scores_url = model.evaluate(x=[X_valid_nlp_url,X_valid_url],y=y_valid_url,verbose=1)
    t4 = time()
    time2test_url= t4-t3

    ##### Testing on Domain-Level (URLs w/o Path) H-DNN Model
    tf.keras.backend.clear_session()
    print('Model evaluating Domain [URLs w/o Path] from {}domain-n-dnn_s-lstm_{}.h5'.format(result_folder_path,model_id))
    t3 = time()
    scores_domain = model4Domain.evaluate(x=[X_valid_nlp_domain,X_valid_domain],y=y_valid_domain,verbose=1)
    t4 = time()
    time2test_domain= t4-t3

    ##### Testing on URL-Level (entire URLs) H-DNN Model
    tf.keras.backend.clear_session()
    print('Model evaluating ALL [URLs w/ and w/o Path] from {}url-n-dnn_s-lstm_{}.h5'.format(result_folder_path,model_id))
    t3 = time()
    scores_full_loaded = model.evaluate(x=[X_valid_nlp_full,X_valid_full],y=y_valid,verbose=1)
    t4 = time()
    time2test_full_loaded= t4-t3

    ##### Evaluating on URL-Level (URLs w/ Path) H-DNN Model
    predicted_nlp_dnn,predicted_url_lstm,predicted_dnnlstmdnn=model.predict(x=[X_valid_nlp_url,X_valid_url],verbose=1)
    y_pre_nlp_dnn = [item for sublist in predicted_nlp_dnn for item in sublist]
    y_pre_url_lstm = [item for sublist in predicted_url_lstm for item in sublist]
    y_pre_dnnlstmdnn = [item for sublist in predicted_dnnlstmdnn for item in sublist]
    predicted_df = pd.DataFrame()
    predicted_df['Predicted-N-DNN'] = y_pre_nlp_dnn
    predicted_df['Predicted-S-LSTM'] = y_pre_url_lstm
    predicted_df['Predicted-N-DNN_S-LSTM_Hybrid-DNN'] = y_pre_dnnlstmdnn
    predicted_df['Actual'] = y_valid_url
    predicted_df.to_csv(result_folder_path+'/predicted_url_eval_{}.csv'.format(model_id),encoding='utf-8', index=False, sep=',')

    ##### Evaluating on URL-Level (URLs w/o Path) H-DNN Model
    predicted_nlp_dnn,predicted_url_lstm,predicted_dnnlstmdnn=model4Domain.predict(x=[X_valid_nlp_domain,X_valid_domain],verbose=1)
    y_pre_nlp_dnn = [item for sublist in predicted_nlp_dnn for item in sublist]
    y_pre_url_lstm = [item for sublist in predicted_url_lstm for item in sublist]
    y_pre_dnnlstmdnn = [item for sublist in predicted_dnnlstmdnn for item in sublist]
    predicted_df = pd.DataFrame()
    predicted_df['Predicted-N-DNN'] = y_pre_nlp_dnn
    predicted_df['Predicted-S-LSTM'] = y_pre_url_lstm
    predicted_df['Predicted-N-DNN_S-LSTM_Hybrid-DNN'] = y_pre_dnnlstmdnn
    predicted_df['Actual'] = y_valid_domain
    predicted_df.to_csv(result_folder_path+'/predicted_domain_eval_{}.csv'.format(model_id),encoding='utf-8', index=False, sep=',')

    ##### Evaluating on URL-Level (entire URLs) H-DNN Model
    predicted_nlp_dnn,predicted_url_lstm,predicted_dnnlstmdnn=model.predict(x=[X_valid_nlp_full,X_valid_full],verbose=1)
    y_pre_nlp_dnn = [item for sublist in predicted_nlp_dnn for item in sublist]
    y_pre_url_lstm = [item for sublist in predicted_url_lstm for item in sublist]
    y_pre_dnnlstmdnn = [item for sublist in predicted_dnnlstmdnn for item in sublist]
    predicted_df = pd.DataFrame()
    predicted_df['Predicted-N-DNN'] = y_pre_nlp_dnn
    predicted_df['Predicted-S-LSTM'] = y_pre_url_lstm
    predicted_df['Predicted-N-DNN_S-LSTM_Hybrid-DNN'] = y_pre_dnnlstmdnn
    predicted_df['Actual'] = y_valid
    predicted_df.to_csv(result_folder_path+'/predicted_full_eval_{}.csv'.format(model_id),encoding='utf-8', index=False, sep=',')

    tf.keras.backend.clear_session()

    return time2train,time2train4domain,scores_url,time2test_url,scores_domain,time2test_domain,scores_full_loaded,time2test_full_loaded
    
##### Function:Initializing Default Parameters >>>
##### Parameters need to be updated depending on Dataset
def get_defaults(nlp_featlen=24, nlp_dnn_num_layers=2, nlp_dnn_num_units=64, nlp_dnn_dropout_rate=0.1, nlp_dnn_activation='relu',  nlp_dnn_initializer='TruncatedNormal', nlp_dnn_loss='binary_crossentropy', nlp_dnn_optimizer='Adamax', nlp_dnn_learning_rate=0.001, nlp_dnn_epoch_size=100, nlp_dnn_batch_size=64,
    nlp_lstm_num_layers=2, nlp_lstm_num_units=128, nlp_lstm_dropout_rate=0.3, nlp_lstm_activation='relu',  nlp_lstm_initializer='uniform', nlp_lstm_loss='binary_crossentropy', nlp_lstm_optimizer='Adamax', nlp_lstm_learning_rate=0.01, nlp_lstm_epoch_size=100, nlp_lstm_batch_size=128,
    nlp_cnn_num_layers=4, nlp_cnn_num_units=128, nlp_cnn_dropout_rate=0.1, nlp_cnn_activation='relu',  nlp_cnn_initializer='uniform', nlp_cnn_loss='binary_crossentropy', nlp_cnn_optimizer='Adamax', nlp_cnn_learning_rate=0.001, nlp_cnn_epoch_size=100, nlp_cnn_batch_size=32,
    url_max_len=200, url_vocab_size=5000, 
    url_bilstm_embed_dim=32, url_bilstm_num_layers=8, url_bilstm_num_units=32, url_bilstm_dropout_rate=0.2, url_bilstm_activation='relu', url_bilstm_initializer='glorot_uniform', url_bilstm_loss='binary_crossentropy', url_bilstm_optimizer='Adamax', url_bilstm_learning_rate=0.001, url_bilstm_epoch_size=100, url_bilstm_batch_size=64,
    url_lstm_embed_dim=128, url_lstm_num_layers=2, url_lstm_num_units=32, url_lstm_dropout_rate=0.5, url_lstm_activation='relu', url_lstm_initializer='TruncatedNormal', url_lstm_loss='binary_crossentropy', url_lstm_optimizer='Adamax', url_lstm_learning_rate=0.001, url_lstm_epoch_size=100, url_lstm_batch_size=64,
    url_cnn_embed_dim=128, url_cnn_num_layers=8, url_cnn_num_units=32, url_cnn_dropout_rate=0.3, url_cnn_activation='relu', url_cnn_initializer='random_uniform', url_cnn_loss='binary_crossentropy', url_cnn_optimizer='Adamax', url_cnn_learning_rate=0.001, url_cnn_epoch_size=100, url_cnn_batch_size=64,
    url_transformer_embed_dim=128, url_transformer_num_layers=2, url_transformer_num_units=64, url_transformer_num_heads=8, url_transformer_dropout_rate=0.5, url_transformer_activation='relu', url_transformer_initializer='uniform', url_transformer_loss='binary_crossentropy', url_transformer_optimizer='Adam', url_transformer_learning_rate=0.01, url_transformer_epoch_size=100, url_transformer_batch_size=64,
    num_layers=4, num_units=16, dropout_rate=0.1, activation='relu', initializer='glorot_normal', loss='binary_crossentropy',
    optimizer='Adamax', learning_rate=0.001, epoch_size=100, batch_size=16, activation_out='sigmoid',class_weights={1: 1.0, 0: 1.0},
    metrics=['accuracy', metrics.Precision(name='precision'), metrics.Recall(name='recall'), metrics.FalsePositives(name='false_positives'), metrics.TruePositives(name='true_positives'), metrics.FalseNegatives(name='false_negatives'), metrics.TrueNegatives(name='true_negatives')]):

    defaults = {

        'class_weights': class_weights,
        'metrics': metrics,

        ### Hybrid Parameters ###
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'epoch_size': epoch_size,
        'batch_size': batch_size,
        'activation_out': activation_out,
        'num_layers': num_layers,
        'num_units': num_units,
        'dropout_rate' : dropout_rate,
        'activation': activation,
        'initializer': initializer,
        'loss': loss,

        ### NLP Parameters ###
        'nlp_featlen': nlp_featlen,

        ### NLP DNN Parameters ###
        'nlp_dnn_num_layers': nlp_dnn_num_layers,
        'nlp_dnn_num_units': nlp_dnn_num_units,
        'nlp_dnn_dropout_rate' : nlp_dnn_dropout_rate,
        'nlp_dnn_activation': nlp_dnn_activation,
        'nlp_dnn_initializer': nlp_dnn_initializer,
        'nlp_dnn_loss': nlp_dnn_loss,
        'nlp_dnn_optimizer': nlp_dnn_optimizer,
        'nlp_dnn_learning_rate': nlp_dnn_learning_rate,
        'nlp_dnn_epoch_size': nlp_dnn_epoch_size,
        'nlp_dnn_batch_size': nlp_dnn_batch_size,

        ### NLP LSTM Parameters ###
        'nlp_lstm_num_layers': nlp_lstm_num_layers,
        'nlp_lstm_num_units': nlp_lstm_num_units,
        'nlp_lstm_dropout_rate': nlp_lstm_dropout_rate,
        'nlp_lstm_activation': nlp_lstm_activation,
        'nlp_lstm_initializer': nlp_lstm_initializer,
        'nlp_lstm_loss': nlp_lstm_loss,
        'nlp_lstm_optimizer': nlp_lstm_optimizer,
        'nlp_lstm_learning_rate': nlp_lstm_learning_rate,
        'nlp_lstm_epoch_size': nlp_lstm_epoch_size,
        'nlp_lstm_batch_size': nlp_lstm_batch_size,

        ### NLP CNN Parameters ###
        'nlp_cnn_num_layers': nlp_cnn_num_layers,
        'nlp_cnn_num_units': nlp_cnn_num_units,
        'nlp_cnn_dropout_rate': nlp_cnn_dropout_rate,
        'nlp_cnn_activation': nlp_cnn_activation,
        'nlp_cnn_initializer': nlp_cnn_initializer,
        'nlp_cnn_loss': nlp_cnn_loss,
        'nlp_cnn_optimizer': nlp_cnn_optimizer,
        'nlp_cnn_learning_rate': nlp_cnn_learning_rate,
        'nlp_cnn_epoch_size': nlp_cnn_epoch_size,
        'nlp_cnn_batch_size': nlp_cnn_batch_size,

        ### URL Parameters ###
        'url_max_len': url_max_len,
        'url_vocab_size': url_vocab_size,

        ### URL BiLSTM Parameters ###
        'url_bilstm_embed_dim': url_bilstm_embed_dim,
        'url_bilstm_num_layers': url_bilstm_num_layers,
        'url_bilstm_num_units': url_bilstm_num_units,
        'url_bilstm_dropout_rate': url_bilstm_dropout_rate,
        'url_bilstm_activation': url_bilstm_activation,
        'url_bilstm_initializer': url_bilstm_initializer,
        'url_bilstm_loss': url_bilstm_loss,
        'url_bilstm_optimizer': url_bilstm_optimizer,
        'url_bilstm_learning_rate': url_bilstm_learning_rate,
        'url_bilstm_epoch_size': url_bilstm_epoch_size,
        'url_bilstm_batch_size': url_bilstm_batch_size,

        ### URL LSTM Parameters ###
        'url_lstm_embed_dim': url_lstm_embed_dim,
        'url_lstm_num_layers': url_lstm_num_layers,
        'url_lstm_num_units': url_lstm_num_units,
        'url_lstm_dropout_rate': url_lstm_dropout_rate,
        'url_lstm_activation': url_lstm_activation,
        'url_lstm_initializer': url_lstm_initializer,
        'url_lstm_loss': url_lstm_loss,
        'url_lstm_optimizer': url_lstm_optimizer,
        'url_lstm_learning_rate': url_lstm_learning_rate,
        'url_lstm_epoch_size': url_lstm_epoch_size,
        'url_lstm_batch_size': url_lstm_batch_size,

        ### URL CNN Parameters ###
        'url_cnn_embed_dim': url_cnn_embed_dim,
        'url_cnn_num_layers': url_cnn_num_layers,
        'url_cnn_num_units': url_cnn_num_units,
        'url_cnn_dropout_rate': url_cnn_dropout_rate,
        'url_cnn_activation': url_cnn_activation,
        'url_cnn_initializer': url_cnn_initializer,
        'url_cnn_loss': url_cnn_loss,
        'url_cnn_optimizer': url_cnn_optimizer,
        'url_cnn_learning_rate': url_cnn_learning_rate,
        'url_cnn_epoch_size': url_cnn_epoch_size,
        'url_cnn_batch_size': url_cnn_batch_size,

        ### URL Transformer Parameters ###
        'url_transformer_embed_dim': url_transformer_embed_dim,
        'url_transformer_num_layers': url_transformer_num_layers,
        'url_transformer_num_units': url_transformer_num_units,
        'url_transformer_num_heads': url_transformer_num_heads,
        'url_transformer_dropout_rate': url_transformer_dropout_rate,
        'url_transformer_activation': url_transformer_activation,
        'url_transformer_initializer': url_transformer_initializer,
        'url_transformer_loss': url_transformer_loss,
        'url_transformer_optimizer': url_transformer_optimizer,
        'url_transformer_learning_rate': url_transformer_learning_rate,
        'url_transformer_epoch_size': url_transformer_epoch_size,
        'url_transformer_batch_size': url_transformer_batch_size,

    }

    return defaults

##### Function: Writing Evaluation >>> 
def writeResult2File(score_lst,result_folder_path,mode):
    loss, accuracy, precision, recall,  false_positive, true_positive, false_negative,  true_negative, f_measure = [], [], [], [], [], [], [], [], []
    nlp_loss, nlp_accuracy, nlp_precision, nlp_recall,  nlp_false_positive, nlp_true_positive, nlp_false_negative,  nlp_true_negative, nlp_f_measure = [], [], [], [], [], [], [], [], []
    url_loss, url_accuracy, url_precision, url_recall,  url_false_positive, url_true_positive, url_false_negative,  url_true_negative, url_f_measure = [], [], [], [], [], [], [], [], []

    result_df = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F Measure', 'FP', 'TP', 'FN', 'TN', 'Loss'])
    for i in range(n_fold):
        nlp_loss.append(score_lst[i, 1])
        url_loss.append(score_lst[i, 2])
        loss.append(score_lst[i, 3])

        nlp_accuracy.append(score_lst[i, 4])
        nlp_precision.append(score_lst[i, 5])
        nlp_recall.append(score_lst[i, 6])
        nlp_false_positive.append(score_lst[i, 7])
        nlp_true_positive.append(score_lst[i, 8])
        nlp_false_negative.append(score_lst[i, 9])
        nlp_true_negative.append(score_lst[i, 10])
        try:
            f_score = (2*score_lst[i, 5]*score_lst[i, 6]) / (score_lst[i, 5]+score_lst[i, 6])
        except:
            f_score = 0
        nlp_f_measure.append(f_score)

        url_accuracy.append(score_lst[i, 11])
        url_precision.append(score_lst[i, 12])
        url_recall.append(score_lst[i, 13])
        url_false_positive.append(score_lst[i, 14])
        url_true_positive.append(score_lst[i, 15])
        url_false_negative.append(score_lst[i, 16])
        url_true_negative.append(score_lst[i, 17])
        try:
            f_score = (2*score_lst[i, 12]*score_lst[i, 13]) / (score_lst[i, 12]+score_lst[i, 13])
        except:
            f_score = 0
        url_f_measure.append(f_score)

        accuracy.append(score_lst[i, 18])
        precision.append(score_lst[i, 19])
        recall.append(score_lst[i, 20])
        false_positive.append(score_lst[i, 21])
        true_positive.append(score_lst[i, 22])
        false_negative.append(score_lst[i, 23])
        true_negative.append(score_lst[i, 24])
        try:
            f_score = (2*score_lst[i, 19]*score_lst[i, 20]) / (score_lst[i, 19]+score_lst[i, 20])
        except:
            f_score = 0
        f_measure.append(f_score)

    ###  Hybrid  ##############################
    mean_loss = np.mean(np.array(loss))
    loss.append(np.std(np.array(loss)))
    loss.append(mean_loss)
    mean_accuracy = np.mean(np.array(accuracy))
    accuracy.append(np.std(np.array(accuracy)))
    accuracy.append(mean_accuracy)
    mean_precision = np.mean(np.array(precision))
    precision.append(np.std(np.array(precision)))
    precision.append(mean_precision)
    mean_recall = np.mean(np.array(recall))
    recall.append(np.std(np.array(recall)))
    recall.append(mean_recall)
    mean_false_positive = np.mean(np.array(false_positive))
    false_positive.append(np.std(np.array(false_positive)))
    false_positive.append(mean_false_positive)
    mean_true_positive = np.mean(np.array(true_positive))
    true_positive.append(np.std(np.array(true_positive)))
    true_positive.append(mean_true_positive)
    mean_false_negative = np.mean(np.array(false_negative))
    false_negative.append(np.std(np.array(false_negative)))
    false_negative.append(mean_false_negative)
    mean_true_negative = np.mean(np.array(true_negative))
    true_negative.append(np.std(np.array(true_negative)))
    true_negative.append(mean_true_negative)
    mean_f_measure = np.mean(np.array(f_measure))
    f_measure.append(np.std(np.array(f_measure)))
    f_measure.append(mean_f_measure)
    print(' F measure : {}'.format(mean_f_measure))

    result_df['Accuracy'] = accuracy
    result_df['Precision'] = precision
    result_df['Recall'] = recall
    result_df['F Measure'] = f_measure
    result_df['FP'] = false_positive
    result_df['TP'] = true_positive
    result_df['FN'] = false_negative
    result_df['TN'] = true_negative
    result_df['Loss'] = loss
    result_df.to_csv('{}hybrid_mean_evaluation_{}.csv'.format(result_folder_path,mode), sep=',', encoding='utf-8', index=False)

    ###  NLP  ##############################
    nlp_mean_loss = np.mean(np.array(nlp_loss))
    nlp_loss.append(np.std(np.array(nlp_loss)))
    nlp_loss.append(nlp_mean_loss)
    nlp_mean_accuracy = np.mean(np.array(nlp_accuracy))
    nlp_accuracy.append(np.std(np.array(nlp_accuracy)))
    nlp_accuracy.append(nlp_mean_accuracy)
    nlp_mean_precision = np.mean(np.array(nlp_precision))
    nlp_precision.append(np.std(np.array(nlp_precision)))
    nlp_precision.append(nlp_mean_precision)
    nlp_mean_recall = np.mean(np.array(nlp_recall))
    nlp_recall.append(np.std(np.array(nlp_recall)))
    nlp_recall.append(nlp_mean_recall)
    nlp_mean_false_positive = np.mean(np.array(nlp_false_positive))
    nlp_false_positive.append(np.std(np.array(nlp_false_positive)))
    nlp_false_positive.append(nlp_mean_false_positive)
    nlp_mean_true_positive = np.mean(np.array(nlp_true_positive))
    nlp_true_positive.append(np.std(np.array(nlp_true_positive)))
    nlp_true_positive.append(nlp_mean_true_positive)
    nlp_mean_false_negative = np.mean(np.array(nlp_false_negative))
    nlp_false_negative.append(np.std(np.array(nlp_false_negative)))
    nlp_false_negative.append(nlp_mean_false_negative)
    nlp_mean_true_negative = np.mean(np.array(nlp_true_negative))
    nlp_true_negative.append(np.std(np.array(nlp_true_negative)))
    nlp_true_negative.append(nlp_mean_true_negative)
    nlp_mean_f_measure = np.mean(np.array(nlp_f_measure))
    nlp_f_measure.append(np.std(np.array(nlp_f_measure)))
    nlp_f_measure.append(nlp_mean_f_measure)
    print(' F measure : {}'.format(nlp_mean_f_measure))

    result_df = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F Measure', 'FP', 'TP', 'FN', 'TN', 'Loss'])
    result_df['Accuracy'] = nlp_accuracy
    result_df['Precision'] = nlp_precision
    result_df['Recall'] = nlp_recall
    result_df['F Measure'] = nlp_f_measure
    result_df['FP'] = nlp_false_positive
    result_df['TP'] = nlp_true_positive
    result_df['FN'] = nlp_false_negative
    result_df['TN'] = nlp_true_negative
    result_df['Loss'] = nlp_loss
    result_df.to_csv('{}nlp_mean_evaluation_{}.csv'.format(result_folder_path,mode), sep=',', encoding='utf-8', index=False)

    ###  URL  ##############################
    url_mean_loss = np.mean(np.array(url_loss))
    url_loss.append(np.std(np.array(url_loss)))
    url_loss.append(url_mean_loss)
    url_mean_accuracy = np.mean(np.array(url_accuracy))
    url_accuracy.append(np.std(np.array(url_accuracy)))
    url_accuracy.append(url_mean_accuracy)
    url_mean_precision = np.mean(np.array(url_precision))
    url_precision.append(np.std(np.array(url_precision)))
    url_precision.append(url_mean_precision)
    url_mean_recall = np.mean(np.array(url_recall))
    url_recall.append(np.std(np.array(url_recall)))
    url_recall.append(url_mean_recall)
    url_mean_false_positive = np.mean(np.array(url_false_positive))
    url_false_positive.append(np.std(np.array(url_false_positive)))
    url_false_positive.append(url_mean_false_positive)
    url_mean_true_positive = np.mean(np.array(url_true_positive))
    url_true_positive.append(np.std(np.array(url_true_positive)))
    url_true_positive.append(url_mean_true_positive)
    url_mean_false_negative = np.mean(np.array(url_false_negative))
    url_false_negative.append(np.std(np.array(url_false_negative)))
    url_false_negative.append(url_mean_false_negative)
    url_mean_true_negative = np.mean(np.array(url_true_negative))
    url_true_negative.append(np.std(np.array(url_true_negative)))
    url_true_negative.append(url_mean_true_negative)
    url_mean_f_measure = np.mean(np.array(url_f_measure))
    url_f_measure.append(np.std(np.array(url_f_measure)))
    url_f_measure.append(url_mean_f_measure)
    print(' F measure : {}'.format(url_mean_f_measure))

    result_df = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F Measure', 'FP', 'TP', 'FN', 'TN', 'Loss'])
    result_df['Accuracy'] = url_accuracy
    result_df['Precision'] = url_precision
    result_df['Recall'] = url_recall
    result_df['F Measure'] = url_f_measure
    result_df['FP'] = url_false_positive
    result_df['TP'] = url_true_positive
    result_df['FN'] = url_false_negative
    result_df['TN'] = url_true_negative
    result_df['Loss'] = url_loss
    result_df.to_csv('{}url_mean_evaluation_{}.csv'.format(result_folder_path,mode), sep=',', encoding='utf-8', index=False)

##############################################################################################################################################
############################################## Endo of Functions Definition ##################################################################
##############################################################################################################################################

### Loading Dataset >>>
############################################## Loading DataFrame: df ##########################################################################
print('----------------------------------- Loading CSV Dataset -----------------------------------------')
path_to_df = path+FLAGS["data.path_to_df"]
df = pd.read_csv(path_to_df, encoding="utf-8", sep=',')
print('Loaded data size : {}'.format(len(df)))

##### Loading Pre-Tokenized Data : Word-level URLs and Domains >>>
############################################## Loading Tokenized Data File ####################################################################
X_url_word, X_domain_word = dataLoader.load_tokenized_data(path, "all")

##################### Loading Cross-Validated Data ############################################################################################
f_train_list = []
f_valid_list =[]
for path_file, Directory, files in os.walk(indx_folder_path):  
    for file in files:
        if (file.startswith("train_idx") and file.endswith(".csv")):
            f_train_list.append(file)
            f_train_list.sort()
        
        if (file.startswith("test_idx") and file.endswith(".csv")):
            f_valid_list.append(file) 
            f_valid_list.sort()
    break

##### Initializing Evaluation Mertics' Variables >>>
loss, accuracy, precision, recall,  false_positive, true_positive, false_negative,  true_negative, f_measure = [], [], [], [], [], [], [], [], []
nlp_loss, nlp_accuracy, nlp_precision, nlp_recall,  nlp_false_positive, nlp_true_positive, nlp_false_negative,  nlp_true_negative, nlp_f_measure = [], [], [], [], [], [], [], [], []
url_loss, url_accuracy, url_precision, url_recall,  url_false_positive, url_true_positive, url_false_negative,  url_true_negative, url_f_measure = [], [], [], [], [], [], [], [], []
result_df = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F Measure', 'FP', 'TP', 'FN', 'TN', 'Loss'])
t2train_total, t2train_total4domain = [], []
score_lst_url,t2test_total_url  = [],[]
score_lst_domain,t2test_total_domain = [],[]
score_lst_full,t2test_total_full = [],[]
_time_url,_time_nlp = [],[]
_time_url_domain,_time_nlp_domain = [],[]
_time_full, _time_nlp_full = [],[]

##### Main Task : Preprocessing, Model Training and Evaluation on Cross-Validated Data >>>
for indx in range(n_fold):
    char = str(f_train_list[indx])[-5]
    if char.endswith("0"):
        char = str('10')
    
    ##### Retrieving cross-validated Indexes >>>
    train_idx_df = pd.DataFrame(pd.read_csv(indx_folder_path+f_train_list[indx], sep=',', encoding='utf-8'))
    valid_idx_df = pd.DataFrame(pd.read_csv(indx_folder_path+f_valid_list[indx], sep=',', encoding='utf-8'))
    train_idx=train_idx_df['index'].to_list()
    valid_idx=valid_idx_df['index'].to_list()

    ##### Splitting URLs w/ or w/o Path >>>
    X_valid_domain, X_valid_url, X_valid_full  = preprocessor.splitByPath(df,valid_idx_df)

    ##### Indexing cross-validated data >>>
    X_train = pd.DataFrame(df.iloc[train_idx],index=None,columns=['URLs','Labels'])
    X_valid = pd.DataFrame(df.iloc[valid_idx],index=None,columns=['URLs','Labels'])

    ##### Assigning cross-validated training data: (1) entire URLs, and (2) URLs w/o Path >>>
    X_train['URL_Word'] = pd.DataFrame(X_url_word,index=None).iloc[train_idx]
    X_train['Domain_Word'] = pd.DataFrame(X_domain_word,index=None).iloc[train_idx]

    ##### Assigning cross-validated testing data: (1) entire URLs w/>>>
    X_valid['URL_Word'] = pd.DataFrame(X_url_word,index=None).iloc[valid_idx]

    ##### Assigning cross-validated testing data: (2) URLs w/o Path >>>
    tmp_index = pd.DataFrame(X_domain_word,index=None).iloc[list(X_valid_domain['index'].values)]
    X_valid_domain['Domain_Word'] = tmp_index.values

    ##### Assigning cross-validated testing data: (3) URLs w/ Path >>>
    tmp_index = pd.DataFrame(X_url_word,index=None).iloc[list(X_valid_url['index'].values)]
    X_valid_url['URL_Word'] = tmp_index.values

    #tmp_index = pd.DataFrame(X_url_word,index=None).iloc[list(X_valid_full['index'].values)]
    #X_valid_full['URL_Word'] = tmp_index.values

    ##### Saving cross-validated data >>>
    X_train.to_csv(result_folder_path+'X_train_{}.csv'.format(char),sep=',',index=False,encoding='utf-8')
    X_valid.to_csv(result_folder_path+'X_valid_full_{}.csv'.format(char),sep=',',index=False,encoding='utf-8')
    X_valid_url.to_csv(result_folder_path+'X_valid_url_{}.csv'.format(char),sep=',',index=False,encoding='utf-8')
    X_valid_domain.to_csv(result_folder_path+'X_valid_domain_{}.csv'.format(char),sep=',',index=False,encoding='utf-8')

    ##### Converting Labels' datatype to array >>>  
    y_train = np.array(X_train['Labels'])
    y_valid = np.array(X_valid['Labels'])
    y_valid_url = np.array(X_valid_url['Labels'])
    y_valid_domain = np.array(X_valid_domain['Labels'])

    ###########################################################################################################################################
    ##### Preprocessing for URL Features >>>  
    ###########################################################################################################################################
    
    print('###########################################################################################################################################')
    print('Preprocessing for URL Features >>>  \n')    
    print('###########################################################################################################################################')
    
    ##### Adapting Training Data  >>>  
    print('Adapting Training Data .....')
    data2adap = X_train['URL_Word'].to_list()
    text_dataset = tf.data.Dataset.from_tensor_slices(data2adap)
    vectorized_layer = text2Vect.adap_data(text_dataset, maxlen, vocab_size)
    print('Adapting Training Data .....Finished')

    ##### Constructing Vocabulary file with Training Data >>>  
    print('Creating Vocabulary File.....')
    vocab = vectorized_layer.get_vocabulary()
    vocab_ = ''
    for word in range(len(vocab)):
        vocab_ += vocab[word]+"\t"+str(word)+'\n'
    vocab_file = open('{}orgurl_vocab{}.txt'.format(result_folder_path,char), mode='w', encoding="utf-8")
    vocab_file.writelines(vocab_)
    vocab_file.close()
    print('Creating Vocabulary File.....Finished')

    ##### Pickling the config and weights of Vectorizer >>>
    pickle.dump({'config': vectorized_layer.get_config(), 'weights': vectorized_layer.get_weights()}, open("{}textVectorizer_url{}.pkl".format(result_folder_path,char), "wb"))
    
    ##### Unpickle and use 
    print('Loading Vectorized Layer .....')
    from_disk = pickle.load(open("{}textVectorizer_url{}.pkl".format(result_folder_path,char), "rb"))
    loaded_vectorized_layer = preprocessing.TextVectorization.from_config(from_disk['config'])
    loaded_vectorized_layer.set_weights(from_disk['weights'])

    ##### Loading and Vectorizing Training Data - URL >>>  
    print('Vectorizing Training Data - URL .....')
    X_train_vectorized = text2Vect.vectorize_data(data=X_train['URL_Word'].to_list(), vectorize_layer=loaded_vectorized_layer)
    print('Vectorizing Training Data - URL .....Finished')

    ### Loading and Vectorizing Training Data - Domain >>>  
    print('Vectorizing Training Data - Domain .....')
    X_train_vectorized4Domain = text2Vect.vectorize_data(data=X_train['Domain_Word'].to_list(), vectorize_layer=loaded_vectorized_layer)
    print('Vectorizing Training Data - Domain .....Finished')

    #### Vectorizing Validation Data - Domain >>> 
    print('Vectorizing Validation Data : Domain .....')
    start_time_url4domain = time()
    X_valid_vectorized_domain = text2Vect.vectorize_data(data=X_valid_domain['Domain_Word'].to_list(), vectorize_layer=loaded_vectorized_layer)
    stop_time_url4domain = time()
    _time_url_domain.append(stop_time_url4domain-start_time_url4domain)
    print('Vectorizing Validation Data : Domain .....Finished')

    #### Vectorizing Validation Data - URL w/ Path >>> 
    print('Vectorizing Validation Data : URL .....')
    start_time_url4url = time()
    X_valid_vectorized_url = text2Vect.vectorize_data(data=X_valid_url['URL_Word'].to_list(), vectorize_layer=loaded_vectorized_layer)
    stop_time_url4url = time()
    _time_url.append(stop_time_url4url-start_time_url4url)
    print('Vectorizing Validation Data : URL .....Finished')

    #### Vectorizing Validation Data - entire URLs >>> 
    print('Vectorizing Validation Data : FULL .....')
    start_time_url4full = time()
    X_valid_vectorized_full = text2Vect.vectorize_data(data=X_valid['URL_Word'].to_list(), vectorize_layer=loaded_vectorized_layer)
    stop_time_url4full = time()
    _time_full.append(stop_time_url4full-start_time_url4full)
    print('Vectorizing Validation Data : FULL .....Finished')

    #### Computing Class Weights  >>> 
    print('Assigning Weights from KERAS  >>> ')
    class_weights=class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(list(y_train)),y=list(y_train))
    class_weights = dict(enumerate(class_weights))
    print('Assigned Weights from KERAS : {}'.format(class_weights))

    ###########################################################################################################################################
    ### Preprocessing for NLP Features ######################################################################################################## 
    ###########################################################################################################################################
    
    print('###########################################################################################################################################')
    print('Preprocessing for NLP Features >>>  \n')
    print('###########################################################################################################################################')

    ##### Initializing Variables  >>>      
    X_train_4nlp = pd.DataFrame(columns=['URLs', 'Labels'])
    X_valid_4nlp = pd.DataFrame(columns=['URLs', 'Labels'])
    X_valid_4nlp_url = pd.DataFrame(columns=['URLs', 'Labels'])
    X_valid_4nlp_domain = pd.DataFrame(columns=['URLs', 'Labels'])

    ##### Assigning Cross-validated Data for (1) entire URLs, (2) URLs w/ Path and (3) URLs w/o Path>>>  
    X_train_4nlp['URLs'] = X_train['URLs'].to_list()
    X_valid_4nlp['URLs'] = X_valid['URLs'].to_list()
    X_valid_4nlp_url['URLs'] = X_valid_url['URLs'].to_list()
    X_valid_4nlp_domain['URLs'] = X_valid_domain['URLs'].to_list()

    ##### Converting & Assigning Labels' datatype to array to DF >>>  
    X_train_4nlp['Labels'] = np.array(X_train['Labels'].to_list())
    X_valid_4nlp['Labels'] = np.array(X_valid['Labels'].to_list())
    X_valid_4nlp_url['Labels'] = np.array(X_valid_url['Labels'].to_list())
    X_valid_4nlp_domain['Labels'] = np.array(X_valid_domain['Labels'].to_list())

    ##### Extracting NLP Features : Training [URL] >>> 
    print('Extracting Training Data : NLP Features  >>> ')
    X_train_nlp_features = NLPFeatureExtractor.extract_NLP_feat(X_train_4nlp,FLAGS["model.model_dir"]+'data/')
    print('Extracting Training Data : NLP Features  >>> Finished\n')

    ### Extracting NLP Features : Training [Domain] >>> 
    print('Extracting Training Data - Domain: NLP Features  >>> ')
    X_train_nlp_features4Domain = NLPFeatureExtractor.extract_Domain_NLP_feat(X_train_4nlp,FLAGS["model.model_dir"]+'data/')
    print('Extracting Training Data - Domain: NLP Features  >>> Finished\n')

    ### Extracting NLP Features : Testing [Domain] >>> 
    print('Extracting Validation Data : NLP Features4Domain >>> ')
    start_time_nlp4domain = time()
    X_valid_nlp_features4Domain = NLPFeatureExtractor.extract_Domain_NLP_feat(X_valid_4nlp_domain,FLAGS["model.model_dir"]+'data/')
    stop_time_nlp4domain = time()
    _time_nlp_domain.append(stop_time_nlp4domain-start_time_nlp4domain)
    print('Extracting Validation Data : NLP Features4Domain  >>> Finished\n')

    ##### Extracting NLP Features : Testing [URLw/Path] >>> 
    print('Extracting Validation Data : NLP Features4URL  >>> ')
    start_time_nlp4url = time()
    X_valid_nlp_features4URL = NLPFeatureExtractor.extract_NLP_feat(X_valid_4nlp_url,FLAGS["model.model_dir"]+'data/')
    stop_time_nlp4url = time()
    _time_nlp.append(stop_time_nlp4url-start_time_nlp4url)
    print('Extracting Validation Data : NLP Features4URL  >>> Finished\n')

    ##### Extracting NLP Features : Testing [entire URLs] >>> 
    print('Extracting Validation Data : NLP Features4URLFULL  >>> ')
    start_time_nlp4full = time()
    X_valid_nlp_features4FULL = NLPFeatureExtractor.extract_NLP_feat(X_valid_4nlp,FLAGS["model.model_dir"]+'data/')
    stop_time_nlp4full = time()
    _time_nlp_full.append(stop_time_nlp4full-start_time_nlp4full)
    print('Extracting Validation Data : NLP Features4URL  >>> Finished\n')

    ##### Saving NLP Features >>> 
    X_train_nlp_features.to_csv(result_folder_path+'X_train_nlp_url_'+char+'.csv', sep=',', encoding="utf-8", index=None)
    X_train_nlp_features4Domain.to_csv(result_folder_path+'X_train_nlp_domain_'+char+'.csv', sep=',', encoding="utf-8", index=None)
    X_valid_nlp_features4URL.to_csv(result_folder_path+'X_valid_nlp_url_'+char+'.csv', sep=',', encoding="utf-8", index=None)
    X_valid_nlp_features4Domain.to_csv(result_folder_path+'X_valid_nlp_domain_'+char+'.csv', sep=',', encoding="utf-8", index=None)
    X_valid_nlp_features4FULL.to_csv(result_folder_path+'X_valid_nlp_full_'+char+'.csv', sep=',', encoding="utf-8", index=None)
    
    ##### Initializing Best-Parameters >>> 
    ### Need to be updated according to Dataset
    param_dict_defaults = get_defaults(nlp_dnn_num_layers=2, nlp_dnn_num_units=64, nlp_dnn_dropout_rate=0.1, nlp_dnn_activation='relu',  nlp_dnn_initializer='random_uniform', nlp_dnn_loss='binary_crossentropy', nlp_dnn_optimizer='Adamax', nlp_dnn_learning_rate=0.001, nlp_dnn_epoch_size=100, nlp_dnn_batch_size=64,
                            url_lstm_embed_dim=128, url_lstm_num_layers=2, url_lstm_num_units=32, url_lstm_dropout_rate=0.5, url_lstm_activation='relu', url_lstm_initializer='TruncatedNormal', url_lstm_loss='binary_crossentropy', url_lstm_optimizer='Adamax', url_lstm_learning_rate=0.001, url_lstm_epoch_size=100, url_lstm_batch_size=64,nlp_featlen=X_train_nlp_features.shape[-1],class_weights=class_weights)
    param_dict_defaults4domain = get_defaults(nlp_dnn_num_layers=2, nlp_dnn_num_units=64, nlp_dnn_dropout_rate=0.1, nlp_dnn_activation='relu',  nlp_dnn_initializer='TruncatedNormal', nlp_dnn_loss='binary_crossentropy', nlp_dnn_optimizer='Adamax', nlp_dnn_learning_rate=0.001, nlp_dnn_epoch_size=100, nlp_dnn_batch_size=64,
                            url_lstm_embed_dim=32, url_lstm_num_layers=8, url_lstm_num_units=32, url_lstm_dropout_rate=0.2, url_lstm_activation='relu', url_lstm_initializer='glorot_uniform', url_lstm_loss='binary_crossentropy', url_lstm_optimizer='Adamax', url_lstm_learning_rate=0.001, url_lstm_epoch_size=100, url_lstm_batch_size=64,nlp_featlen=X_train_nlp_features4Domain.shape[-1],class_weights=class_weights)

    ##### Assigning Traning Data with Concatenated Features >>> 
    X_train_data_df=pd.DataFrame(np.concatenate((X_train_vectorized,X_train_nlp_features.values),axis=1),index=None,dtype=float)
    X_train_data4Domain_df=pd.DataFrame(np.concatenate((X_train_vectorized4Domain,X_train_nlp_features4Domain.values),axis=1),index=None,dtype=float)
    
    ##### Assigning Testing Data with Concatenated Features >>> 
    X_valid_data4url_df=pd.DataFrame(np.concatenate((X_valid_vectorized_url,X_valid_nlp_features4URL.values),axis=1),index=None,dtype=float)
    X_valid_data4domain_df=pd.DataFrame(np.concatenate((X_valid_vectorized_domain,X_valid_nlp_features4Domain.values),axis=1),index=None,dtype=float)
    X_valid_data4full_df = pd.DataFrame(np.concatenate((X_valid_vectorized_full,X_valid_nlp_features4FULL.values),axis=1),index=None,dtype=float)

    ##### Training and Evaluating >>> 
    time2train,time2train4domain,scores_url,time2test_url,scores_domain,time2test_domain,scores_full_loaded,time2test_full_loaded = build_eval(X_train_data_df,X_train_data4Domain_df,y_train,X_valid_data4url_df,y_valid_url,X_valid_data4domain_df,y_valid_domain,X_valid_data4full_df,y_valid,param_dict=param_dict_defaults,param_dict4domain=param_dict_defaults4domain,result_folder_path=result_folder_path,model_id=char)

    ##### Assigning Evaluated Data >>>     
    t2train_total.append(time2train)
    t2train_total4domain.append(time2train4domain)
    t2test_total_url.append(time2test_url)
    t2test_total_domain.append(time2test_domain)
    score_lst_url.append(scores_url)
    score_lst_domain.append(scores_domain)
    score_lst_full.append(scores_full_loaded)
    t2test_total_full.append(time2test_full_loaded)

##### Assigning Evaluated Scores >>>  
str2Write = ''
t2train = np.average(np.array(t2train_total))/60
t2train4domain = np.average(np.array(t2train_total4domain))/60
t_url_url = np.average(np.array(_time_url))/60
t_url_domain = np.average(np.array(_time_url_domain))/60
t_url_full = np.average(np.array(_time_full))/60
t_nlp_url = np.average(np.array(_time_nlp))/60
t_nlp_domain = np.average(np.array(_time_nlp_domain))/60
t_nlp_full = np.average(np.array(_time_nlp_full))/60
t2test_url = np.average(np.array(t2test_total_url))/60
t2test_domain = np.average(np.array(t2test_total_domain))/60
t2test_full = np.average(np.array(t2test_total_full))/60

##### Printing Evaluated Scores >>> 
print("Finished cross-validation. Took {:.3f} minutes to Train.".format(t2train))
print("Finished cross-validation. Took {:.3f} minutes to Train.".format(t2train4domain))
print("Finished cross-validation. Took {:.3f} minutes to Test. [Segmentation-Preprocessing-URL]".format(t_url_url))
print("Finished cross-validation. Took {:.3f} minutes to Test. [Segmentation-Preprocessing-Domain]".format(t_url_domain))
print("Finished cross-validation. Took {:.3f} minutes to Test. [Segmentation-Preprocessing-FULL]".format(t_url_full))
print("Finished cross-validation. Took {:.3f} minutes to Test.. [NLP-Preprocessing-URL]".format(t_nlp_url))
print("Finished cross-validation. Took {:.3f} minutes to Test.. [NLP-Preprocessing-Domain]".format(t_nlp_domain))
print("Finished cross-validation. Took {:.3f} minutes to Test.. [NLP-Preprocessing-FULL]".format(t_nlp_full))
print("Finished cross-validation. Took {:.3f} minutes to Test-URL.".format(t2test_url))
print("Finished cross-validation. Took {:.3f} minutes to Test-Domain.".format(t2test_domain))
print("Finished cross-validation. Took {:.3f} minutes to Test-FULL.".format(t2test_full))

##### Assigning Evaluated Scores for Writing to a file >>> 
str2Write = str2Write + "Finished cross-validation. Took "+ str(t2train) +" minutes to Train.\n"
str2Write = str2Write + "Finished cross-validation. Took "+ str(t2train4domain) +" minutes to Train - Domain.\n"
str2Write = str2Write + "Finished cross-validation. Took "+ str(t_url_url) +" minutes to Test. [Segmentation-Preprocessing-URL]\n"
str2Write = str2Write + "Finished cross-validation. Took "+ str(t_url_domain) +" minutes to Test. [Segmentation-Preprocessing-Domain]\n"
str2Write = str2Write + "Finished cross-validation. Took "+ str(t_url_full) +" minutes to Test. [Segmentation-Preprocessing-FULL]\n"
str2Write = str2Write + "Finished cross-validation. Took "+ str(t_nlp_url) +" minutes to Test. [NLP-Preprocessing-URL]\n"
str2Write = str2Write + "Finished cross-validation. Took "+ str(t_nlp_domain) +" minutes to Test. [NLP-Preprocessing-Domain]\n"
str2Write = str2Write + "Finished cross-validation. Took "+ str(t_nlp_full) +" minutes to Test. [NLP-Preprocessing-FULL]\n"
str2Write = str2Write + "Finished cross-validation. Took "+ str(t2test_url) +" minutes to Test-URL.\n"
str2Write = str2Write + "Finished cross-validation. Took "+ str(t2test_domain) +" minutes to Test-Domain.\n"
str2Write = str2Write + "Finished cross-validation. Took "+ str(t2test_full) +" minutes to Test-FULL.\n"

##### Writing Evaluation to a file >>> 
f_ = open(result_folder_path+'time_saver.txt','w',encoding='utf-8')
f_.writelines(str2Write)
f_.close()

score_lst = np.array(score_lst_url)
writeResult2File(score_lst,result_folder_path,mode='url')

score_lst = np.array(score_lst_domain)
writeResult2File(score_lst,result_folder_path,mode='domain')

score_lst = np.array(score_lst_full)
writeResult2File(score_lst,result_folder_path,mode='full')

##############################################################################################################################################
############################################## Endo of Implementation ##################################################################
##############################################################################################################################################
