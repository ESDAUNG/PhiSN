import os,argparse
import pandas as pd
import dataLoader

from sklearn.model_selection import StratifiedKFold

# Final Verstion
parser = argparse.ArgumentParser(description="Cross-validating by index and Saving DF")

### Initializing Model Directory >>>
parser.add_argument('--model.model_dir', type=str, default="/path-to-main-folder/", metavar="MODELDIR", help="location of model")

### Initializing Datasets >>>
parser.add_argument('--data.cur_directory', type=str, default="path-to-data-folder/",metavar="CurrentDir",help="location of current directory")
parser.add_argument('--data.path_to_df', type=str, default="data.csv",metavar="DataFrameDF",help="DataFrame DF file")

### Initializing arguments >>>
parser.add_argument('--log.kfold', type=int, default=10, metavar="Strafified KFold", help="number of k-fold")

### Parsing arguments >>>
FLAGS = vars(parser.parse_args())
n_fold = FLAGS["log.kfold"]

### Assigning Directory >>>
parent_directory = FLAGS["model.model_dir"]
cur_directory = FLAGS["data.cur_directory"]
path = os.path.join(parent_directory, cur_directory)
if not os.path.exists(path):
    os.mkdir(path)
else:
    path = path
print('Created path ', path)

### Assigning Result Directory >>>
cur_folder_path = 'all_models_cv/'
result_folder_path = os.path.join(path, cur_folder_path)
if not os.path.exists(result_folder_path):
    os.mkdir(result_folder_path)
else:
    result_folder_path = result_folder_path
print('Created result folder path ', result_folder_path)

##############################################################################################################################################
##############################################################################################################################################
### Loading Dataset >>>
############################################## Loading DataFrame df ################################################################################################
print('----------------------------------- Loading CSV Dataset -----------------------------------------')
path_to_df = path+FLAGS["data.path_to_df"]
df = pd.read_csv(path_to_df, encoding="utf-8", sep=',')
print('Loaded data size : {}'.format(len(df)))

### Loading Pre-Tokenized Data by SegURLizer >>>
############################################## Loading Tokenized Data File ################################################################################################
X_url_word, X_domain_word = dataLoader.load_tokenized_data(path, "all")

### Cloning Pre-Tokenized Data to a new dataframe >>>
df_tokenized = pd.DataFrame()
df_tokenized['URLs'] = df['URLs'].to_list()
df_tokenized['Labels'] = df['Labels'].to_list()
df_tokenized['URL_Word'] = X_url_word

##############################################################################################################################################
##################### Function: Cross-Validation ###########################################################################################
##############################################################################################################################################
def crossValidate(n_splits,data_df,result_folder_path):
    kfold = StratifiedKFold(n_splits, shuffle=True, random_state=42) #, random_state=seed
    fold_no = 0

    print("Starting {}-fold cross-validation".format(n_splits))
    for train_idx, test_idx in kfold.split(X=data_df, y=data_df['Labels']):
        fold_no = fold_no+1
        print("Fold = {} ... ".format(fold_no))

        ### Computing Class Weights  >>> 
        train_idx_df = pd.DataFrame(train_idx,index=None,columns=['index'])
        test_idx_df = pd.DataFrame(test_idx,index=None,columns=['index'])
        train_idx_df.to_csv(result_folder_path+'train_idx{}.csv'.format(fold_no),sep=',',index=False,encoding='utf-8')
        test_idx_df.to_csv(result_folder_path+'test_idx{}.csv'.format(fold_no),sep=',',index=False,encoding='utf-8')

### Cross-validating DF >>>
crossValidate(n_splits=n_fold,data_df=df_tokenized,result_folder_path=result_folder_path)

##############################################################################################################################################
##################### Loading Cross-Validated Data [OPTIONAL] #################################################################################
##############################################################################################################################################

f_train_list, f_valid_list = [],[]
for path_file, Directory, files in os.walk(result_folder_path):  
    for file in files:
        if (file.startswith("train_idx") and file.endswith(".csv")):
            f_train_list.append(file)
            f_train_list.sort()
        
        if (file.startswith("test_idx") and file.endswith(".csv")):
            f_valid_list.append(file) 
            f_valid_list.sort()
    break

##############################################################################################################################################
##################### Saving Cross-Validated Data [OPTIONAL] #################################################################################
##############################################################################################################################################

for indx in range(10):
    char = str(f_train_list[indx])[-5]
    if char.endswith("0"):
        char = str('10')
    
    ### Retrieving cross-validated Indexes >>>
    train_idx_df = pd.DataFrame(pd.read_csv(result_folder_path+f_train_list[indx], sep=',', encoding='utf-8'))
    valid_idx_df = pd.DataFrame(pd.read_csv(result_folder_path+f_valid_list[indx], sep=',', encoding='utf-8'))
    train_idx=train_idx_df['index'].to_list()
    valid_idx=valid_idx_df['index'].to_list()

    ### Indexing cross-validated data to new dataframe for URLs and Labels >>>
    train_df = pd.DataFrame(df.iloc[train_idx],index=None,columns=['URLs','Labels'])
    valid_df = pd.DataFrame(df.iloc[valid_idx],index=None,columns=['URLs','Labels'])

    ### Indexing cross-validated data to new dataframe for URL_Word >>>
    train_df['URL_Word'] = pd.DataFrame(X_url_word,index=None).iloc[train_idx]
    valid_df['URL_Word'] = pd.DataFrame(X_url_word,index=None).iloc[valid_idx]

    ### Saving cross-validated data >>>
    train_df.to_csv(result_folder_path+'X_train_{}.csv'.format(char),sep=',',index=False,encoding='utf-8')
    valid_df.to_csv(result_folder_path+'X_valid_{}.csv'.format(char),sep=',',index=False,encoding='utf-8')