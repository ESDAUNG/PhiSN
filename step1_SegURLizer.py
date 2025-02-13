import os, argparse, pandas as pd
import segURLizer, preprocessor
from transformers import BertTokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
parser = argparse.ArgumentParser(description="Tokenize Data")

### Initializing Model Directory >>>
parser.add_argument('--model.model_dir', type=str, default="/path-to-main-folder/", metavar="MODELDIR", help="location of model")

### Initializing Datasets >>>
parser.add_argument('--data.cur_directory', type=str, default="path-to-data-folder/",metavar="CurrentDir",help="location of current directory")
parser.add_argument('--data.path_to_df', type=str, default="data.csv",metavar="DataFrameDF",help="DataFrame DF file")

### Initializing arguments >>>
parser.add_argument('--log.kfold', type=int, default=10, metavar="Strafified KFold", help="number of k-fold")
parser.add_argument('--model.max_len', type=int, default=200, metavar="MaxLength", help="model max. number of word")
parser.add_argument('--model.vocab_size', type=int, default=5000, metavar="VocabSize", help="model max. vocabulary size")

### Parsing arguments >>>
FLAGS = vars(parser.parse_args())
n_fold = FLAGS["log.kfold"]
maxlen = FLAGS["model.max_len"]  # Only consider the first 200 words
vocab_size = FLAGS["model.vocab_size"] # Only consider the top 5000 words to be stored in Dictionary

### Assigning Directory >>>
print('----------------------------------- Creating PATH -----------------------------------------')
parent_directory = FLAGS["model.model_dir"]
cur_directory = FLAGS["data.cur_directory"]
path = os.path.join(parent_directory, cur_directory)
if not os.path.exists(path):
    os.mkdir(path)
else:
    path = path
print('----------------- Created PATH {} -----------------'.format(path))

### Loading Dataset >>>
############################################## Loading DataFrame df ##########################################################################################
print('----------------------------------- Loading CSV Dataset -----------------------------------------')
path_to_df = path+FLAGS["data.path_to_df"]
df = pd.read_csv(path_to_df, encoding="utf-8", sep=',')
print('----------------------------------- Loaded Dataset Size : {} ------------------------------------'.format(len(df)))

############################################## BERT Tokenizer ################################################################################################
tz_folder_name='tokenizer'
if not os.path.exists(path+tz_folder_name):
  os.mkdir(path+tz_folder_name)
  tokenizer_ = BertTokenizer.from_pretrained('bert-base-uncased')
  tokenizer_.save_pretrained(path+tz_folder_name)  # Save Bert Pretrained Tokenizer
else:
  path=path
  tokenizer_ = BertTokenizer.from_pretrained('bert-base-uncased')
print('----------------------------------- Loaded BERT-base-uncased Tokeinzer -----------------------------------------')

### This step is done for validation but used only if necessary. ###
print('----------------------------------- Converting IP2Domain if CONVERTIBLE -----------------------------------------')
df_converted = preprocessor.ip2host_converter(df)
df_converted.to_csv(path+'df_ip2host.csv',sep=',',encoding='utf-8',index=False)
print('----------------------------------- Converted IP2Domain if CONVERTIBLE -----------------------------------------')

############################################## Tokenization by SegURLizer ###########################################################################################
print('----------------------------------- Tokenization by SegURLizer -----------------------------------------')
# URL-Tokenizer followed by Text Vectorization 
segURLizer.tokenizer(df_converted['URLs'],'all',path,tokenizer_)