from datetime import time
from urllib.parse import unquote_plus,urlparse
from html import unescape
from wordsegment import load, segment
from transformers import BertTokenizer

import re, time, enchant, preprocessor
import numpy as np

# Final Version 
d = enchant.Dict("en_US")   # Initialize 'en_US' dictionary
load()  # Load WordSegment library

def recursive_tz_journal(word,tokenizer_:BertTokenizer):

    word_tz=tokenizer_.tokenize(word)  # Tokenized Words [List]

    if len(word_tz)==1: # Single Word <Cannot Tokenize anymore>
        return word_tz[0]
    
    else:   # Multiple Words <Tokenize from 2nd Words again>
        rec_str=word[len(word_tz[0]):]
        return word_tz[0]+'-'+recursive_tz_journal(rec_str,tokenizer_)

### SegURLizer [Word-Level]
def ws_bt_ec(urls,tokenizer_:BertTokenizer,level,path):

    tok_url,tok_id,tok_url_str_lst=[],[],[] # Initialize WORD tokens and their ID tokens
    tok_url_str=''
    timer_=[]   # Record preprocessing <Tokenization> time
    url_latency_recursion=[]
    unk_words_lst=[]    # List of UNKNOWN words
    words_corpus_lst=[] # List of Word Corpus
    unk_words_str=''   # String of UNKNOWN words
    words_corpus_str='' # String of Word Corpus

    for url in urls:

        start_time=time.time()  # Record START time of Tokenizer
        
        if not str(url).lower().startswith('http:') and not str(url).lower().startswith('https:') and  str(url).lower().startswith('www.'):
            url='http://'+str(url)
        elif not str(url).lower().startswith('http:') and not str(url).lower().startswith('https:') and  not str(url).lower().startswith('www.'):
            url='https://'+str(url)
        else:
                url=str(url)
        
        url=unescape(unquote_plus(url, encoding="utf-8")) # Unquote URLs
        parsedURL=urlparse(url) # URL Parser

        if level=='url': # URL in word-level tokenization
            if not parsedURL.scheme.lower().startswith('http:') and not parsedURL.scheme.lower().startswith('https:') and  parsedURL.scheme.lower().startswith('www.'):
                url='http://'+parsedURL.netloc.lower()+parsedURL.path.lower()
            elif not parsedURL.scheme.lower().startswith('http:') and not parsedURL.scheme.lower().startswith('https:') and  not parsedURL.scheme.lower().startswith('www.'):
                url='https://'+parsedURL.netloc.lower()+parsedURL.path.lower()
            else:
                url=parsedURL.scheme.lower()+'://'+parsedURL.netloc.lower()+parsedURL.path.lower()
        elif level=='domain':    # Domain in word-level tokenization
            if not parsedURL.scheme.lower().startswith('http:') and not parsedURL.scheme.lower().startswith('https:') and  parsedURL.scheme.lower().startswith('www.'):
                url='http://'+parsedURL.netloc.lower()
            elif not parsedURL.scheme.lower().startswith('http:') and not parsedURL.scheme.lower().startswith('https:') and  not parsedURL.scheme.lower().startswith('www.'):
                url='https://'+parsedURL.netloc.lower()
            else:
                url=parsedURL.scheme.lower()+'://'+parsedURL.netloc.lower()

        # Step 1. Tokenizing by Special Characters
        tmp_words=re.split(r"[-_;:,.=?^@\s$&?+!*\'()[\]{}|\"%~#<>/]",url)   # Step 1. Tokenizing by Special Characters
        tok_words=[]    # Tokenized Words [List] for Step 3. Input to Tokenizer
        w_list=[]   # Returned Word [List] tokenized / each word for Step 4.
        res_w_list=[]   # Save Final Word [List] after recursion / all URLs
        sp_chars=[] # Special Characters [List] for Step 1.

        # Extrating Special Characters
        string_check= re.compile(r'[-_;:,.=@\s$&?^+!*\'()[\]{}|\"%~#<>/]')  # Special Characters [string]
        for i in url:   # Each word in URL
            if (not i.isalnum()) and (string_check.search(i)!=None): # Speical Characters Extraction [List]
                sp_chars.append(i)

        # Step 2: Tokenizing Words with Suspicious Pattern
        for each in tmp_words:  # Words [List] w/o Special Characters

            # Step 2.1: To validate each WORD is NOT LARGER THAN 50, 
            if len(each)>0 and len(each)<=50:

                # Step 2.1.1: Decompose Consecutive Letter-Digit-Symbol patten
                str_cont_lst=preprocessor.susPattern_decomposer(each.lower()) #### Tokenize by Letter-Digit-Letter/Digit-Letter-Digit

                # Add validated WORD [0<word<=50] to tok_words
                tok_words=tok_words+str_cont_lst

            elif len(each)>50: # Step 2.2: To validate each WORD is LARGER THAN 50, 

                # Step 2.2.1: Decompose Consecutive Letter-Digit-Symbol patten
                str_cont_lst=preprocessor.susPattern_decomposer(each[:50].lower())    #### Tokenize by Letter-Digit-Letter/Digit-Letter-Digit

                # Add truncated WORD [0<word<=50] to tok_words
                tok_words=tok_words+str_cont_lst

            if len(sp_chars) >0:    # Merge validated words [List] : tok_words, w/ Special Characters [List] : sp_chars
                if str(sp_chars[0])==' ':
                    sp_chars.remove(sp_chars[0])
                else:
                    tok_words.append(str(sp_chars[0]))
                    sp_chars.remove(sp_chars[0])
        
        # Step 3: Segmenting Words by WORDSEGMENT
        # WordSegmenting a decomposed WORD [List] : tok_words
        ws_word=[]
        for decomp_word in tok_words:
            if str(decomp_word).isalnum():
                ws=segment(decomp_word)
                ws_word=ws_word+ws
            else:
                ws_word=ws_word+[decomp_word]
        
        word_latency_recursion=0.0
        # Step 4: Start Recursive_tokenization  (Return word [list] tokenized by Recursive_Tokenizer / each word)
        for word in ws_word:
            # Step 4.1 : Recursion on Alphabetic characters
            if  str(word).encode().isalpha(): # Check alphabetic character a-zA-Z
                # Step 4.1.1 : Recursive Tokenization of substring / each word
                start_recursion = time.time()
                rec_ret=re.split(r"[-]",recursive_tz_journal(word,tokenizer_)) # Add to Recursive_returned word [list] / each  word 
                stop_recursion = time.time()
                
                word_latency_recursion = word_latency_recursion + (stop_recursion-start_recursion)
                
                for i in rec_ret:   #Each subword in [word of segmented list]

                    if (d.check(i) or d.check(str(i).upper())) and (i ==rec_ret[0]):    # 1. English Dictionary Check==True & 2. First Word Check==True

                        w_list.append(str(i).lower()) # Add to Returned word [list] / each  word 

                    elif (d.check(i) or d.check(str(i).upper())) and (i !=rec_ret[0]):  # 1. English Dictionary Check==True & 2. First Word Check==Flase

                        w_list.append(str(i).lower())   # Add to Returned word [list] / each  word 

                    elif (not (d.check(i) or d.check(str(i).upper()))) and (i ==rec_ret[0]):    # 1. English Dictionary Check==False & 2. First Word Check==True

                        w_list.append(str(i).lower())   # Add to Returned word [list] / each  word 
                        
                    else:

                        w_list.append('##'+str(i).lower())# Add '##' to Returned word [list] / each  word , meaning as RANDOM words
                    
            else: # Step 4.2 : Check non alphabetic character ://.

                w_list.append(word) # Add to Returned word [list] / each  word 

        end_time=time.time()    # Record END time of Tokenizer
        timer_.append(end_time-start_time)  # Record preprocessing time of Tokenizer
        url_latency_recursion.append(word_latency_recursion)
        
        # Step 5: Replacing [UNK] for non-alphanumeric and non special characters  
        # Check Returned word [list] is neither alpha numeric characters nor special characters 
        # Convert to UNKNOWN word
        string_check= re.compile(r'[-_;:,.=@\s$&?^+!*\'()[\]{}|\"%~#<>/]')  # Special Characters [string]
        for each in w_list:

            if (string_check.search(each)==None) and (not str(each).encode().isalnum()) and (len(str(each).encode().strip())>0):    # Returned word [list] is neither alpha numeric characters nor special characters ?
                
                res_w_list.append('[UNK]')  # Convert/add to/as UNKNOWN word

                if (not each in unk_words_lst):

                    if len(str(each))>0:
                        unk_words_lst.append(each)  # Add to UNKNOWN word [list]
                        unk_words_str=unk_words_str+str(each)+'\n'  # Add to UNKNOWN word [String]
            
            elif (len(str(each).encode().strip())>0):

                res_w_list.append(each) # Add to final Word [List]
                
                if (not each in words_corpus_lst) and (string_check.search(each)==None):

                    words_corpus_lst.append(each)
                    words_corpus_str=words_corpus_str+str(each).strip()+'\n'

        tok_url.append(res_w_list)  # Add Tokens [List] to tok_url [List]
        tok_url_str_lst.append(' '.join(res_w_list))
        tok_url_str=tok_url_str+' '.join(res_w_list)+'\n'

    if len(unk_words_lst)>0:
        # Creating unknown words file
        unk_label=path+'unknown_words_{}.txt'.format(level)
        unk_file=open(unk_label,'w',encoding="utf-8")
        unk_file.writelines(unk_words_str)
        unk_file.close()

    if len(words_corpus_lst)>0:
        # Creating word corpus file
        corpus_label=path+'words_corpus_{}.txt'.format(level)
        corpus_file=open(corpus_label,'w',encoding="utf-8")
        corpus_file.writelines(words_corpus_str)
        corpus_file.close()

    timer_label=path+'record_url_tokenizer_{}.txt'.format(level)
    timer_rec=open(timer_label,'w',encoding="utf-8")
    total_time=np.sum(timer_)
    avg_time=np.mean(timer_)
    max_timer=np.max(timer_)
    min_timer=np.min(timer_)
    line='Preprocessing Time : '+str(total_time)+'\nAverage Tokenization time per URL: '+str(avg_time)+'\nMax Tokenization time : '+str(max_timer)+\
    '\nMin Tokenization time : '+str(min_timer)
    line+='\nRecursion Time : '+str(np.sum(url_latency_recursion))+'\nAverage Recursion time per URL: '+str(np.mean(url_latency_recursion))+'\nMax Recursion time : '+str(np.max(url_latency_recursion))+\
    '\nMin Tokenization time : '+str(np.min(url_latency_recursion))
    timer_rec.writelines(line)
    timer_rec.close()

    return tok_url,tok_url_str_lst,tok_url_str

# URL-Tokenizer [Char-level] Path tokenization 
# Not Used
def ws_bt_ec_path(urls,tokenizer_:BertTokenizer,path):

    load()  # Load WordSegment library
    path_tok_url,path_tok_id,path_tok_url_str_lst=[],[],[]    # Initialize CHAR tokens and their ID tokens
    path_tok_url_str=''

    unk_words_lst=[]    # List of UNKNOWN words
    words_corpus_lst=[] # List of Word Corpus
    unk_words_str=''   # String of UNKNOWN words
    words_corpus_str='' # String of Word Corpus

    for url in urls:

        if not str(url).lower().startswith('http:') and not str(url).lower().startswith('https:') and  str(url).lower().startswith('www.'):
            url='http://'+str(url)
        elif not str(url).lower().startswith('http:') and not str(url).lower().startswith('https:') and  not str(url).lower().startswith('www.'):
            url='https://'+str(url)
        else:
                url=str(url)
        
        url=unescape(unquote_plus(url, encoding="utf-8")) # Unquote URLs
        parsedURL=urlparse(url) # URL Parser

        url=''.join(list(parsedURL.path.lower()))   # Convert List to String without whitespace
        tmp_words=re.split(r"[-_;:,.=?@\s$&?^+!*\'()[\]{}|\"%~#<>/]",url)   # Step 1. Tokenizing by Special Characters

        sp_chars=[] # Special Characters [List] for Step 1.
        tok_words=[]    # Tokenized Words [List] for Step 3. Input to Tokenizer
        c_list=[]   # Returned Char [List] tokenized / each word for Step 4.
        res_c_list=[]   # Save Final Char [List] after recursion / all URLs
        res_c_str=''
       
        # Step 1. Tokenizing by Special Characters
        string_check= re.compile(r'[-_;:,.=@\s$&?^+!*\'()[\]{}|\"%~#<>/]')  # Special Characters [string]
        for i in url:   # Each word in URL
            if (not i.isalnum()) and (string_check.search(i)!=None): # Speical Characters Extraction [List]
                #print('added ',i)
                sp_chars.append(i)
        
        for each in tmp_words:  # Words [List] w/o Special Characters

            # Step 1.1: To validate each WORD is NOT LARGER THAN 50, 
            if len(each)>0 and len(each)<=50:
                # Add validated WORD [0<word<=50] to tok_words
                tok_words.append(each.lower())
            elif len(each)>50: # Step 1.2: To validate each WORD is LARGER THAN 50, 
                tok_words.append(each[:50].lower())

            if len(sp_chars) >0:    # Merge validated words [List] : tok_words, w/ Special Characters [List] : sp_chars
                if str(sp_chars[0])==' ':
                    sp_chars.remove(sp_chars[0])
                    #print('SPACE is removed!!!')
                else:
                    tok_words.append(str(sp_chars[0]))
                    sp_chars.remove(sp_chars[0])

        # Step 2. Convert WORD [List] to CHAR [List]
        c_list=re.sub(r"\s+","",''.join(tok_words))
        string_check= re.compile(r'[-_;:,.=@\s$&?^+!*\'()[\]{}|\"%~#<>/]')  #Special Characters 'string'
        
        # Check char [list] is neither alpha numeric characters nor special characters 
        # Convert to UNKNOWN word
        for each in c_list:

            # Step 2.1. Check if char [list] is neither alpha numeric characters nor special characters
            if (string_check.search(each)==None) and (not str(each).encode().isalnum()) and (len(str(each).encode().strip())>0):    # char [list] is neither alpha numeric characters nor special characters ?
                
                res_c_list.append('[UNK]')  # Convert/add to/as UNKNOWN word
                res_c_str=res_c_str+'[UNK]'+' '
                
                if (not each in unk_words_lst):
                    
                    if len(str(each))>0:
                    
                        unk_words_lst.append(each)  # Add to UNKNOWN word [list]
                        unk_words_str=unk_words_str+str(each)+'\n'  # Add to UNKNOWN word [String]
            
            elif (len(str(each).encode().strip())>0):
                
                res_c_list.append(each)
                res_c_str=res_c_str+each+' '
                
                if (not each in words_corpus_lst) and (string_check.search(each)==None):
                
                    words_corpus_lst.append(each)
                    words_corpus_str=words_corpus_str+str(each).strip()+'\n'
        
        path_tok_url.append(res_c_list)
        path_tok_url_str_lst.append(' '.join(res_c_list))
        path_tok_url_str=path_tok_url_str+res_c_str+'\n'
    
    if len(unk_words_lst)>0:

        # Creating unknown words file
        unk_label=path+'unknown_words_{}.txt'.format('path')
        unk_file=open(unk_label,'w',encoding="utf-8")
        unk_file.writelines(unk_words_str)
        unk_file.close()

    if len(words_corpus_lst)>0:

        # Creating word corpus file
        corpus_label=path+'words_corpus_{}.txt'.format('path')
        corpus_file=open(corpus_label,'w',encoding="utf-8")
        corpus_file.writelines(words_corpus_str)
        corpus_file.close()
    
    return path_tok_url,path_tok_url_str_lst,path_tok_url_str

# Char-level URL tokenization 
# Not Used
def ws_bt_ec_url_char(urls,tokenizer_:BertTokenizer,path):
    load()  # Load WordSegment library
    tok_url,tok_id,tok_url_str_lst=[],[],[]    # Initialize CHAR tokens and their ID tokens
    tok_url_str=''

    unk_words_lst=[]    # List of UNKNOWN words
    words_corpus_lst=[] # List of Word Corpus
    unk_words_str=''   # String of UNKNOWN words
    words_corpus_str='' # String of Word Corpus

    for url in urls:

        url=unescape(unquote_plus(url, encoding="utf-8")) # Unquote URLs
        #parsedURL=urlparse(url) # URL Parser
        w_list=[]   # Returned Word [List] tokenized / each word 
        res_c_list=[]   # Save Final Word [List] after recursion / all URLs
        res_c_str=''

        string_check= re.compile(r'[-_;:,.=@\s$&?^+!*\'()[\]{}|\"%~#<>/]')  # Special Characters [string] for Step 2.

        # Step 1. Converting WORD [List] to CHAR [List]
        url=url.lower()
        w_list=w_list + [i for i in list(url) if (len(str(i).strip())>0 and len(str(i).strip())<=50)]

        for each in w_list:

            # Step 2. Check if char [list] is neither alpha numeric characters nor special characters
            if (string_check.search(each)==None) and (not str(each).encode().isalnum()):    # Returned char [list] is neither alpha numeric characters nor special characters ?
                res_c_list.append('[UNK]')  # Convert/add to/as UNKNOWN word
                res_c_str=res_c_str+'[UNK]'
            else:
                res_c_list.append(each)
                res_c_str=res_c_str+each+' '

                if (not each in words_corpus_lst) and (string_check.search(each)==None):
                
                    words_corpus_lst.append(each)
                    words_corpus_str=words_corpus_str+str(each).strip()+'\n'

        tok_url.append(res_c_list)
        tok_url_str_lst.append(' '.join(res_c_list))
        tok_url_str=tok_url_str+res_c_str+'\n'
    
    if len(unk_words_lst)>0:

        # Creating unknown words file
        unk_label=path+'unknown_words_{}.txt'.format('url_char')
        unk_file=open(unk_label,'w',encoding="utf-8")
        unk_file.writelines(unk_words_str)
        unk_file.close()

    if len(words_corpus_lst)>0:

        # Creating word corpus file
        corpus_label=path+'words_corpus_{}.txt'.format('url_char')
        corpus_file=open(corpus_label,'w',encoding="utf-8")
        corpus_file.writelines(words_corpus_str)
        corpus_file.close()

    return tok_url,tok_url_str_lst,tok_url_str

# SegURLizer 
# Main Function
def tokenizer(urls,mode,path,tokenizer_):
    # Call SegURLizer-Algorithm for Word-level Tokenization of URL
    tok_word_url,tok_word_url_lst,tok_word_url_str=ws_bt_ec(urls,tokenizer_,'url',path)
    # Writing tokens file 
    label=path+"{}_url.txt".format(mode)
    with open(label, "w",encoding="utf-8") as opfile:
        opfile.writelines(tok_word_url_str)
    
    # Call SegURLizer-Algorithm for Char-level URL Tokenization of URL
    #tok_char_url,tok_char_url_lst,tok_char_url_str=ws_bt_ec_url_char(urls,tokenizer_,path)
    # Writing tokens file 
    #label=path+"{}_url_char.txt".format(mode)
    #with open(label, "w",encoding="UTF-8") as opfile:
    #    opfile.writelines(tok_char_url_str)

    # Call SegURLizer-Algorithm for Word-level Tokenization of Domain
    #tok_word_domain,tok_word_domain_lst,tok_word_domain_str=ws_bt_ec(urls,tokenizer_,'domain',path)
    # Writing tokens file 
    #label=path+"{}_domain.txt".format(mode)
    #with open(label, "w",encoding="utf-8") as opfile:
    #    opfile.writelines(tok_word_domain_str)

    # Call SegURLizer-Algorithm for Char-level Tokenization of Path
    #tok_char_path,tok_char_path_lst,tok_char_path_str=ws_bt_ec_path(urls,tokenizer_,path) # char-level tokenization (including 50 chars in one word)
    # Writing tokens file <tok_char_path_str>
    #label=path+"{}_path.txt".format(mode)
    #with open(label, "w",encoding="utf-8") as opfile:
    #    opfile.writelines(tok_char_path_str)

    # Concatenation of Word-level Domain && Char-level Path
    #cplx_url_str=''
    #cplx_id_url,cplx_url=[],[]   # Word-level Domain + Char-level Path Tokenization
    #for i in range(0,len(tok_word_domain)):
    #    cplx_url.append(tok_word_domain[i]+[' ']+tok_char_path[i])
    #    cplx_url_str=cplx_url_str+' '.join(tok_word_domain[i])+' '+' '.join(tok_char_path[i])+'\n'
    # Writing tokens file 
    #label=path+"{}_cplx_url.txt".format(mode)
    #with open(label, "w",encoding="utf-8") as opfile:
    #    opfile.writelines(cplx_url_str)
