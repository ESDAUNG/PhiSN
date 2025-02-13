import pandas as pd
import re, socket
from urllib.parse import urlparse, unquote_plus
from tld import get_tld

# Retrieve HOST by IP address
def tld_checker(url:str):

    parsedURL=urlparse(url.strip())

    domain=parsedURL.scheme.lower().strip()+'://'+parsedURL.netloc.lower().strip()

    try:
        res=get_tld(domain,as_object=True)
    except:
        try:
            split_url =''
            if ':' in parsedURL.netloc.lower().strip():
                split_url = parsedURL.netloc.lower().split(':')[0].strip()
            else:
                split_url = parsedURL.netloc.lower().strip()
            DNS_hostname = socket.gethostbyaddr(split_url)[0].strip()

            domain = parsedURL.scheme.lower().strip()+'://'+DNS_hostname
            res=get_tld(domain,as_object=True)

            ip2host_url = parsedURL.scheme.lower().strip()+'://'+DNS_hostname+parsedURL.path.strip()+parsedURL.params.strip()+parsedURL.query.strip()+parsedURL.fragment.strip()

            return res.tld,ip2host_url
        except:
            return '-','-'

    return res.tld,'-'

# Replace IP address with HOST if HOST exists
def ip2host_converter(df:pd.DataFrame):
    tlds=[] # For tlds list
    converted_url = []
    for index,row in df.iterrows():
        tmp_tld, ip2host_url=tld_checker(str(row['URLs']).strip()[:200])
        tlds.append(tmp_tld)
        if ip2host_url != '-':
            df = df.replace([row['URLs']], ip2host_url)
            converted_url.append(ip2host_url[:200])
        else:
            converted_url.append(str(row['URLs']).strip()[:200])
    df['URLs']=converted_url
    df['TLD']=tlds
    df=df.drop(columns=['TLD'])
    return df

def susPattern_decomposer(url:str):
    pre='None'
    url_word=[]
    url_word_digit=''
    url_word_letter=''
    url_word_symbol=''

    for i in url:
        if i.isalpha():
            if pre=='digit':
                url_word.append(url_word_digit)
                url_word_digit=''
            if pre=='symbol':
                url_word.append(url_word_symbol)
                url_word_symbol=''
            url_word_letter=url_word_letter+i
            pre='letter'
        
        if i.isnumeric():
            if pre=='letter':
                url_word.append(url_word_letter)
                url_word_letter=''
            if pre=='symbol':
                url_word.append(url_word_symbol)
                url_word_symbol=''
            url_word_digit=url_word_digit+i
            pre='digit'
        
        if not i.isalnum():
            if pre=='letter':
                url_word.append(url_word_letter)
                url_word_letter=''
            if pre=='digit':
                url_word.append(url_word_digit)
                url_word_digit=''
            url_word_symbol=url_word_symbol+i
            pre='symbol'
    if pre=='letter':
        url_word.append(url_word_letter)
    elif pre=='digit':
        url_word.append(url_word_digit)
    else:
        url_word.append(url_word_symbol)

    return url_word

def splitByPath(df:pd.DataFrame,indx_df:pd.DataFrame):
    domain_, url_, full_, domain_all_ = [], [], [], []
    label_domain_, label_url_, label_full_, label_domain_all_ = [],[], [], []
    indx_domain_, indx_url_, indx_full_, indx_domain_all_ = [], [], [], []
    url_pos_, url_neg_, domain_pos_, domain_neg_, full_pos_, full_neg_ = 0,0,0,0, 0,0

    for indx,row in indx_df.iterrows():
        rowIdx = int(row['index'])
        url = df['URLs'].iloc[rowIdx]
        #print('Now : {}'.format(url))

        parsedDomain , parsedScheme = '' , ''
        unquote_url = unquote_plus(str(url).strip())
        if unquote_url.startswith('http'):
            parsedURL = urlparse(unquote_url)
            parsedScheme = parsedURL.scheme+'://'
        else:
            parsedURL = urlparse('https://'+unquote_url)
            parsedScheme = ''

        parsedPath = parsedURL.path
        parsedParam = parsedURL.params
        parsedQuery = parsedURL.query
        parsedFragment = parsedURL.fragment
        path_info = parsedPath+parsedParam+parsedQuery+parsedFragment

        if len(path_info) > 1:
            url_.append(url)
            label_url_.append(df['Labels'].iloc[rowIdx])
            indx_url_.append(rowIdx)
            #print('Add to URL : {}, Label : {}'.format(url,df['Labels'].iloc[rowIdx]))
            if int(df['Labels'].iloc[rowIdx]) == 1:
                url_pos_= url_pos_+1
            else:
                url_neg_ = url_neg_+1

        else:
            parsedDomain = parsedScheme+parsedURL.netloc
            domain_.append(parsedDomain)
            label_domain_.append(df['Labels'].iloc[rowIdx])
            indx_domain_.append(rowIdx)
            #print('Add to Domain : {}, Label : {}'.format(parsedDomain,df['Labels'].iloc[rowIdx]))
            if int(df['Labels'].iloc[rowIdx]) == 1:
                domain_pos_= domain_pos_+1
            else:
                domain_neg_ = domain_neg_+1
        
        full_.append(url)
        label_full_.append(df['Labels'].iloc[rowIdx])
        indx_full_.append(rowIdx)
        domain_all_.append(parsedScheme+parsedURL.netloc)
        label_domain_all_.append(df['Labels'].iloc[rowIdx])
        indx_domain_all_.append(rowIdx)

    domain_df = pd.DataFrame()
    url_df = pd.DataFrame()
    full_df = pd.DataFrame()
    domain_all_df = pd.DataFrame()

    domain_df['URLs'] = domain_
    domain_df['Labels'] = label_domain_
    domain_df['index'] = indx_domain_
    url_df['URLs'] = url_
    url_df['Labels'] = label_url_
    url_df['index'] = indx_url_
    full_df['URLs'] = full_
    full_df['Labels'] = label_full_
    full_df['index'] = indx_full_
    domain_all_df['URLs']= domain_all_
    domain_all_df['index'] = indx_domain_all_
    domain_all_df['Labels'] = label_domain_all_

    return domain_df,url_df,full_df,domain_all_df