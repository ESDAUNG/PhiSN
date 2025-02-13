def load_tokenized_data(path,mode):
    ##### Loading Word-level tokenization of URL
    X_url_word=[]
    file_name="{}_url.txt".format(mode)
    with open(path+file_name, "r",encoding='utf-8') as opfile:
        for line in opfile:
            X_url_word.append(line.strip())

    ##### Loading Character-level tokenization of URL
    #X_url_char_word=[]
    #file_name="{}_url_char.txt".format(mode)
    #with open(path+file_name, "r",encoding='utf-8') as opfile:
    #    for line in opfile:
    #        X_url_char_word.append(line.strip())

    ##### Loading Word-level tokenization of Domain
    X_domain_word=[]
    file_name="{}_domain.txt".format(mode)
    with open(path+file_name, "r",encoding='utf-8') as opfile:
        for line in opfile:
            X_domain_word.append(line.strip())

    ##### Loading word-level tokenization of Path
    #X_path_word=[]
    #file_name="{}_path.txt".format(mode)
    #with open(path+file_name, "r",encoding='utf-8') as opfile:
    #    for line in opfile:
    #        X_path_word.append(line.strip())

    ##### Loading Word-level tokenization of Domain + Character-level tokenization of Path
    #X_cplx_word_url=[]
    #file_name="{}_cplx_url.txt".format(mode)
    #with open(path+file_name, "r",encoding='utf-8') as opfile:
    #    for line in opfile:
    #        X_cplx_word_url.append(line.strip())

    #return X_url_word,X_url_char_word,X_domain_word,X_path_word,X_cplx_word_url
    return X_url_word,X_domain_word
