import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import re,time,os,json
from collections import Counter
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path

#import multiprocessing, psutil,collections  

pd.set_option('max_colwidth',120)

def tokenizer_fr(text):
    """A tokenizer for french applying nltk.word_tokenize , every number kept"""
    res = [ v for v in word_tokenize(text, language="french") 
            if (v in list("1234567890") or len(v)>1) and v not in ["''", ".."] ]#keep numbers
    return res


class NgramCorpus:
    #this class depend on nltk wordtokenizer, numpy,pandas, CountVectorizer of sklearn
    def __init__(self,  corpus_file, corpus = None, tokenizer = tokenizer_fr):
        """corpus_file:  a dict with key: filename, val:text in file"""
        self.corpus_file = corpus_file       
        self.corpus = corpus if(corpus) else ''.join(corpus_file.values())
        self.tokenizer = tokenizer
        
    
    def get_ngram_freq(self, nrange = (2,2), col0 = "bigram", col1 = "freq"):
        """find the ngrams of current corpus, with the frequency(integer) of each ngram """
        vectorizer = CountVectorizer(analyzer="word",tokenizer = self.tokenizer, lowercase=False,
                                      ngram_range=nrange)
        X = vectorizer.fit_transform([self.corpus])
        res_df = pd.DataFrame(vectorizer.get_feature_names_out(), columns = [col0])
        res_df[col1] = pd.DataFrame(X.sum(axis = 0)).T
        return X, res_df
    
    
    def get_distribution_common(self, common_text):
        """given some text, find their location in current corpus
        return 
        file_common with information with respect to text (text_index : file containing this text)
        file_common_f2s with information with respect to file (filename : textindex which relevant text exist in the file)
        """
        file_common = {}
        file_common_f2s ={}

        for i, txt in enumerate(common_text):
            file_common[i] = []

        for f, doc in self.corpus_file.items():
            txt = ' '.join(self.tokenizer(doc))#tokenizer used in countvectoriser
            for idx, common in enumerate(common_text):
                if(common in txt):
                    file_common[idx].append(f)
                    file_common_f2s[f] = file_common_f2s.get(f,[])
                    file_common_f2s[f].append(idx)
            
        return file_common, file_common_f2s
    
    
    

    
class CorpusCompare:
    
    def __init__(self,  corpusA, corpusR, resFolder = 'res_report'):
        """corpusA, corpusR: corpus of class ngramCorpus"""
        self.corpusA = corpusA
        self.corpusR = corpusR
        self.resFolder = resFolder
        Path(resFolder).mkdir(parents=True, exist_ok=True)
        
        assert(corpusA.tokenizer == corpusR.tokenizer)
        self.tokenizer = corpusA.tokenizer            
        
        #added for multi processing
        self.A_dfs = None
        self.R_dfs = None
        self.common_all_df = None
        
        
    @classmethod
    def fromFile(cls, corpusA_file, corpusR_file, corpusA = None, corpusR = None, tokenizer = tokenizer_fr, resFolder = 'res_report'):
        """corpus_file:  a dict with key: filename, val:text in file"""
        corpusA = ngramCorpus(corpusA_file, corpusA, tokenizer = tokenizer, resFolder = resFolder)
        corpusR = ngramCorpus(corpusR_file, corpusR, tokenizer = tokenizer, resFolder = resFolder )
        
        return cls(corpusA, corpusR )
    
    def reset(self):
        """added for multiprocessing that doesn't work well"""
        self.A_dfs = None
        self.R_dfs = None
        self.common_all_df = None
        
    
    def get_common_stat(self,colname = "bigram", nrange = (2,2)):
        X0, corpus0_df = self.corpusA.get_ngram_freq(nrange = nrange, col0 = colname, col1 = "freq_c0")
        X1, corpus1_df = self.corpusR.get_ngram_freq( nrange = nrange, col0 = colname, col1 = "freq_c1")

        corpus0_df["freq_c0"] = pd.DataFrame(X0.sum(axis = 0)).T
        corpus1_df["freq_c1"] = pd.DataFrame(X1.sum(axis = 0)).T

        common = pd.merge(corpus0_df, corpus1_df, on = [colname])

        common["percent_c0"] =  common["freq_c0"] /X0.sum() #frequency of ngram in common in corpus A
        common["percent_c1"] =  common["freq_c1"] /X1.sum() #frequency of ngram in common in corpus R 
        common["percent_common_c0"] =  common["freq_c0"] /common['freq_c0'].sum() #frequency among ngram in common

        return common, corpus0_df, corpus1_df
    
    def get_common(self, n, key = 'ngram'):
        """return the ngram in common with their frequency(int)"""
        _, A_df = self.corpusA.get_ngram_freq(nrange = (n,n), col0 = key, col1 = "freq_c0")
        _, R_df = self.corpusR.get_ngram_freq(nrange = (n,n), col0 = key, col1 = "freq_c1")

        common_df = pd.merge(A_df, R_df, on = [key])
        return common_df
    
    def find_longest_common(self, step_begin = 8, n_begin = 8, filename = 'longest_match.csv', store = True):
        #first simple version to improve 
        """
        a  dichotomy and ngram approach to find the longes text in common
        filename should be name of a tsv file
        """
        n = n_begin
        step = step_begin
        find = -1
        begin = -1
        res = 0

        while(find == -1 and step>0 and n>0):
            key =str(n)+'gram'
            print("looking for common text in "+key)
            _, A_df = self.corpusA.get_ngram_freq(nrange = (n,n), col0 = key, col1 = "freq_c0")
            _, R_df = self.corpusR.get_ngram_freq(nrange = (n,n), col0 = key, col1 = "freq_c1")

            common_df = pd.merge(A_df, R_df, on = [key])
            len_common = len(common_df)
            
            if(len_common == 1): #if the longest match unique
                find = n 
                print("The longest match found in "+key)
                res = common_df
                break
                # common_df.rename(columns = {key: 'text_common'}, inplace=True)
                # return common_df

            if(begin != -1 and len_common > 1): #check if the longest match not unique
                common = self.get_common(n+1, key = str(n+1)+'gram')
                if(len(common) == 0):
                    find = n
                    print("The longest match found in "+key)
                    res = common_df
                    break
                    # common_df.rename(columns = {key: 'text_common'}, inplace=True)
                    # return common_df

            #update n and step
            if(len_common>0):
                print(f'---- got {len_common} {key}')
                step = int(step*0.5) if(begin != -1) else step
                step = 1 if(len_common == 2) else step #to go faster 
                n = n + step
            else:
                print(f"0 {n}gram in common")
                begin = n #where n = 0
                step = int(step*0.5)
                n = n-step
        if(step == 0 and find == -1):
            print('non common text')
            return res
        res.rename(columns = {key: 'text_common'}, inplace=True)
        #store the result 
        if store:
            res.to_csv(self.resFolder+'/'+filename, sep = '\t')
        return res
 
    
    def ngram_info(self, n_start, max_n, info = 'A_vs_R', store = True):
        """
        find ngram with n in [n_start, max_n]
        compute also the coverage, recouvrement, frequency, and amount of ngram in both corpus to compare
        """
        tbegin = time.time()
        assert(n_start>0 and max_n >= n_start)

        cover = {}
        df_detail = {}
        freq_normed = {}
        len_all = {'len_A':{}, 'len_R':{}, 'len_all':{}}
        type_tok_ratio = {'ttr_A':{}, 'ttr_R':{}, 'ttr_all':{}}
        
        print(f"computing ngram in common with n in range [{n_start},{max_n}]")
        _, self.A_dfs = self.corpusA.get_ngram_freq(nrange = (n_start,max_n), col0 = "ngram", col1 = "freq_c0")
        _, self.R_dfs = self.corpusR.get_ngram_freq(nrange = (n_start,max_n), col0 = "ngram", col1 = "freq_c1")

        self.common_all_df = pd.merge(self.A_dfs, self.R_dfs, on = ["ngram"])
        
        print(f"done with {time.time() -tbegin}s, analysing result for each n ")
        tbegin = time.time()
        
        for n in range(n_start,max_n+1):
            key = str(n)+'gram'
            is_common_df = self.common_all_df.ngram.apply(lambda x: 1 if len(x.split()) == n else None).dropna()
            is_A_df = self.A_dfs.ngram.apply(lambda x: 1 if len(x.split()) == n else None).dropna()
            is_R_df = self.R_dfs.ngram.apply(lambda x: 1 if len(x.split()) == n else None).dropna()
            
            #ngram in common
            common_df = self.common_all_df.loc[is_common_df.index]
            common_df.reset_index(inplace = True, drop = True)
            common_df.rename(columns = {'ngram': key}, inplace=True)
            
            type_a = self.A_dfs.loc[is_A_df.index]['freq_c0'].sum()
            type_r = self.R_dfs.loc[is_R_df.index]['freq_c1'].sum() 
            common_df["percent_c0"] = common_df["freq_c0"]/type_a #frequency of ngram in common in corpus A
            common_df["percent_c1"] = common_df["freq_c1"]/type_r #frequency of ngram in common in corpus R 
            common_df["percent_common_c0"] =  common_df["freq_c0"] /common_df['freq_c0'].sum() #frequency among ngram in common

               
            df_detail[key] = common_df.sort_values("freq_c0", ascending=False)
            if len(is_A_df):
                cover[key] = len(common_df)/len(is_A_df)
            else:
                cover[key] = 0.
                print(f'no {n}gram in the corpus A')
                
            print("There are ", len(common_df), f"{n}gram in both corpus. Coverage of A to R is {cover[key]}")
            
            len_all['len_A'][key] = len(is_A_df)
            len_all['len_R'][key]= len(is_R_df)
            len_all['len_all'][key] = (len(is_A_df) + len(is_R_df) - len(common_df) )
            freq_normed[key]= len(common_df)/len_all['len_all'][key]#recouvrement
            
            #token_type_ratio
            type_tok_ratio['ttr_A'][key] = len_all['len_A'][key]/type_a
            type_tok_ratio['ttr_R'][key] = len_all['len_R'][key]/type_r
            type_tok_ratio['ttr_all'][key] = len_all['len_all'][key]/(type_a + type_r)
            
            
        #dataframe occupies less space than dict
        cover = pd.DataFrame(cover, index = ['couverture_'+info])  
        recouvr = pd.DataFrame(freq_normed, index = ['recouvrement_'+info])
        len_all = pd.DataFrame(len_all)
        ttr = pd.DataFrame(type_tok_ratio)
        print(f"done with {time.time() -tbegin}s")
        
        #store result in file
        res_fd = self.resFolder
        if store:
            res_fd = self.resFolder+f'/{info}_{n_start}_to_{max_n}'
            Path(res_fd).mkdir(parents=True, exist_ok=True)
            
            print("writing results in "+res_fd)
            #tried to put details in a single json file but failed to read them after, so one file for each ngram          
            # with open(f'{resFolder}/ngram_details_{n_start}_to_{max_n}.json', 'w') as f:
            #     for n in range(n_start, max_n+1):
            #         json.dump(json.dumps(df_detail[str(n)+'gram'].to_dict(), indent = 4), f)
            #         f.write('\n\n')
            
            Path(res_fd+'/ngram_common').mkdir(parents=True, exist_ok=True)        
            for k, d in df_detail.items():
                d.to_csv(f'{res_fd}/ngram_common/{k}.tsv', sep = '\t')
                
            #coverage and recouvrement
            pd.concat([cover, recouvr]).to_csv(f'{res_fd}/cover_recouv.tsv', sep = '\t')
            #The amount of ngram
            len_all.to_csv(f'{res_fd}/len_ngram.tsv', sep = '\t')
            #type_token_ratio
            ttr.to_csv(f'{res_fd}/ttr.tsv', sep = '\t')
            
        self.reset()
        
        return CorpusCompareReport(cover, df_detail, recouvr, len_all, ttr , res_fd ) #cover, df_detail, freq_normed
 


    
class CorpusCompareReport:
    
    def __init__(self, cover, df_details,  recouvr, len_all, ttr, res_folder):
        """
        output of CorpusCompare(corpusA, corpusR).ngram_info(), 
        corpusA, corpusR: corpus of class ngramCorpus
        cover, ngrams in df_detail, recouvrement, length of ngram, ttr(type_token_ratio) 
        """
        self.cover = cover
        self.df_details= df_details
        self.freq_normed = recouvr
        self.len_all = len_all
        self.ttr = ttr
        self.res_folder = res_folder
        
    @classmethod
    def fromFolder(cls, resFolder):
        """
        create an instance from ngram_info stored in resFolder
        name of resFolder with format '{name_prefix}_{n_start}_to_{max_n}'
        """
        info = '_'.join(resFolder.split('/')[-1].split('_')[:-3]) #find name_prefix 

        cover_couvr = pd.read_csv(resFolder+ f'/cover_recouv.tsv', sep = '\t', index_col = 0)
        cover = cover_couvr.loc[['couverture_'+info]]
        recouvr = cover_couvr.loc[['recouvrement_'+info]]
        len_all = pd.read_csv(resFolder + f'/len_ngram.tsv', sep = '\t', index_col = 0)
        ttr = pd.read_csv(resFolder + f'/ttr.tsv', sep = '\t', index_col = 0)
        df_detail = get_details_from_tsv(resFolder)
        
        return cls(cover, df_detail,  recouvr, len_all, ttr, resFolder )
    
    def get_coverage(self, index = None):
        """
        coverage of first corpus to second:(e.g. row cover_AR shows the coverage of A to R,
        i.e len(common_ngarm)/len(ngarm_in_A))
        """
        if index:
            self.cover.rename(index = {self.cover.index.values[0] : index} )
        return self.cover
    
    def get_normed_freq(self, index = None):
        """The recouvrement = len(common_ngram)/len(all_ngram_in_both_corpus)"""
        if index:
            self.freq_normed.rename(index = {self.freq_normed.index.values[0] : index} )
        return self.freq_normed
    
    def get_detail(self):
        return self.df_details
    
    def get_len_all(self):
        return self.len_all
    
    def get_ttr(self,imgName = 'diver_ngram(TTR).png' ):
        #type/token ratio
        self.ttr.plot(kind = 'bar', title = 'ngram diversity(TTR)'+self.res_folder.split('/')[-1])
        plt.savefig(self.res_folder + '/' + imgName)
        return self.ttr
    
    def get_rare_txt(self,k = 5):
        #return ngram which frequency in corpus R is smaller or equal tha k
        return {key: v[v['freq_c1']<=k] for key, v in self.df_details.items()}
    
    def get_rare_report(self, k = 5, show = True):
        rare_dict = self.get_rare_txt(k = k)
        rare_len = {key:len(rare_dict[key]) for key in sorted(rare_dict.keys())}
        rare_len_df = pd.DataFrame(rare_len, index = ['len_k'+str(k)])
        if show:
            print(f'\033[1mK = {k}\033[0m')
            print(f'total nomber of rare ngram set (frequency <= {k}) for each n:')
            display(rare_len_df)
            print('total nomber of ngram set for each n:')
            display(self.get_freq_report())
        return rare_len_df
    

    def get_freq_report(self):
        """Return a DataFrame including the length of ngram in common for different ngram in df_detail """
        len_common = {k :len(self.df_details[k]) for k in sorted(self.df_details.keys())}
        len_common_df = pd.DataFrame(len_common, index = ['length'])
        return len_common_df
    
    def get_common_text(self, min_n = 2, verbose = True, filename = "common_text_backward.txt", store = True):
        """freq_df, dataFrame including(ngram_name: length of ngram in common ), output of get_freq_report """
        #we suppose the ngram in self.df_detail with the largest n contains the most long text in common 
        freq_df = self.get_freq_report()
        keys = list(self.df_details.keys())

        #Begin with the longest ngram in common
        current_idx = -2 if(freq_df[keys[-1]].length==0) else -1
        common_text = self.df_details[keys[current_idx]][keys[current_idx]].values.tolist()
        
        #may depend on get_common_text_singleN 
        common_text = get_common_text_singleN(common_text, list(self.df_details[keys[current_idx]]['freq_c0'].values),int(keys[current_idx][:-4])) if(len(common_text) > 1) else common_text
        
        if(verbose):
            print(print(keys[current_idx], len(common_text),'\n', common_text))

        while(int(keys[current_idx][:-4])> max(min_n,int(keys[0][:-4])) and min_n > 1):
            current_idx -= 1
            #print("====",int(keys[current_idx][:-4]))
            if(freq_df[ keys[current_idx] ].length> freq_df[ keys[current_idx+1] ].length+ len(common_text)):
                #if there are other ngram than that in common_text:
                for txt in self.df_details[ keys[current_idx]][ keys[current_idx] ].values:
                    if(np.all([txt not in c for c in common_text])):
                        common_text.append(txt)
                        if(verbose):
                            print(keys[current_idx], len(common_text),'\n',txt)
        if store:
            with open(self.res_folder+'/'+filename, 'w') as f:
                f.write('\n'.join(common_text))
        return common_text
    

    
    def report_file_s2f(self, filename, common_text, distribution0, distribution1, name0 = 'corpusR', name1 = 'corpusA'):
        with open(self.res_folder+'/'+filename,'w') as f:
            f.write("\tdistribution_"+name0+"\tdistribution_"+ name1+"\ttext_common\n")
            
            for i in np.arange(len(common_text)):
                f.write(str(i)+'\t'+ '||'.join(distribution0[i]) + '\t' 
                        + '||'.join(distribution1[i])+'\t' + common_text[i] + '\n' )
                
                
    def report_file_f2s(self, filename, common_text, distribution0_f2s, distribution1, name0 = 'fileR', name1 = 'fileA'):
        #format where each file relevant to distribution0_f2s occupies one ro
        len_max = np.max([len(x) for _, x in distribution0_f2s.items()])
        
        with open(self.res_folder+'/'+filename, 'w') as f:
            f.write(name0 +("\ttext_idx\ttext_common\t"+name1)*len_max + '\n')

            for filename, v_list in distribution0_f2s.items():
                f.write(filename)
                for v in v_list:
                    f.write('\t' + str(v)+'\t'+ common_text[v]+'\t' + '||'.join(distribution1[v]) )
                f.write('\n')
                
                
    def report_file_f2s_new(self, filename, common_text, distribution0_f2s, distribution1, name0 = 'fileR', name1 = 'fileA'):
        #another format: each row correpond to only one text in common,
        with open(self.res_folder+'/'+filename, 'w') as f:
            f.write(name0 +("\ttext_idx\ttext_common\t"+name1) + '\n')

            for filename, v_list in distribution0_f2s.items():
                for v in v_list:
                    f.write(filename+'\t' + str(v)+'\t'+ common_text[v]+'\t' + '||'.join(distribution1[v])+'\n' )
                
    
    def getK(self):
        n = len(self.df_details)
        if not np.all(self.df_details[str(n)+'gram']['freq_c1'] <= 5):
            return -1
            
        while(n>0):
            if(np.any(self.df_details[str(n)+'gram']['freq_c1'] >= 5)):#where the ngram in common with freq>=5 appears
                return n
            else:
                n = n-1
 
    
 
    
    
class CorpusCompareList:
    
    def __init__(self, compareReports, names, resFolder = 'res_report'):
        """
        compareReports: a list of CorpusCompareReport
        resFolder: usually the same resFolder as that of the CorpusCompare instance, the root folder of compareReports.res_folder
        """

        self.compareReports = compareReports if isinstance(compareReports, list) else [compareReports ]
        self.names = [names] if isinstance(names, str) else names
        assert(len(self.compareReports ) == len(self.names))
        self.klist = None
        self.resFolder = resFolder
       
        
    def add(self,compareRep, name):
        self.compareReports.append(compareRep)
        self.names.append(name)
        if(self.klist):
            self.klist.append(compareRep.getK())
        
    def get_cover_all(self, filename = 'coverage_all.tsv',store =True):
        res = pd.concat([self.compareReports[i].get_coverage(index = "cover_"+ self.names[i])
                        for i in range(len(self.compareReports))])
        
        self._show_cover(res)
        if store:
            filename = filename+'.tsv' if(filename[-4:]!='.tsv') else filename
            res.to_csv(self.resFolder+'/'+filename,sep = '\t')
        return res      

    def get_recouvrement(self,filename = 'recouvrement.tsv', store = True):
        """The recouvrement = len(common_ngram)/len(all_ngram_in_both_corpus)"""
        res = pd.concat([self.compareReports[i].get_normed_freq(index = "recouvr_"+ self.names[i])
                        for i in range(len(self.compareReports))])
        
        if store:
            filename = filename+'.tsv' if(filename[-4:]!='.tsv') else filename
            res.to_csv(self.resFolder+'/'+filename,sep = '\t')
        return res      
        
    def __set_klist(self):
        k_list = []
        for detail in self.compareReports:
            k_list.append(detail.getK())
        self.klist = k_list
        return k_list
    
    def get_klist(self):
        return self.klist if self.klist is not None else self.__set_klist()

    def __highlightK(self, col):
        assert(self.klist is not None)
        res = ['' for i in range(len(self.klist))]
        for i,k in enumerate(self.klist):
            if(col.name == str(k)+'gram'):
                res[i] = 'color: red'
        return res

    def _show_cover(self, cover_df):
        """highlight coverage cell from which k <=5 in the following cells"""
        if(self.klist is None):
            self.__set_klist()
        display(cover_df.style.apply(self.__highlightK))
                                #,subset=([str(k)+'gram' for k in self.klist])))
        
    def show_recouv_k5(self, k = 5, mode = 'max', img_name ='recouv_k', store = True):
        assert(mode in ['min', 'max'])
        title = '>' if mode == 'min' else '<='
        
        freq_klist = []
        if(mode == 'min'):
            for i, detail in enumerate(self.compareReports):
                len_all_i = detail.get_len_all()['len_all']
                detail_df = detail.get_detail()
                freq_klist.append(pd.DataFrame([len(detail_df[key][detail_df[key]['freq_c1']> k ])/len_all_i[key] 
                                                for key in sorted(detail_df.keys())], 
                                               index = [k for k in sorted(detail_df.keys())], columns = [self.names[i]]) )
        else:
            for i, detail in enumerate(self.compareReports):
                len_all_i = detail.get_len_all()['len_all']
                detail_df = detail.get_detail()
                freq_klist.append(pd.DataFrame([len(detail_df[key][detail_df[key]['freq_c1']<= k ])/len_all_i[key] 
                                                for key in sorted(detail_df.keys())], 
                                               index = [k for k in sorted(detail_df.keys())], columns = [self.names[i]]) )
        len_common2_k5 = pd.concat(freq_klist, axis = 1)
        len_common2_k5.plot(kind = 'bar',figsize= (15,6), title = f"normed frequency number of ngrams in common with k {title} {k}")
        if store:
            plt.savefig(self.resFolder+f'/{img_name}{title}{k}.png') #store the figure
        return len_common2_k5

    

    
    
#GLOBAL FUNCTIONS FOR ALL CLASSES  
def get_details_from_tsv(resFolder):
    """
    read the ngram in common (output of  CorpusCompare.ngram_info) 
    stored in a folder named 'ngram_common' located under resFolder named '{name_prefix}_{start_n}_to_{max_n}'
    """
    file_detail = os.listdir(resFolder+'/ngram_common')

    detail_dict = {}
    for fname in file_detail:
        key = fname[:-4]
        detail_dict[key] = pd.read_csv(resFolder+'/ngram_common'+f'/{key}.tsv', sep = '\t', index_col = 0).reset_index(drop = True)
    return detail_dict



def get_common_text_singleN(text_tofind, text_freq, n, filepath = 'text_common_n.txt', store = False):
    '''
    n: number of n for current ngram
    recovery common text from a list of ngram and their frequency
    '''
    #simple version to improve
    try:
        if isinstance(text_tofind, str):
            raise(TypeError)
        text_tofind = list(text_tofind) if not isinstance(text_tofind, list) else text_tofind
        text_freq = list(text_freq) if not isinstance(text_freq, list) else text_freq
        assert(len(text_tofind) == len(text_freq))
    except TypeError:
        print("TypeError: the first argument should be a list")
        return
    
    #print(len(text_tofind), len(text_freq))
    common_text = []

    while(text_tofind):
        current = text_tofind[0]
        to_remove = []
        
        text_freq[0] -=1
        if text_freq[0] == 0:
            to_remove.append(current)

        tokens = re.split(' ', current)
        common1 = ' '.join(tokens[1-n:])# current = a token + common1
        common2= ' '.join(tokens[:n-1])# current = common2+ a token
        
        #go through the ngram list to find the potential following part of current

        for idx, t in enumerate(text_tofind[1:]):
            #print('====t: ', idx, len(text_tofind))
            changed = False
            t_tokens = re.split(' ', t)
            may_common1, may_end = ' '.join(t_tokens[:n-1]), ' '.join(t_tokens[n-1:])#t.rsplit(' ', 1)
            may_head, may_common2 = ' '.join(t_tokens[:1-n]), ' '.join(t_tokens[1-n:])#re.split(' ', t, maxsplit = 1)
            
            if(common1 == may_common1):
                #print('find end', t)
                current = current+ ' ' + may_end
                changed = True
            if(common2 == may_common2):
                #print('find head', t)
                current = may_head+' ' + current
                changed = True

             
            if(changed):
                #print('update', idx+1)
                #remember the text used to concatenate,
                assert(text_freq[idx+1]>0)
                text_freq[idx+1] -= 1
                if text_freq[idx+1] == 0:
                    to_remove.append(t)
                
                #update current text in common 
                common2 = ' '.join( re.split(' ',current, maxsplit = n-1)[:n-1] ) #current = common2+ tokens
                common1 = ' '.join( current.rsplit(' ', n-1)[1-n:] )# current = tokens + common1
                
        common_text.append(current)
        
        #remove the text used to concatenate
        text_tofind = [t for t in text_tofind if t not in to_remove]
        #print(len(text_tofind), len(text_freq), to_remove)
        del to_remove
        text_freq = [i for i in text_freq if i > 0]
        #print(len(text_tofind), len(text_freq))
        assert(len(text_tofind) == len(text_freq))
        
    common_text = list(set(common_text))#list(set([x.lower() for x in common_text]))
    #print(len(common_text)) 
    if store:
        with open(filepath, 'w') as f:
            f.write('\n'.join(common_text))
    return common_text



