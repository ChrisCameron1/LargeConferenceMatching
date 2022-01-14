#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import yaml
import numpy as np
import argparse
import os


def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str)
    return parser


def get_args(arg_str=None):
    parser = set_parser()
    if arg_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(arg_str.split())
    #   
    print(args)

    return args



def read_tpms_scores(tpms_file_list):
#read tpms scores and acl scores separately
#read tpms scores:
    tpms_list = []
    for this_file in tpms_file_list:
        print('read tpms scores from : {}'.format(this_file))
        role,track = os.path.basename(this_file).split('.')[0].split('_')
        tpms = pd.read_csv(this_file)
        tpms.columns = ['pid','rmail','tpms']
        tpms['track'] = txt2track[track]
        tpms['role'] = role
        tpms_list.append(tpms)
    # 
    tpms = pd.concat(tpms_list)
    tpms = tpms.reset_index(drop=True)
    tpms['rmail'] = tpms['rmail'].apply(lambda x: x.strip().lower())
    gby = ['pid','rmail']
    tpms = tpms.groupby(gby).tail(1)
    tpms.set_index(['pid','rmail'],inplace=True)
    return tpms

def read_subject_area_scores(sa_file_list):
#read subject area scores
    sa_list = []
    for sa_file in sa_file_list:
        print('read sa scores from : {}'.format(this_file))
        sa_list.append(pd.read_csv(sa_file,sep='\t',header=None))
    #
    sa = pd.concat(sa_list)
    #sa.columns = ['pid','rmail','k']
    sa.columns = ['pid','rmail','k','ind']
    sa['rmail'] = sa['rmail'].apply(lambda x: x.strip().lower())
    sa = sa.reset_index(drop=True)
    del sa['ind']
    gby = ['pid','rmail']
    sa = sa.groupby(gby).tail(1)
    sa.set_index(['pid','rmail'],inplace=True)
    return sa

def read_acl_scores(acl_file_list):
    acl_list  = []
    for acl_file in acl_file_list:
        print('read acl scores from : {}'.format(this_file))
        acl_list.append(pd.read_csv(acl_file,header=None))
        
    acl = pd.concat(acl_list)
    
    acl.columns = ['pid','rmail','acl']
    acl['rmail'] = acl['rmail'].apply(lambda x: x.strip().lower())
    acl = acl.reset_index(drop=True)
    gby = ['pid','rmail']
    acl = acl.groupby(gby).tail(1)
    acl.set_index(['pid','rmail'],inplace=True)
    return acl


def normalize_score(all_scores, col_name,a1, b1, y_cutoff): 
    # #####  Fit y = max(0, a1x + b1) till y_cutoff. Beyond y_cut_off, fit a linear curve between (x_cutoff, y_cutoff) and (x_max, 1). 
    # ##### Here x_cutoff = (y_cutoff - b1)/a1
   
    print(" normalize ", col_name)
    coefs = {}
    cutoff = (y_cutoff - b1)/a1
    max_score = all_scores[col_name].max()
    
    a2,b2 = a1,b1
    if y_cutoff != 1:
        a2 = (1 - y_cutoff)/(max_score - cutoff)
        b2 = y_cutoff - a2*cutoff
    
    coefs[col_name] = [(a1,b1,cutoff),(a2,b2)]
    #print(coefs)
    #col_name+'0' : first linear part: (a1x + b1)
    all_scores[col_name+'0'] = coefs[col_name][0][0]*all_scores[col_name] + coefs[col_name][0][1]
   
    #'n'+col_name: 2nd linear part: (a2x + b2). Will also contain the final normalized score.
    all_scores['n'+col_name] = coefs[col_name][1][0]*all_scores[col_name] + coefs[col_name][1][1]
  
    is_x_less_than_cutoff = all_scores[col_name] <= coefs[col_name][0][2]
    all_scores.loc[is_x_less_than_cutoff,'n'+col_name] = all_scores.loc[is_x_less_than_cutoff,col_name+'0']
   
    #norm_score = max(0, norm_score) 
    all_scores.loc[all_scores['n'+col_name] <=0,'n'+col_name] = 0.0
    return all_scores


if __name__ == '__main__':
    args = get_args()
    config_file = args.config
    with open(config_file,'rb') as fh:
        config = yaml.safe_load(fh)
    
    sa_file_list = config['sa_file_list']#['data/scores/20201014/subject_scores.txt']
    tpms_file_list = config['tpms_file_list']
    acl_file_list = config['acl_file_list'] #['data/scores/20201014/acl_scores_for_ilp_phase_2.csv']

    output_file =config['output_file'] # 'data/scores/20201014/normalized_raw_scores_20201014_v2.tsv'

    txt2track= {'AAAI': 'AAAI', 'AISI': 'AISI', 'FT': 'AAAI'}
    
    tpms = read_tpms_scores(tpms_file_list)
    sa = read_subject_area_scores(sa_file_list)
    acl = read_acl_scores(acl_file_list)

    all_scores = sa.join([acl,tpms],how='outer')

    # ### FOR ACL
    a1,b1,y_cutoff = 2.22,-0.22,0.94
    # #####  Fit y = max(0, a1x + b1) till y_cutoff. Beyond y_cut_off, fit a linear curve between (x_cutoff, y_cutoff) and (x_max, 1). 
    # ##### Here x_cutoff = (y_cutoff - b1)/a1
    all_scores = normalize_score(all_scores, 'acl', a1, b1, y_cutoff) 

    # ### FOR TPMS
    a1,b1,y_cutoff = 3.65,-2.65,1
    all_scores = normalize_score(all_scores, 'tpms', a1, b1, y_cutoff)

    # ### FOR SUBJECT AREA SCORES
    a1,b1,y_cutoff = 1.35,-0.35,1
    all_scores = normalize_score(all_scores, 'k', a1, b1, y_cutoff)
    #
    all_scores = all_scores.reset_index()
    all_scores[['pid','rmail','ntpms','nacl','nk','tpms','acl','k']].to_csv(output_file,sep='\t',index=False)

