#initialize
import json
import itertools
import logging
import pandas as pd
import numpy as np
from copy import deepcopy
import yaml
from A4_helper import *
from A4_dblp_helper import *

# load config file
config_file = 'config.yaml'
with open(config_file, 'r') as f:
        cfg = yaml.load(f)

papers_path = cfg['papers_path']
paper_sheet = cfg['paper_sheet']
info_path = cfg['info_path']
users_path = cfg['users_path']
score_file = cfg['score_file']
output_info_file = cfg['output_info_file']
logname = cfg['logname']
popular = cfg['popular']

#reviewers
reviewers_path = cfg['reviewers_path']
meta_reviewers_path = cfg['meta_reviewers_path']
reviewers_path_t2 = cfg['reviewers_path_t2']
meta_reviewers_path_t2 = cfg['meta_reviewers_path_t2']
AC_path = cfg['AC_path']
AC_path_t2 = cfg['AC_path_t2']

MAX_SUBJECT_LENGTH = cfg['MAX_SUBJECT_LENGTH'] #CMT files seem to have that

#load data
import pandas as pd
papers = pd.read_excel(papers_path,sheet_name=paper_sheet)
info = pd.read_excel(info_path)
country_info = loadReviewerFiles(users_path)
papers=papers.rename(columns=dict(papers.loc[1]))[2:]
reviewers = loadReviewerFiles(reviewers_path)
reviewers_t2 = loadReviewerFiles(reviewers_path_t2)
meta_reviewers = loadReviewerFiles(meta_reviewers_path)
meta_reviewers_t2 = loadReviewerFiles(meta_reviewers_path_t2)
AC_data = get_AC_data(AC_path)
AC_data_t2 = get_AC_data(AC_path_t2)

with open(logname,"w") as f:
    f.write("logging info population")
logging.basicConfig(filename=logname,
                            filemode='a',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

print("#Papers:",len(papers))
print("#Users:",len(info))
print("#Reviewers:",len(reviewers))
print("#Meta Reviewers:",len(meta_reviewers))
print("#Reviewers (Track 2):",len(reviewers_t2))
print("#Meta Reviewers (Track 2):",len(meta_reviewers_t2))
print("#AC Reviewers:",len(AC_data))
print("#AC Reviewers (Track 2):",len(AC_data_t2))


#specific to the data we have
q10_column = 'Q10 (How many times have you been part of the program committee (as PC/SPC/AC, etc) of AAAI, IJCAI, NeurIPS, ACL, SIGIR, WWW, RSS, NAACL, KDD, IROS, ICRA, ICML, ICCV, EMNLP, EC, CVPR, AAMAS, HCOMP, HRI, ICAPS, ICDM, ICLR, ICWSM, IUI, KR, SAT, WSDM, UAI, AISTATS, COLT, CORL, CP, CPAIOR, ECAI, OR ECML in the past?)'
columns_of_use = ['Email', 'Publication Emails', 'Conflict Domains', 'Primary Subject Area',\
                  'First Name','Last Name','DBLP URL',q10_column]
info = info[columns_of_use]

#left join with reviewers to limit to the universe of reviewers
reviewers = reviewers.append(meta_reviewers,ignore_index=True)
reviewers = reviewers.append(AC_data,ignore_index=True)
reviewers = reviewers.drop_duplicates()

#get track2 reviewers
t1_reviewers = list(set(reviewers['E-mail']))
reviewers_t2 = reviewers_t2.append(meta_reviewers_t2,ignore_index=True)
reviewers_t2 = reviewers_t2.append(AC_data_t2,ignore_index=True)
reviewers_t2 = reviewers_t2[~reviewers_t2['E-mail'].isin(t1_reviewers)]
reviewers = reviewers.append(reviewers_t2,ignore_index=True)

a=reviewers.merge(info,how='left',left_on=['E-mail'],right_on=['Email'])

a['Primary Subject Area'] = a.apply(lambda row:row['Primary Subject Area_y'] \
                                    if type(row['Primary Subject Area_y'])!=float \
                                    else row['Primary Subject Area_x'],axis=1)
a[q10_column] = a.apply(lambda row:row[q10_column] if type(row[q10_column])!=float else 'This is my first time reviewing for any conference.',axis=1)
a = a.rename(columns={'First Name_x':'First Name','Last Name_x':'Last Name'})
a['Email'] = a['E-mail']
info = deepcopy(a[columns_of_use])
print(len(reviewers))

#populate country and region
logger = logging.getLogger('extract-Country')
country_info = country_info[['E-mail','Country']]
info=info.merge(country_info,how='left',left_on=['Email'],right_on=['E-mail'])
info['Country'] = info['Country'].fillna('')
def get_region(r):
    for k,v in country_set.items():
        if r['Country'] in v:
            return k
    return 'Unknown' 
info['Region']=info.apply(lambda row:get_region(row),axis=1)

#popular papers
print("Popular areas are: ", popular)
info = populate_if_popular(info,popular)

#add seniority info
q10_map = {'This is my first time reviewing for any of the listed conferences, but I have reviewed for other conferences which are not listed above.' : 0,
           'more than 10':10,
           '1-3':2,
           '4-10':7,
           'This is my first time reviewing for any conference.':0}
info['n_Reviewed'] = info.apply(lambda row:q10_map[row[q10_column]],axis=1)
del info[q10_column]
logger = logging.getLogger('extract-DBLP')
info = get_dblp_info(info,logger)

info['S1'] = info.apply(lambda row: 1 if row['n_Reviewed']>=3 or row['n_DBLP']>10 else 0,axis=1)
info['S2'] = info.apply(lambda row: 1 if row['n_Reviewed']>=3 or row['n_DBLP']>4 else 0,axis=1)
info['S3'] = info.apply(lambda row: 1 if row['n_Reviewed']>=1 or row['n_DBLP']>2 else 0,axis=1)

#save data
info_out = info[['Email', 'Country', 'Region', 'Popular', 'S1', 'S2', 'S3']]
info_out[:5].to_excel(output_info_file,index=False)