import itertools
import numpy as np
import pandas as pd
import math
from copy import deepcopy
import csv 
import yaml

with open("./config.yaml", 'r') as f:
    cfg = yaml.load(f)
papers_path  =  cfg['papers_path']
paper_sheet  =  cfg['paper_sheet']
reviewers_path  =  cfg['reviewers_path']
meta_reviewers_path  =  cfg['meta_reviewers_path']
subjects_track  =  cfg['subjects_track']
papers_path_t2  =  cfg['papers_path_t2']
paper_sheet_t2  =  cfg['paper_sheet_t2']
reviewers_path_t2  =  cfg['reviewers_path_t2']
meta_reviewers_path_t2  =  cfg['meta_reviewers_path_t2']
subjects_track_t2  =  cfg['subjects_track_t2']
papers_path_t3  =  cfg['papers_path_t3']
paper_sheet_t3  =  cfg['paper_sheet_t3']
reviewers_path_t3  =  cfg['reviewers_path_t3']
meta_reviewers_path_t3  =  cfg['meta_reviewers_path_t3']
subjects_track_t3  =  cfg['subjects_track_t3']
score_file  =  cfg['score_file']
acl_tpms_file  =  cfg['acl_tpms_file']
all_scores_file  =  cfg['all_scores_file']
info_path  =  cfg['info_path']
AC_path  =  cfg['AC_path']
AC_path_t2  =  cfg['AC_path_t2']
AC_path_t3  =  cfg['AC_path_t3']
boring_areas  =  cfg['boring_areas']
MAX_SUBJECT_LENGTH  =  cfg['MAX_SUBJECT_LENGTH']

def loadReviewerFiles(path,ac=False):
    email_col = 'Senior Meta-Reviewer Email' if ac else 'E-mail'
    with open(path,"r",encoding='utf-8') as f:
        reviewers = f.readlines()
        reviewers = [x.strip() for x in reviewers]
        reviewers=[x.split("\t") for x in reviewers]
        columns=reviewers[0]
        reviewers = pd.DataFrame(reviewers[1:],columns=columns)
        reviewers[email_col] = reviewers.apply(lambda row:row[email_col].lower(),axis=1)
    return reviewers

def loadPapersFiles(path,paper_sheet):
    papers = pd.read_excel(path,sheet_name=paper_sheet)
    papers=papers.rename(columns=dict(papers.loc[1]))[2:]
    return papers
    
def get_paper_reviewer_ids():
    #NOT USED ANYMORE
    #papers
    papers = loadPapersFiles(papers_path,paper_sheet)
    papers_t2 = loadPapersFiles(papers_path_t2,paper_sheet_t2)
    papers['node'] = list(range(0,len(papers)))
    papers_t2['node'] = list(range(len(papers),len(papers)+len(papers_t2)))
    #map
    papers_t2_nodes = dict(zip(papers_t2['Paper ID'].astype(str),papers_t2['node']))
    papers_nodes = dict(zip(papers['Paper ID'].astype(str),papers['node']))
    papers_t2_nodes.update(papers_nodes)

    #reviewers
    reviewers = loadReviewerFiles(reviewers_path)
    reviewers_t2 = loadReviewerFiles(reviewers_path_t2)
    meta_reviewers = loadReviewerFiles(meta_reviewers_path)
    meta_reviewers_t2 = loadReviewerFiles(meta_reviewers_path_t2)
    reviewers['node'] = list(range(0,len(reviewers)))
    reviewers_t2['node'] = list(range(len(reviewers),len(reviewers)+len(reviewers_t2)))
    meta_reviewers['node'] = list(range(len(reviewers)+len(reviewers_t2), len(reviewers)+len(reviewers_t2) + len(meta_reviewers)))
    meta_reviewers_t2['node'] = list(range(len(reviewers)+len(reviewers_t2) + len(meta_reviewers), len(reviewers)+len(reviewers_t2) + len(meta_reviewers) + len(meta_reviewers_t2)))
    #map
    reviewers_nodes = dict(zip(reviewers['E-mail'],reviewers['node']))
    reviewers_t2_nodes = dict(zip(reviewers_t2['E-mail'],reviewers_t2['node']))
    meta_reviewers_nodes = dict(zip(meta_reviewers['E-mail'],meta_reviewers['node']))
    meta_reviewers_t2_nodes = dict(zip(meta_reviewers_t2['E-mail'],meta_reviewers_t2['node']))
    meta_reviewers_t2_nodes.update(meta_reviewers_nodes)
    meta_reviewers_t2_nodes.update(reviewers_t2_nodes)
    meta_reviewers_t2_nodes.update(reviewers_nodes)

    print("#Papers:",len(papers))
    print("#Reviewers:",len(reviewers))
    print("#Meta Reviewers:",len(meta_reviewers))

    print("#Papers (Track 2):",len(papers_t2))
    print("#Reviewers (Track 2):",len(reviewers_t2))
    print("#Meta Reviewers (Track 2):",len(meta_reviewers_t2))

    return papers_t2_nodes, meta_reviewers_t2_nodes

def insert_subject_scores_in_matrix(path, score_matrix, paper_nodes, reviewer_nodes):
    with open(path,"r",encoding='utf-8') as f:
        data = f.read().splitlines()
        data=[x.split("\t") for x in data]
        for row in data:
            i,j,score = row
            if i in paper_nodes and j in reviewer_nodes:
                score_matrix[paper_nodes[i],reviewer_nodes[j],2]=float(score)
    
    print("Read: ",path.split("/")[-1],"\n#lines=", len(data),"\nSize=", score_matrix.shape)
    return score_matrix

def fetch_other_scores_from_file(path):
    with open(path,"r",encoding='utf-8') as f:
        data = f.read().splitlines()
        data=[x.split("\t") for x in data]
        
        #get mapping
        data_ = deepcopy(np.array(data))
        papers = list(set(data_[:,0]))
        reviewers = list(set(data_[:,1]))
        papers.sort()
        reviewers.sort()
        paper_dict = dict(zip(papers, list(range(len(papers)))))
        reviewers_dict = dict(zip(reviewers, list(range(len(reviewers)))))
        
        
        score_matrix = np.zeros((len(paper_dict),len(reviewers_dict),3))
        for row in data:
            i,j,acl_score,tpms_score = row
            score_matrix[paper_dict[i],reviewers_dict[j],:]=[float(acl_score),float(tpms_score),0]
    
    print("Read: ",path.split("/")[-1],"\n#lines=", len(data),"\nSize=", score_matrix.shape)
    return score_matrix, (paper_dict, reviewers_dict)

def read_all_scores():
    score_matrix, (paper_dict, reviewers_dict) = fetch_other_scores_from_file(acl_tpms_file)
    score_matrix = insert_subject_scores_in_matrix(score_file, score_matrix, paper_dict, reviewers_dict)
    
    #return subject score, tpms score, acl score
    return score_matrix[:,:,2], score_matrix[:,:,1], score_matrix[:,:,0], (paper_dict, reviewers_dict)


def write_final_scores(scores, reviewers_index, paper_index, score_file,mode="w"):
    inv_reviewers_index = {v: k for k, v in reviewers_index.items()}
    inv_paper_index = {v: k for k, v in paper_index.items()}
    with open(score_file,mode) as f:
        csvwriter = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE) 
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]): 
                csvwriter.writerow([str(inv_paper_index[i]),inv_reviewers_index[j],scores[i][j][0],scores[i][j][1],scores[i][j][2]])
                
def combine_all_scores_and_write():
    score_matrix, (paper_dict, reviewers_dict) = fetch_other_scores_from_file(acl_tpms_file)
    score_matrix = insert_subject_scores_in_matrix(score_file, score_matrix, paper_dict, reviewers_dict)
    write_final_scores(score_matrix, reviewers_dict, paper_dict, all_scores_file,mode="w")


def get_AC_data(path):
    AC_data = loadReviewerFiles( path,True)
    # lastCol_ = list(AC_data.columns)[-1]
    # lastCol = lastCol_.split("\n")[0]
    # AC_data[lastCol] = AC_data.apply(lambda r:r[lastCol_].split("\n")[0],axis=1)
    # del AC_data[lastCol_]
    
    AC_data = AC_data.rename(columns={"Senior Meta-Reviewer Email":"E-mail"})
    AC_data_primary = AC_data[AC_data['Is Primary']=="Yes"][['E-mail', 'Subject Area']]
    AC_data_primary = AC_data_primary.rename(columns={"Subject Area":"Primary Subject Area"})
    AC_data_secondary = AC_data[AC_data['Is Primary']!="Yes"][['E-mail', 'Subject Area']]
    AC_data_secondary['Subject Area'] = AC_data_secondary.groupby(['E-mail'])['Subject Area'].transform(lambda x: ';'.join(x))
    AC_data_secondary = AC_data_secondary.drop_duplicates()
    AC_data_secondary = AC_data_secondary.rename(columns={"Subject Area":"Secondary Subject Areas"})
    AC_data_merged = AC_data_primary.merge(AC_data_secondary,how='left',on=['E-mail'])
    AC_data_merged = AC_data_merged.fillna('')
    AC_data_merged['First Name'] = ""
    AC_data_merged['Middle Initial (optional)'] = ""
    AC_data_merged['Last Name'] = ""
    AC_data_merged['Organization'] = ""
    AC_data_merged = AC_data_merged[['First Name', 'Middle Initial (optional)', 'Last Name', 'E-mail',
       'Organization', 'Primary Subject Area', 'Secondary Subject Areas']]
    
    return AC_data_merged


if __name__ == '__main__':
    combine_all_scores_and_write()