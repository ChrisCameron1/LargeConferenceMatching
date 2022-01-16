#!/usr/bin/env python
# coding: utf-8

# In[1]:


#initialize
import json
import itertools
import pandas as pd
import numpy as np
from copy import deepcopy
import sys
sys.path.insert(0,'scripts')
from scores_helper import *
from A4_helper import *


# #### 1. Load Input
# 
# papers, reviewers and user information

# In[2]:
papers = loadPapersFiles(papers_path,paper_sheet)
papers_t2 = loadPapersFiles(papers_path_t2,paper_sheet_t2)
papers_t3 = loadPapersFiles(papers_path_t3,paper_sheet_t3)

reviewers = loadReviewerFiles(reviewers_path)
reviewers_t2 = loadReviewerFiles(reviewers_path_t2)
reviewers_t3 = loadReviewerFiles(reviewers_path_t3)

meta_reviewers = loadReviewerFiles(meta_reviewers_path)
meta_reviewers_t2 = loadReviewerFiles(meta_reviewers_path_t2)
meta_reviewers_t3 = loadReviewerFiles(meta_reviewers_path_t3)

AC_data = get_AC_data(AC_path)
AC_data_t2 = get_AC_data(AC_path_t2)
AC_data_t3 = get_AC_data(AC_path_t3)


subject_areas = parseSubjectAreas(subjects_track)
subject_areas = list(set(subject_areas))
subject_areas_t2 = parseSubjectAreas(subjects_track_t2)
subject_areas_t2 = list(set(subject_areas_t2))
subject_areas_t3 = deepcopy(subject_areas)

info = pd.read_excel(info_path)

print("#Papers:",len(papers))
print("#Reviewers:",len(reviewers))
print("#Meta Reviewers:",len(meta_reviewers))
print("#AC Reviewers:",len(AC_data))

print("#Papers (Track 2):",len(papers_t2))
print("#Reviewers (Track 2):",len(reviewers_t2))
print("#Meta Reviewers (Track 2):",len(meta_reviewers_t2))
print("#AC Reviewers (Track 2):",len(AC_data_t2))

print("#Papers (Track 3):",len(papers_t3))
print("#Reviewers (Track 3):",len(reviewers_t3))
print("#Meta Reviewers (Track 3):",len(meta_reviewers_t3))
print("#AC Reviewers (Track 3):",len(AC_data_t3))

print("#Info:",len(info))


# In[3]:


def remove_reviewers_with_empty_areas(reviewers_t2):
    temp_rev = deepcopy(reviewers_t2)
    n_rows = (len(temp_rev))
    temp_rev = temp_rev.drop(temp_rev[(temp_rev['Primary Subject Area'] == '') & (temp_rev['Secondary Subject Areas'] == '')].index)
    print( "Rows Removed:",n_rows-len(temp_rev))
    return temp_rev

reviewers = remove_reviewers_with_empty_areas(reviewers)
reviewers_t2 = remove_reviewers_with_empty_areas(reviewers_t2)
reviewers_t3 = remove_reviewers_with_empty_areas(reviewers_t3)
meta_reviewers = remove_reviewers_with_empty_areas(meta_reviewers)
meta_reviewers_t2 = remove_reviewers_with_empty_areas(meta_reviewers_t2)
meta_reviewers_t3 = remove_reviewers_with_empty_areas(meta_reviewers_t3)
AC_data = remove_reviewers_with_empty_areas(AC_data)
AC_data_t2 = remove_reviewers_with_empty_areas(AC_data_t2)
AC_data_t3 = remove_reviewers_with_empty_areas(AC_data_t3)


# In[4]:

boring_areas = [x.strip() for x in subject_areas if len([y for y in boring_areas if y in x])>0]
boring_areas = [x.strip() for x in subject_areas_t3 if len([y for y in boring_areas if y in x])>0]


# In[7]:


#PROCESS track3 Papers file
papers_t3 = papers_t3[papers_t3['Status']=='Awaiting Decision']
papers_t2 = papers_t2[papers_t2['Status']=='Awaiting Decision']
papers = papers[papers['Status']=='Awaiting Decision']
# papers_t3 = papers_t3[papers_t3['Paper ID']!=9227] ##remove invalid entries
papers_t3 = papers_t3[papers_t3['Number of Files']>=1]
papers_t3['Primary Subject Area']=papers_t3.apply(lambda row: row['Primary Subject Area'] 
                                                  if type(row['Primary Subject Area'] ) != float 
                                                  and '->' in  row['Primary Subject Area']  
                                                  else np.nan,axis=1)
def handle_secondary_t3(row):
    if type(row['Secondary Subject Areas'])==float:
        return row['Secondary Subject Areas']
    sas = row['Secondary Subject Areas'].split(";")
    sas = [x.strip() for x in sas if '->' in x]
    if len(sas)==0:
        return np.nan
    return ";".join(sas)
papers_t3['Secondary Subject Areas']=papers_t3.apply(handle_secondary_t3,axis=1)
# papers_t3['Primary Subject Area']=papers_t3.apply(lambda row: "Computer Vision (CV) -> CV: Other Foundations of Computer Vision" 
#                                      if row['Paper ID']==9852 else row['Primary Subject Area'],axis=1) #another manual fix


# ### 2 Get Paper primary and secondary vectors

# In[8]:


def get_paper_vecs(subject_areas, papers, boring_areas):
    sorted_subject_areas = deepcopy(subject_areas)
    sorted_subject_areas.sort()
    subject_index = dict(zip(sorted_subject_areas,list(range(len(subject_areas)))))
    paper_v1, paper_v2 = get_paper_vectors(papers, subject_index, MAX_SUBJECT_LENGTH, boring_areas)
    return (paper_v1, paper_v2), subject_index

(paper_v1, paper_v2), subject_index = get_paper_vecs(subject_areas, papers, boring_areas)
(paper_v1_t2, paper_v2_t2), subject_index_t2 = get_paper_vecs(subject_areas_t2, papers_t2, boring_areas)
(paper_v1_t3, paper_v2_t3), subject_index_t3 = get_paper_vecs(subject_areas_t3, papers_t3, boring_areas)


# ### 2.2 Calculate PMI or P(s1/s2)

# In[9]:


def get_pmi_and_p(reviewers,meta_reviewers,papers,subject_areas,subject_index,MAX_SUBJECT_LENGTH,track2=False):
    reviewers_ = deepcopy(reviewers[['E-mail','Primary Subject Area', 'Secondary Subject Areas']])
    meta_reviewers_ = deepcopy(meta_reviewers[['E-mail','Primary Subject Area', 'Secondary Subject Areas']])
    papers_ = deepcopy(papers[['Paper ID', 'Primary Subject Area', 'Secondary Subject Areas']])
    papers_ = papers_.rename(columns={'Paper ID':'E-mail'})
    combined = reviewers_.append(meta_reviewers_,ignore_index=True)
    combined = combined.append(papers_,ignore_index=True)
    frequencies, pair_frequencies = get_subject_frequencies_rev(deepcopy(combined),deepcopy(subject_areas), MAX_SUBJECT_LENGTH,track2)
    pmi_matrix, p_matrix, (max_pmi, max_p) = get_pmi_matrix(deepcopy(combined), deepcopy(subject_areas), deepcopy(subject_index), frequencies, pair_frequencies, MAX_SUBJECT_LENGTH)
    #test_pmi_randomly(frequencies, pair_frequencies, subject_index, pmi_matrix, papers)
    return frequencies, pair_frequencies, pmi_matrix, p_matrix, (max_pmi, max_p)

frequencies, pair_frequencies, pmi_matrix, p_matrix, (max_pmi, max_p) = get_pmi_and_p(reviewers,meta_reviewers,papers,subject_areas,subject_index,MAX_SUBJECT_LENGTH)
frequencies_t2, pair_frequencies_t2, pmi_matrix_t2, p_matrix_t2, (max_pmi_t2, max_p_t2) = get_pmi_and_p(reviewers_t2,meta_reviewers_t2,papers_t2,subject_areas_t2,subject_index_t2,MAX_SUBJECT_LENGTH,True)
frequencies_t3, pair_frequencies_t3, pmi_matrix_t3, p_matrix_t3, (max_pmi_t3, max_p_t3) = get_pmi_and_p(reviewers_t3,meta_reviewers_t3,papers_t3,subject_areas_t3,subject_index_t3,MAX_SUBJECT_LENGTH)


# ### 2.3 Get Reviewer Paper-subjects

# In[10]:


#check_author_emails_in_papers(papers)
author_subject_papers = get_author_paper_subjects(deepcopy(papers), MAX_SUBJECT_LENGTH)

#check_author_emails_in_papers(papers_t2)
author_subject_papers_t2 = get_author_paper_subjects(deepcopy(papers_t2), MAX_SUBJECT_LENGTH)

#check_author_emails_in_papers(papers_t3)
author_subject_papers_t3 = get_author_paper_subjects(deepcopy(papers_t3), MAX_SUBJECT_LENGTH)


# ### 2.4 Calculate vector for reviewers

# In[11]:


reviewer_r = get_reviewer_vectors(deepcopy(reviewers), deepcopy(subject_index), author_subject_papers, p_matrix, MAX_SUBJECT_LENGTH)
reviewer_r_t2 = get_reviewer_vectors(deepcopy(reviewers_t2), deepcopy(subject_index_t2), author_subject_papers_t2, p_matrix_t2, MAX_SUBJECT_LENGTH, track2=True)
reviewer_r_t3 = get_reviewer_vectors(deepcopy(reviewers_t3), deepcopy(subject_index_t3), author_subject_papers_t3, p_matrix_t3, MAX_SUBJECT_LENGTH)

meta_reviewer_r = get_reviewer_vectors(deepcopy(meta_reviewers), deepcopy(subject_index), author_subject_papers, p_matrix, MAX_SUBJECT_LENGTH)
meta_reviewer_r_t2 = get_reviewer_vectors(deepcopy(meta_reviewers_t2), deepcopy(subject_index_t2), author_subject_papers_t2, p_matrix_t2, MAX_SUBJECT_LENGTH, track2=True)
meta_reviewer_r_t3 = get_reviewer_vectors(deepcopy(meta_reviewers_t3), deepcopy(subject_index_t3), author_subject_papers_t3, p_matrix_t3, MAX_SUBJECT_LENGTH)

AC_reviewer_r = get_reviewer_vectors(deepcopy(AC_data), deepcopy(subject_index), author_subject_papers, p_matrix, MAX_SUBJECT_LENGTH)
AC_reviewer_r_t2 = get_reviewer_vectors(deepcopy(AC_data_t2), deepcopy(subject_index_t2), author_subject_papers_t2, p_matrix_t2, MAX_SUBJECT_LENGTH, track2=True)
AC_reviewer_r_t3 = get_reviewer_vectors(deepcopy(AC_data_t3), deepcopy(subject_index_t3), author_subject_papers_t3, p_matrix_t3, MAX_SUBJECT_LENGTH)


# ## MISC: Get L1 and rule set

# In[12]:


paper_l1_dict, paper_rule2_dict = enrich_with_l1_and_rule2(deepcopy(papers),'Paper ID')
paper_l1_dict_t2, paper_rule2_dict_t2 = enrich_with_l1_and_rule2(deepcopy(papers_t2),'Paper ID',True)
paper_l1_dict_t3, paper_rule2_dict_t3 = enrich_with_l1_and_rule2(deepcopy(papers_t3),'Paper ID',True)

rev_l1_dict, rev_rule2_dict = enrich_with_l1_and_rule2(deepcopy(reviewers), 'E-mail')
rev_l1_dict_t2, rev_rule2_dict_t2 = enrich_with_l1_and_rule2(deepcopy(reviewers_t2), 'E-mail',True)
rev_l1_dict_t3, rev_rule2_dict_t3 = enrich_with_l1_and_rule2(deepcopy(reviewers_t3), 'E-mail',True)

meta_rev_l1_dict, meta_rev_rule2_dict = enrich_with_l1_and_rule2(deepcopy(meta_reviewers), 'E-mail')
meta_rev_l1_dict_t2, meta_rev_rule2_dict_t2 = enrich_with_l1_and_rule2(deepcopy(meta_reviewers_t2), 'E-mail',True)
meta_rev_l1_dict_t3, meta_rev_rule2_dict_t3 = enrich_with_l1_and_rule2(deepcopy(meta_reviewers_t3), 'E-mail')

AC_rev_l1_dict, AC_rev_rule2_dict = enrich_with_l1_and_rule2(deepcopy(AC_data), 'E-mail')
AC_rev_l1_dict_t2, AC_rev_rule2_dict_t2 = enrich_with_l1_and_rule2(deepcopy(AC_data_t2), 'E-mail',True)
AC_rev_l1_dict_t3, AC_rev_rule2_dict_t3 = enrich_with_l1_and_rule2(deepcopy(AC_data_t3), 'E-mail')


# ### 2.5 Calculate Reviewer Paper Score Matrix

# In[18]:


def calculate_subject_score(reviewer_r, paper_v1, paper_v2,paper_rule2_dict,rev_rule2_dict,paper_l1_dict,rev_l1_dict):
    reviewer_matrix, v1_matrix, v2_matrix, (reviewers_index, paper_index) = convert_to_matrices(reviewer_r, paper_v1, paper_v2)
    rule2_matrix = make_rule2_matrix(paper_index,reviewers_index,paper_rule2_dict,rev_rule2_dict)
    scores = get_scores(reviewer_matrix, v1_matrix, v2_matrix)
    scores = np.nan_to_num(scores)
    scores = np.multiply(scores,1-rule2_matrix)+np.multiply(rule2_matrix,np.minimum(0.31,scores))
    
    overlap_matrix = get_overlap_matrix(paper_index,reviewers_index,paper_rule2_dict,rev_rule2_dict)
    
    return scores, (reviewers_index, paper_index), overlap_matrix
print("#Calculating Scores for Track1-PC#")
scores, (reviewers_index, paper_index), overlap_matrix = calculate_subject_score(reviewer_r, paper_v1, paper_v2,
                                                                 paper_rule2_dict,rev_rule2_dict,
                                                                                 paper_l1_dict,rev_l1_dict)
print("#Calculating Scores for Track2-PC#")
scores_t2, (reviewers_index_t2, paper_index_t2), overlap_matrix_t2 = calculate_subject_score(reviewer_r_t2, paper_v1_t2, paper_v2_t2,
                                                                         paper_rule2_dict_t2,rev_rule2_dict_t2,
                                                                            paper_l1_dict_t2,rev_l1_dict_t2)
print("#Calculating Scores for Track3-PC#")
scores_t3, (reviewers_index_t3, paper_index_t3), overlap_matrix_t3 = calculate_subject_score(reviewer_r_t3, paper_v1_t3, paper_v2_t3,
                                                                         paper_rule2_dict_t3,rev_rule2_dict_t3,
                                                                            paper_l1_dict_t3,rev_l1_dict_t3)


print("#Calculating Scores for Track1-SPC#")
meta_scores, (meta_reviewers_index, meta_paper_index), meta_overlap_matrix = calculate_subject_score(meta_reviewer_r, paper_v1, paper_v2,
                                                                               paper_rule2_dict,meta_rev_rule2_dict,
                                                                                paper_l1_dict,meta_rev_l1_dict)
print("#Calculating Scores for Track2-SPC#")
meta_scores_t2, (meta_reviewers_index_t2, meta_paper_index_t2), meta_overlap_matrix_t2 = calculate_subject_score(meta_reviewer_r_t2, 
                                                                                         paper_v1_t2, paper_v2_t2,
                                                                                        paper_rule2_dict_t2,
                                                                                         meta_rev_rule2_dict_t2,
                                                                                        paper_l1_dict_t2,meta_rev_l1_dict_t2)

print("#Calculating Scores for Track3-SPC#")
meta_scores_t3, (meta_reviewers_index_t3, meta_paper_index_t3), meta_overlap_matrix_t3 = calculate_subject_score(meta_reviewer_r_t3, 
                                                                                         paper_v1_t3, paper_v2_t3,
                                                                                        paper_rule2_dict_t3,
                                                                                         meta_rev_rule2_dict_t3,
                                                                                        paper_l1_dict_t3,meta_rev_l1_dict_t3)



print("#Calculating Scores for Track1-AC#")
AC_scores, (AC_reviewers_index, AC_paper_index), AC_overlap_matrix = calculate_subject_score(AC_reviewer_r, paper_v1, paper_v2,
                                                                         paper_rule2_dict,AC_rev_rule2_dict,
                                                                        paper_l1_dict,AC_rev_l1_dict)
print("#Calculating Scores for Track2-AC#")
AC_scores_t2, (AC_reviewers_index_t2, AC_paper_index_t2), AC_overlap_matrix_t2 = calculate_subject_score(AC_reviewer_r_t2, paper_v1_t2, 
                                                                                   paper_v2_t2,
                                                                                  paper_rule2_dict_t2,
                                                                                   AC_rev_rule2_dict_t2,
                                                                                    paper_l1_dict_t2,AC_rev_l1_dict_t2)

print("#Calculating Scores for Track3-AC#")
AC_scores_t3, (AC_reviewers_index_t3, AC_paper_index_t3), AC_overlap_matrix_t3 = calculate_subject_score(AC_reviewer_r_t3, paper_v1_t3, 
                                                                                   paper_v2_t3,
                                                                                  paper_rule2_dict_t3,
                                                                                   AC_rev_rule2_dict_t3,
                                                                                    paper_l1_dict_t3,AC_rev_l1_dict_t3)


print("#Done#")


# #### 5. Write Score Matrix

# In[11]:



write_scores(scores, overlap_matrix, reviewers_index, paper_index, score_file, 'w')
write_scores(scores_t2, overlap_matrix_t2, reviewers_index_t2, paper_index_t2, score_file, 'a')
write_scores(scores_t3, overlap_matrix_t3, reviewers_index_t3, paper_index_t3, score_file , 'a')

write_scores(meta_scores, meta_overlap_matrix, meta_reviewers_index, meta_paper_index, score_file, 'a')
write_scores(meta_scores_t2, meta_overlap_matrix_t2, meta_reviewers_index_t2, meta_paper_index_t2, score_file, 'a')
write_scores(meta_scores_t3, meta_overlap_matrix_t3, meta_reviewers_index_t3, meta_paper_index_t3, score_file, 'a')

write_scores(AC_scores, AC_overlap_matrix, AC_reviewers_index, AC_paper_index, score_file, 'a')
write_scores(AC_scores_t2, AC_overlap_matrix_t2, AC_reviewers_index_t2, AC_paper_index_t2, score_file, 'a')
write_scores(AC_scores_t3, AC_overlap_matrix_t3, AC_reviewers_index_t3, AC_paper_index_t3, score_file, 'a')

