import json
import re
import itertools
import pandas as pd
import numpy as np
from copy import deepcopy
import random
import csv

country_set = {'Asia': ['China', 'Macau', 'Taiwan', 'Macau SAR', 'Macao SAR','Hong Kong', 'HK', 'India', 'Japan', 'Israel','Singapore', 'Pakistan', 'North Korea',
                         'South Korea', 'Korea','Korea, Republic of', 'Malaysia', 'Thailand', 'Indonesia', 'Iran', 
                         'Saudi Arabia', 'Qatar', "UAE", 'Russia','Bangladesh','Nepal','Hong Kong SAR', 'Vietnam',
                         'Sri Lanka','Philippines','United Arab Emirates','Lebanon',
                        
                        'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon', 'Cape Verde',
                         'Central African Republic', 'Chad', 'Camoros', 'Democratic Republic of the Congo', 
                         'Republic of the Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 'Ethiopia',
                         'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Ivory Coast', 'Kenya', 
                         'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 
                         'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 
                         'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 
                         'South Africa', 'South Sudan', 'Sudan', 'Swaziland', 'Tanzania', 'Togo', 
                         'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'
                        
                        ],
                'Australia' : ['Australia', 'New Zealand','New Caledonia','Fiji Islands'],
                'Europe' : ["Albania","Andorra",'Armenia','Austria','Azerbaijan','Belarus','Belgium',
                'Bosnia and Herzegovina','Bulgaria','Croatia','Cyprus','Czechia', 'Czech Republic', 'Denmark',
                'Estonia','Finland','France','Georgia','Germany','Greece','Hungary','Iceland',
                'Ireland','Italy','Kazakhstan','Kyrgyzstan','Latvia','Lithuania',
                'Luxembourg','Malta','Monaco','Montenegro','Netherlands','North Macedonia','Norway',
                'Poland','Portugal','Republic of Moldova','Romania','Russian Federation','San Marino',
                'Serbia','Slovakia','Slovenia','Spain','Sweden','Switzerland','Tajikistan',
                'Turkey','Turkmenistan','Ukraine','United Kingdom','UK','Great Britain','Uzbekistan','Liechtenstein'],
                'US' : ['US', 'USA', 'United States', 'United States of America', 'Canada',
                'Venezuela','Chile', 'Bolivarian Republic of','Mexico','Puerto Rico','Honduras','Barbados',
                       
                        'Brazil','Uruguay','Cuba','Colombia','Argentina']}

rule2 = {   "Cognitive Modeling & Cognitive Systems (CMS)":"T1",
    "Machine Learning (ML)":"T1",
    "Reasoning under Uncertainty (RU)":"T1",
    "Data Mining & Knowledge Management (DMKM)":"T2",
    "Speech & Natural Language Processing (SNLP)":"T3",
    "Computer Vision (CV)":"T4",
    "Intelligent Robots (ROB)":"T5",
    "Knowledge Representation and Reasoning (KRR)":"T6",
    "Constraint Satisfaction and Optimization (CSO)":"T7",
    "Search and Optimization (SO)":"T7",
    "Planning, Routing, and Scheduling (PRS)":"T7",
    "Multiagent Systems (MAS)":"T8",
    "Game Theory and Economic Paradigms (GTEP)":"T8",
}

inter_l1_mapping = {
"Machine Learning (ML)" : (["Cognitive Modeling & Cognitive Systems (CMS)", "Reasoning under Uncertainty (RU)"], 0.9),
"Humans and AI (HAI)" : (["Human-Computation and Crowd Sourcing (HCC)"],0.9),
"Human-Computation and Crowd Sourcing (HCC)" : (["Humans and AI (HAI)"],0.9),
"Constraint Satisfaction and Optimization (CSO)" : (["Search and Optimization (SO)"],0.8),
"Search and Optimization (SO)" : (["Constraint Satisfaction and Optimization (CSO)"],0.8),
"Multiagent Systems (MAS)" : (["Game Theory and Economic Paradigms (GTEP)"],0.8),
"Game Theory and Economic Paradigms (GTEP)" : (["Multiagent Systems (MAS)"],0.8)
}

def get_popular_papers(papers, threshold=0.08):
    ## LOGIC
    # fetch primary subject for each paper and get top level subject area name
    # convert into frequency table
    # calculate % from that
    # get papers which have a % greater than the threshold

    papers_primary = papers[['Paper ID', 'Primary Subject Area' ]]
    papers_primary['L1_subject'] = papers_primary.apply(lambda row: row['Primary Subject Area'].split("->")[0].strip(),axis = 1)
    freq = pd.value_counts(papers_primary['L1_subject']).to_frame().reset_index()
    freq['Percentage'] = freq.apply(lambda row: row['L1_subject']/len(papers_primary),axis=1)
    freq['Popular'] = freq.apply(lambda row: True if row['Percentage'] > threshold else False, axis=1)
    popular = list(freq[freq['Popular']==True]['index'])
    return popular

def populate_if_popular(info,popular):
    def indentify_popular(row):
        L1 = row['Primary Subject Area'].split("->")[0].strip()
        if L1 in popular:
            return True
        else:
            return False
    info['Popular'] = info.apply(indentify_popular,axis=1)
    return info


def parseSubjectAreas(filename):
  return [line.rstrip('\n') for line in open(filename)]

def get_paper_vectors(papers, subject_index, max_subject_length=100, boring_areas=[]):
    ## LOGIC
    # get the array of subjects for each paper
    # convert them to indices and make them 1 in the dummy vector
    # return a map

    papers_vec = papers[['Paper ID', 'Primary Subject Area', 'Secondary Subject Areas']]
    dim = len(subject_index)
    application_areas = [x.strip()[:max_subject_length].strip() for x in list(subject_index.keys()) if 'Application' in x]
    dummy = np.zeros(dim)
    paper_v1 = {}
    paper_v2 = {}
    for _,row in papers_vec.iterrows():
        primary = row['Primary Subject Area'].split(';') if type(row['Primary Subject Area'])!= float else []
        secondary = row['Secondary Subject Areas'].split(';') if type(row['Secondary Subject Areas'])!= float else []

        #process boring keywords
        for x in primary:
            if x in boring_areas:
                secondary.append(x)
                primary.remove(x)

        primary_idx = [subject_index[x.strip()[:max_subject_length].strip()]for x in primary]
        secondary_idx = [subject_index[x.strip()[:max_subject_length].strip()] for x in secondary]
        primary_vec = deepcopy(dummy)
        secondary_vec = deepcopy(dummy)
        if len([x for x in primary if x.strip()[:max_subject_length].strip() in application_areas]) > 0:
            primary_vec[primary_idx] = 0.75
        else:
            primary_vec[primary_idx] = 1.    
        secondary_vec[secondary_idx] = .5
        paper_v1[row['Paper ID']] = primary_vec
        paper_v2[row['Paper ID']] = secondary_vec
    return paper_v1, paper_v2

def get_subject_frequencies(papers,subject_areas,max_subject_length=100):
    papers_subs = papers[['Paper ID', 'Primary Subject Area', 'Secondary Subject Areas' ]]
    sorted_subject_areas = deepcopy(subject_areas)
    sorted_subject_areas.sort()
    freq_map = dict(zip(sorted_subject_areas,[0]*len(subject_areas)))
    pair_freq_map = {}
    for _,row in papers_subs.iterrows():
        pri_subs = row['Primary Subject Area'].split(';') if type(row['Primary Subject Area'])!= float else []
        sec_subs = row['Secondary Subject Areas'].split(';') if type(row['Secondary Subject Areas'])!= float else []    
        all_subs = list(set(pri_subs + sec_subs))
        all_subs = list(set([x.strip()[:max_subject_length].strip() for x in all_subs]))
        all_subs.sort()
        for sub in all_subs:
            freq_map[sub] += 1
        for pair in itertools.combinations(all_subs,2):
            if pair not in pair_freq_map:
                pair_freq_map[pair] = 1
            else:
                pair_freq_map[pair] += 1

    return freq_map, pair_freq_map

def get_subject_frequencies_rev(papers,subject_areas,max_subject_length=100,track2=False):
    papers_subs = papers[['E-mail', 'Primary Subject Area', 'Secondary Subject Areas' ]]
    sorted_subject_areas = deepcopy(subject_areas)
    sorted_subject_areas.sort()
    freq_map = dict(zip(sorted_subject_areas,[0]*len(subject_areas)))
    pair_freq_map = {}
    for _,row in papers_subs.iterrows():
        pri_subs = row['Primary Subject Area'].split(';') if type(row['Primary Subject Area'])!= float \
        and (row['Primary Subject Area'].strip())!= '' else []
        sec_subs = row['Secondary Subject Areas'].split(';') if type(row['Secondary Subject Areas'])!= float \
        and (row['Secondary Subject Areas'].strip())!= '' else []    
        all_subs = list(set(pri_subs + sec_subs))
        all_subs = list(set([x.strip()[:max_subject_length].strip() for x in all_subs]))
        l1s = [x for x in all_subs if '->' not in x]
        #if not track2:#sometimes the is only L1 subject area in track1
        all_subs = [x for x in all_subs if '->' in x or ':' in x]
        all_subs.sort()
        for sub in all_subs:
            freq_map[sub] += 1
        for pair in itertools.combinations(all_subs,2):
            if pair not in pair_freq_map:
                pair_freq_map[pair] = 1
            else:
                pair_freq_map[pair] += 1
    return freq_map, pair_freq_map

def get_pmi_matrix(papers, subject_areas, subject_index, frequencies, pair_frequencies, max_subject_length=100):
    pmi_matrix = np.zeros((len(subject_areas),len(subject_areas)))
    p_matrix = np.zeros((len(subject_areas),len(subject_areas)))
    total = len(papers)
    for (s1,s2), freq in pair_frequencies.items():
        s1_i = subject_index[s1]
        s2_i = subject_index[s2]
        p_s1_s2 = freq/total
        p_s1 = frequencies[s1]/total
        p_s2 = frequencies[s2]/total
        pmi_matrix[s1_i,s2_i] = np.log(p_s1_s2/(p_s1 * p_s2))
        pmi_matrix[s2_i,s1_i] = np.log(p_s1_s2/(p_s1 * p_s2)) #the pairs are not permuted in the pair_frequencies
        p_matrix[s1_i,s2_i] = freq/frequencies[s2] if frequencies[s2] >= 5 else 0
        p_matrix[s2_i,s1_i] = freq/frequencies[s1] if frequencies[s1] >= 5 else 0

    max_pmi = np.max(pmi_matrix)
    max_p = np.max(p_matrix)
    pmi_matrix/=(max_pmi*8)
    pmi_matrix[pmi_matrix<0]=0

    return pmi_matrix, p_matrix, (max_pmi, max_p)

def test_pmi_randomly(frequencies, pair_frequencies, subject_index, pmi_matrix, papers):
    total = len(papers)
    item = np.random.choice(len(np.array(list(pair_frequencies.items()))),1)
    item = np.array(list(pair_frequencies.items()))[item][0]
    print("Picked Pair:",item)
    (a,b),c = item
    print("Subjects area pair:",subject_index[a],subject_index[b])
    print("Marginal frequency of the pair:",frequencies[a],frequencies[b])
    print(f"fraction of first:%0.4f"%(frequencies[a]/total))
    print(f"fraction of second:%0.4f"%(frequencies[b]/total))
    print(f"fraction of pair:%0.4f"%(c/total))
    print(f"PMI:%0.4f"%(c/total/(frequencies[a]/total*frequencies[b]/total)))
    print(f"Matrix Value:%0.4f"%pmi_matrix[subject_index[a]][subject_index[b]])


def check_author_emails_in_papers(papers):
    paper_authors = papers[['Paper ID','Author Emails']]
    flagged = []
    for i, row in paper_authors.iterrows():
        if type(row['Author Emails'])==float:
            continue
        author_emails = row['Author Emails'].split(";")
        n_emails = len([m.start() for m in re.finditer('@',row['Author Emails'])])
        if len(author_emails)!= n_emails:
            flagged.append(i)
    if len(flagged)>0:
        raise ValueError("emails are not ; separated. Check the file!")

    _check_author_emails_in_paper_2(papers)

def _check_author_emails_in_paper_2(papers):
    paper_authors = papers[['Paper ID','Primary Contact Author Email','Author Emails']]
    flagged = []
    for index,row in paper_authors.iterrows():
        if type(row['Author Emails'])==float:
            continue
        emails = row['Author Emails'].split(";")
        emails = [x.strip() for x in emails]
        primary = row['Primary Contact Author Email'].strip()
        for mail in emails:
            if not mail[-1].isalpha() and primary not in mail:
                flagged.append(index)
                break
        if len(flagged)>0:
            raise ValueError("Paper Author Emails have special unknown characters")

def get_author_paper_subjects(papers,max_subject_length=100):
    paper_authors = papers[['Primary Contact Author Email','Author Emails','Primary Subject Area', 'Secondary Subject Areas']]
    author_paper_subjects = {}
    for i, row in paper_authors.iterrows():
        primary_email = row['Primary Contact Author Email'].strip()
        secondary_emails = row['Author Emails'].split(";") if type(row['Author Emails'])!=float else []
        secondary_emails = [x.strip() for x in secondary_emails if primary_email not in x] #primary email is starred in secondary emails sometimes
        author_emails = list(set([primary_email] + secondary_emails))

        priamry_subjects = row['Primary Subject Area'].strip() if type(row['Primary Subject Area'])!=float else ''
        secondary_subjects = row['Secondary Subject Areas'].split(";") if type(row['Secondary Subject Areas'])!=float else []
        subjects = list(set([priamry_subjects] + secondary_subjects))
        subjects = [x.strip()[:max_subject_length].strip() for x in subjects]

        for email in author_emails:
            if email in author_paper_subjects:
                author_paper_subjects[email]= list(set(author_paper_subjects[email]+subjects))
            else:
                author_paper_subjects[email] = subjects
    return author_paper_subjects

def check_reviewers_data(reviewers):
    reviewers_subs = reviewers[['E-mail', 'Secondary Subject Areas']]
    flagged = []
    for i, row in reviewers_subs.iterrows():
        if type(row['Secondary Subject Areas'])==float or row['Secondary Subject Areas'] == '':
            continue
        author_emails = row['Secondary Subject Areas'].split(";")
        n_emails = len([m.start() for m in re.finditer('->',row['Secondary Subject Areas'])])
        if len(author_emails)!= n_emails:
            print(len(author_emails), n_emails)
            flagged.append(i)
    if len(flagged)>0:
        print("check these:")
        print(flagged)
        #raise ValueError("emails are not ; separated. Check the file!")

def REVIEWER_VECTOR_get_other_foundation_weights(primary,secondary, l1_other_foundation, subject_index):
    all_sa = primary+secondary
    l1 = [x.split("->")[0].strip() for x in all_sa]
    l1 = [x for x in l1 if x!='' and '(APP)' not in x and '!Focus Area' not in x]
    freqs = {i:l1.count(i) for i in set(l1)}
    weights = {subject_index[l1_other_foundation[k]]:v/10 if v<5 else 0.5 for k,v in freqs.items()}
    return weights

def get_l1_to_subject_index(subject_index):
    subject_areas = list(subject_index.keys())
    l1 = list(set([x.split("->")[0].strip() for x in subject_areas]))
    l1_to_l2_map = {}
    for sub in subject_areas:
        l1 = sub.split("->")[0].strip()
        if l1 not in l1_to_l2_map:
            l1_to_l2_map[l1]=[]
        l1_to_l2_map[l1].append(sub)
    l1_to_l2_index_map = {}
    for k,v in l1_to_l2_map.items():
        indices = [subject_index[x] for x in v]
        l1_to_l2_index_map[k]=indices
    return l1_to_l2_index_map

def get_l1_to_all_subject_index(subject_index):
    subject_areas = list(subject_index.keys())
    l1 = list(set([x.split("->")[0].strip().split(":")[0].strip() for x in subject_areas]))
    l1_to_l2_map = {}
    for sub in subject_areas:
        l1 = sub.split("->")[0].strip().split(":")[0].strip()
        if l1 not in l1_to_l2_map:
            l1_to_l2_map[l1]=[]
        l1_to_l2_map[l1].append(sub)
    l1_to_l2_index_map = {}
    for k,v in l1_to_l2_map.items():
        indices = [subject_index[x] for x in v]
        l1_to_l2_index_map[k]=indices
    return l1_to_l2_index_map

def get_inter_sa_given_sa(secondary_l1,dummy,l1_to_l2_index_map,wt):
    all_inter_sa = [inter_l1_mapping[x] for x in secondary_l1 if x in inter_l1_mapping]
    all_inter_sa = [[(sa,w) for sa in sas] for sas,w in all_inter_sa]
    all_inter_sa = list(itertools.chain.from_iterable(all_inter_sa))
    
    inter_l1_vec = deepcopy(dummy)
    for sa,w in all_inter_sa:
        inter_l1_vec[l1_to_l2_index_map[sa]]=np.maximum(inter_l1_vec[l1_to_l2_index_map[sa]],w*wt)
    return inter_l1_vec

def get_reviewer_vectors(reviewers, subject_index, author_subject_papers, pmi_matrix, max_subject_length=100, track2=False):
    #check_reviewers_data(reviewers)
    reviewers_vec = reviewers[['E-mail', 'Primary Subject Area', 'Secondary Subject Areas']]
    dim = len(subject_index)
    dummy = np.zeros(dim)
    reviwer_u1 = {} #primary
    reviwer_u2 = {} #secondary
    reviwer_u3 = {} #paper authored
    reviewer_r = {} #max of all these
    primary_sub_weight = 1.
    secondary_sub_weight = 0.8
    paper_sub_weight = 0.2
    reviewer_primary_weight = 0.4
    reviewer_secondary_weight = 0.9 * reviewer_primary_weight

    primary_pmi_weight = 0.5
    secondary_pmi_weight = 0.4
    paper_pmi_weight = 0.2

    other_foundation = [x for x in list(subject_index.keys()) if 'Other Foundations' in x]
    l1_other_foundation = {x.split("->")[0].strip():x for x in other_foundation}
    l1_to_l2_index_map = get_l1_to_subject_index(subject_index)

    max_subject_length = 100
    for _,row in reviewers_vec.iterrows():
        #get primary, secondary and paper subject areas
        email = row['E-mail'].strip()
        primary = row['Primary Subject Area'].split(';') if type(row['Primary Subject Area'])!= float else []
        secondary = row['Secondary Subject Areas'].split(';') if type(row['Secondary Subject Areas'])!= float else []
        primary_idx = [subject_index[x.strip()[:max_subject_length].strip()] for x in primary if x.strip()[:max_subject_length].strip() in subject_index]
        secondary_idx = [subject_index[x.strip()[:max_subject_length].strip()] for x in secondary if x.strip()[:max_subject_length].strip() in subject_index]
        if email in author_subject_papers:
            paper_idx = [subject_index[x.strip()[:max_subject_length].strip()] for x in author_subject_papers[email] if x.strip()[:max_subject_length].strip() in subject_index]
        else:
            paper_idx = []

        if not track2:
            primary_l1 = row['Primary Subject Area'].split("->")[0].strip() if type(row['Primary Subject Area'])!= float else ""
            secondary_l1 = list(set([x.split("->")[0].strip() for x in secondary]))
        else:
            primary_l1=''
            secondary_l1 = []
        #construct vector for reviewer
        primary_vec = deepcopy(dummy)
        secondary_vec = deepcopy(dummy)
        papers_vec = deepcopy(dummy)
        primary_vec[primary_idx] = primary_sub_weight
        secondary_vec[secondary_idx] = secondary_sub_weight
        papers_vec[paper_idx] = paper_sub_weight
        reviwer_u1[email] = primary_vec
        reviwer_u2[email] = secondary_vec
        reviwer_u3[email] = papers_vec

        #other foundation
        o_f_vec = deepcopy(dummy)
        if not track2:
            o_f_weights = REVIEWER_VECTOR_get_other_foundation_weights(primary,secondary,l1_other_foundation,subject_index)
            o_f_vec[list(o_f_weights.keys())]=list(o_f_weights.values())

        #primary l1 vector
        primary_l1_indices = l1_to_l2_index_map[primary_l1] if primary_l1 != "" else []
        primary_l1_vec = deepcopy(dummy)
        primary_l1_vec[primary_l1_indices] = reviewer_primary_weight

        #secondary l1 vector
        secondary_l1_indices = [l1_to_l2_index_map[x] for x in secondary_l1 if x in l1_to_l2_index_map]
        secondary_l1_indices = list(itertools.chain.from_iterable(secondary_l1_indices))
        secondary_l1_vec = deepcopy(dummy)
        secondary_l1_vec[secondary_l1_indices] = reviewer_secondary_weight

        #inter l1 vector
        sec_inter_sa = get_inter_sa_given_sa(secondary_l1,dummy,l1_to_l2_index_map,reviewer_secondary_weight)
        pri_inter_sa = get_inter_sa_given_sa([primary_l1],dummy,l1_to_l2_index_map,reviewer_primary_weight)

        vec = np.maximum.reduce([primary_vec, secondary_vec, papers_vec, o_f_vec, primary_l1_vec, secondary_l1_vec, sec_inter_sa, pri_inter_sa])

        #fetch PMI vectors of primary and secondary and combine all
        secondary_pmi = np.max(pmi_matrix[:,secondary_idx]*np.repeat(vec[ np.newaxis, secondary_idx], pmi_matrix.shape[0], axis=0),axis=1) if len(secondary_idx)>0 else np.zeros(len(subject_index))
        primary_pmi = np.max(pmi_matrix[:,primary_idx]*np.repeat(vec[ np.newaxis, primary_idx], pmi_matrix.shape[0], axis=0),axis=1) if len(primary_idx)>0 else np.zeros(len(subject_index))

        paper_pmi = np.zeros(len(subject_index))
        #paper_pmi = np.max(pmi_matrix[paper_idx,:],axis=1) if len(paper_idx)>0 else np.zeros(len(subject_index))
        vec = np.maximum.reduce([vec, secondary_pmi, primary_pmi, paper_pmi_weight * paper_pmi])
        reviewer_r[email]=vec

    return reviewer_r

def get_papers_as_reviewer_vectors(reviewers, subject_index, author_subject_papers, pmi_matrix, max_subject_length=100, track2=False):
    #check_reviewers_data(reviewers)
    #reviewers['E-mail']=reviewers.apply(lambda x:str(x['Paper ID']),axis=1)
    reviewers_vec = reviewers[['E-mail', 'Primary Subject Area', 'Secondary Subject Areas']]
    dim = len(subject_index)
    dummy = np.zeros(dim)
    reviwer_u1 = {} #primary
    reviwer_u2 = {} #secondary
    reviwer_u3 = {} #paper authored
    reviewer_r = {} #max of all these
    primary_sub_weight = 1.
    secondary_sub_weight = 0.8
    paper_sub_weight = 0.2
    reviewer_primary_weight = 0.4
    reviewer_secondary_weight = 0.9 * reviewer_primary_weight

    primary_pmi_weight = 0.5
    secondary_pmi_weight = 0.4
    paper_pmi_weight = 0.2

    other_foundation = [x for x in list(subject_index.keys()) if 'Other Foundations' in x]
    l1_other_foundation = {x.split("->")[0].strip():x for x in other_foundation}
    l1_to_l2_index_map = get_l1_to_all_subject_index(subject_index)

    max_subject_length = 100
    for _,row in reviewers_vec.iterrows():
        #get primary, secondary and paper subject areas
        email = row['E-mail'].strip()
        primary = row['Primary Subject Area'].split(';') if type(row['Primary Subject Area'])!= float else []
        secondary = row['Secondary Subject Areas'].split(';') if type(row['Secondary Subject Areas'])!= float else []
        primary_idx = [subject_index[x.strip()[:max_subject_length].strip()] for x in primary if x.strip()[:max_subject_length].strip() in subject_index]
        secondary_idx = [subject_index[x.strip()[:max_subject_length].strip()] for x in secondary if x.strip()[:max_subject_length].strip() in subject_index]
        if email in author_subject_papers:
            paper_idx = [subject_index[x.strip()[:max_subject_length].strip()] for x in author_subject_papers[email] if x.strip()[:max_subject_length].strip() in subject_index]
        else:
            paper_idx = []

    
        primary_l1 = row['Primary Subject Area'].split("->")[0].strip().split(":")[0].strip() if type(row['Primary Subject Area'])!= float else ""
        secondary_l1 = list(set([x.split("->")[0].strip().split(":")[0].strip() for x in secondary]))
        #construct vector for reviewer
        primary_vec = deepcopy(dummy)
        secondary_vec = deepcopy(dummy)
        papers_vec = deepcopy(dummy)
        primary_vec[primary_idx] = primary_sub_weight
        secondary_vec[secondary_idx] = secondary_sub_weight
        papers_vec[paper_idx] = paper_sub_weight
        reviwer_u1[email] = primary_vec
        reviwer_u2[email] = secondary_vec
        reviwer_u3[email] = papers_vec

        #other foundation
        o_f_vec = deepcopy(dummy)
        #if primary_l1!='AISI':
        #    o_f_weights = REVIEWER_VECTOR_get_other_foundation_weights(primary,secondary,l1_other_foundation,subject_index)
        #    o_f_vec[list(o_f_weights.keys())]=list(o_f_weights.values())

        #primary l1 vector
        primary_l1_indices = l1_to_l2_index_map[primary_l1] if primary_l1 != "" else []
        primary_l1_vec = deepcopy(dummy)
        primary_l1_vec[primary_l1_indices] = reviewer_primary_weight

        #secondary l1 vector
        secondary_l1_indices = [l1_to_l2_index_map[x] for x in secondary_l1 if x in l1_to_l2_index_map]
        secondary_l1_indices = list(itertools.chain.from_iterable(secondary_l1_indices))
        secondary_l1_vec = deepcopy(dummy)
        secondary_l1_vec[secondary_l1_indices] = reviewer_secondary_weight

        #inter l1 vector
        sec_inter_sa = get_inter_sa_given_sa(secondary_l1,dummy,l1_to_l2_index_map,reviewer_secondary_weight)
        pri_inter_sa = get_inter_sa_given_sa([primary_l1],dummy,l1_to_l2_index_map,reviewer_primary_weight)

        vec = np.maximum.reduce([primary_vec, secondary_vec, papers_vec, o_f_vec, primary_l1_vec, secondary_l1_vec, sec_inter_sa, pri_inter_sa])

        #fetch PMI vectors of primary and secondary and combine all
        secondary_pmi = np.max(pmi_matrix[:,secondary_idx]*np.repeat(vec[ np.newaxis, secondary_idx], pmi_matrix.shape[0], axis=0),axis=1) if len(secondary_idx)>0 else np.zeros(len(subject_index))
        primary_pmi = np.max(pmi_matrix[:,primary_idx]*np.repeat(vec[ np.newaxis, primary_idx], pmi_matrix.shape[0], axis=0),axis=1) if len(primary_idx)>0 else np.zeros(len(subject_index))

        paper_pmi = np.zeros(len(subject_index))
        #paper_pmi = np.max(pmi_matrix[paper_idx,:],axis=1) if len(paper_idx)>0 else np.zeros(len(subject_index))
        vec = np.maximum.reduce([vec, secondary_pmi, primary_pmi, paper_pmi_weight * paper_pmi])
        reviewer_r[email]=vec

    return reviewer_r

def convert_to_matrices(reviewer_r, paper_v1, paper_v2):
    #make reviewer id map
    reviewers_ids = list(reviewer_r.keys())
    reviewers_ids.sort()
    reviewers_index = dict(zip(reviewers_ids,range(len(reviewers_ids))))

    #convert reviewer_r to matrix
    reviewer_matrix = [[]]*len(reviewers_ids)
    for rev, vec in reviewer_r.items():
        reviewer_matrix[reviewers_index[rev]]=vec
    reviewer_matrix = np.array(reviewer_matrix)
    print("reviewer matrix shape:", reviewer_matrix.shape)

    #convert paper_v1, paper_v2 to matrix
    paper_ids = list(paper_v1.keys())
    paper_ids.sort()
    paper_index = dict(zip(paper_ids,range(len(paper_ids))))

    v1_matrix = [[]]*len(paper_ids)
    for rev, vec in paper_v1.items():
        v1_matrix[paper_index[rev]]=vec
    v1_matrix = np.array(v1_matrix)
    print("v1_matrix shape:", v1_matrix.shape)
    v2_matrix = [[]]*len(paper_ids)
    for rev, vec in paper_v2.items():
        v2_matrix[paper_index[rev]]=vec
    v2_matrix = np.array(v2_matrix)
    print("v2_matrix shape:", v2_matrix.shape)

    return reviewer_matrix, v1_matrix, v2_matrix, (reviewers_index, paper_index)

def convert_to_matrices_paper_as_reviewers(reviewer_r, paper_v1, paper_v2):
    reviewer_r = {int(k):v for k,v in reviewer_r.items()}
    #make reviewer id map
    reviewers_ids = list(reviewer_r.keys())
    reviewers_ids.sort()
    reviewers_index = dict(zip(reviewers_ids,range(len(reviewers_ids))))

    #convert reviewer_r to matrix
    reviewer_matrix = [[]]*len(reviewers_ids)
    for rev, vec in reviewer_r.items():
        reviewer_matrix[reviewers_index[rev]]=vec
    reviewer_matrix = np.array(reviewer_matrix)
    print("reviewer matrix shape:", reviewer_matrix.shape)

    #convert paper_v1, paper_v2 to matrix
    paper_ids = list(paper_v1.keys())
    paper_ids.sort()
    paper_index = dict(zip(paper_ids,range(len(paper_ids))))

    v1_matrix = [[]]*len(paper_ids)
    for rev, vec in paper_v1.items():
        v1_matrix[paper_index[rev]]=vec
    v1_matrix = np.array(v1_matrix)
    print("v1_matrix shape:", v1_matrix.shape)
    v2_matrix = [[]]*len(paper_ids)
    for rev, vec in paper_v2.items():
        v2_matrix[paper_index[rev]]=vec
    v2_matrix = np.array(v2_matrix)
    print("v2_matrix shape:", v2_matrix.shape)

    return reviewer_matrix, v1_matrix, v2_matrix, (reviewers_index, paper_index)


def sorted_match(v2_matrix, reviewer_matrix):
    # v2_matrix = (#papers,#keywords)
    # reviewer_matrix = (#keywords,#reviewers)
    geometric_weights = np.ones(reviewer_matrix.shape[0])
    geometric_weights[1:]*=0.5
    geometric_weights = np.cumprod(geometric_weights)
    norm_geometric_weights = np.linalg.norm(geometric_weights,ord=2,axis=0)
    norm_matrix = np.zeros((v2_matrix.shape[0],reviewer_matrix.shape[1]))
    match_score_matrix = np.zeros((v2_matrix.shape[0],reviewer_matrix.shape[1]))
    for paper in range(v2_matrix.shape[0]):
        for reviewer in range(reviewer_matrix.shape[1]):
            m = v2_matrix[paper,:]*reviewer_matrix[:,reviewer]
            m = m[m>0]
            m.sort()
            m=m[::-1]
            #norm_m = np.linalg.norm(m,ord=2,axis=0)
            match_score_matrix[paper,reviewer] = np.dot(m,geometric_weights[:len(m)])
            norm_matrix[paper,reviewer]=np.sum(geometric_weights[:len(m)])*0.5
    return match_score_matrix, norm_matrix

def get_scores(reviewer_matrix, v1_matrix, v2_matrix):
    norm_r = np.linalg.norm(reviewer_matrix,ord=2,axis=1)
    norm_v2 = np.linalg.norm(v2_matrix,ord=2,axis=1)

    reviewer_matrix = np.transpose(reviewer_matrix)
    match_score_matrix, norm_matrix = sorted_match(v2_matrix, reviewer_matrix)

    v1_sum = v1_matrix.sum(axis=1)
    norm_matrix+=v1_sum[:,None]

    score = ((np.dot(v1_matrix, reviewer_matrix) + match_score_matrix))/norm_matrix
    print("Final Scores shape:", score.shape)
    return score

def get_sorted_match_matrix(reviewer_matrix, v2_matrix):
    reviewer_matrix = np.transpose(reviewer_matrix)
    match_score_matrix, norm_matrix = sorted_match(v2_matrix, reviewer_matrix)
    return match_score_matrix

def write_scores(scores, overlap_matrix, reviewers_index, paper_index, score_file,mode="w"):
    inv_reviewers_index = {v: k for k, v in reviewers_index.items()}
    inv_paper_index = {v: k for k, v in paper_index.items()}
    with open(score_file,mode) as f:
        csvwriter = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE) 
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]): 
                csvwriter.writerow([str(inv_paper_index[i]),inv_reviewers_index[j],str(scores[i][j]),str(overlap_matrix[i,j])])

def write_paper_scores_test(scores, overlap_matrix, reviewers_index, paper_index, score_file,mode="w"):
    inv_reviewers_index = {v: k for k, v in reviewers_index.items()}
    inv_paper_index = {v: k for k, v in paper_index.items()}
    with open(score_file,mode) as f:
        csvwriter = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE) 
        i=10
        for j in range(scores.shape[1]): 
            csvwriter.writerow([str(inv_paper_index[i]),inv_reviewers_index[j],str(scores[i][j]),str(scores[j][i]),str((scores[i][j]+scores[j][i])/2)])

def write_papers_scores(scores, overlap_matrix, reviewers_index, paper_index, score_file, selected_papers,mode="w"):
    inv_reviewers_index = {v: k for k, v in reviewers_index.items()}
    inv_paper_index = {v: k for k, v in paper_index.items()}
    with open(score_file,mode) as f:
        csvwriter = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE) 
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                if inv_paper_index[i] in selected_papers and int(inv_reviewers_index[j]) in selected_papers:
                    csvwriter.writerow([str(inv_paper_index[i]),inv_reviewers_index[j],str(scores[i][j]),str(overlap_matrix[i,j])])


def get_l1_sa(row):
    if type(row['Secondary Subject Areas'])==float or row['Secondary Subject Areas']=='':
        r_sec = []
    else:
        r_sec = [x.strip() for x in row['Secondary Subject Areas'].split(";")]
    if type(row['Primary Subject Area'])==float or row['Primary Subject Area']=='':
        r_pri = ""
    else:
        r_pri = row['Primary Subject Area']
    r_all = [r_pri]+r_sec    
    r_l1 = list(set([x.split("->")[0].strip() for x in r_all]))
    return r_l1

def rule_2_reviewers(row):
    r_l1 = row['l1']
    rule2_set = list(set([rule2[x] for x in r_l1 if x in rule2]))
    return rule2_set

def enrich_with_l1_and_rule2(data,id_col,track2=False):
    if track2:
        length = len(data[id_col])
        l1_dict = dict(zip(data[id_col],[[]]*length))
        rule2_dict = dict(zip(data[id_col],[[]]*length))
        return l1_dict, rule2_dict

    data['l1'] = data.apply(get_l1_sa,axis=1)
    data['rule2_set'] = data.apply(rule_2_reviewers,axis=1)
    l1_dict = dict(zip(data[id_col],data['l1']))
    rule2_dict = dict(zip(data[id_col],data['rule2_set']))
    return l1_dict, rule2_dict

def make_rule2_matrix(paper_index,reviewers_index,paper_rule2_dict,rev_rule2_dict):
    rule2_matrix = np.zeros((len(paper_index),len(reviewers_index)))
    for paper, paper_r2_set in paper_rule2_dict.items():
        for reviewer, reviewer_r2_set in rev_rule2_dict.items():
            if paper in paper_index and reviewer in reviewers_index:
                p_idx = paper_index[paper]
                rev_idx = reviewers_index[reviewer] 
                set_diff = list(set(paper_r2_set)-set(reviewer_r2_set))
                if len(set_diff)>0:
                    rule2_matrix[p_idx,rev_idx]=1
    return rule2_matrix

def get_overlap_matrix(paper_index,reviewers_index,paper_l1_dict,rev_l1_dict):
    overlap_matrix = np.zeros((len(paper_index),len(reviewers_index)))
    for paper, paper_l1_set in paper_l1_dict.items():
        for reviewer, reviewer_l1_set in rev_l1_dict.items():
            if paper in paper_index and reviewer in reviewers_index:
                p_idx = paper_index[paper]
                rev_idx = reviewers_index[reviewer] 
                set_intersect = [x for x in paper_l1_set if x in reviewer_l1_set]
                if len(set_intersect)>0:
                    overlap_matrix[p_idx,rev_idx]=1
    return overlap_matrix

def loadReviewerFiles(path,ac=False):
    email_col = 'Senior Meta-Reviewer Email' if ac else 'E-mail'
    with open(path,"r",encoding='utf-8') as f:
        reviewers = f.read().splitlines()
        reviewers=[x.split("\t") for x in reviewers]
        columns=reviewers[0]
        reviewers = pd.DataFrame(reviewers[1:],columns=columns)
        reviewers[email_col] = reviewers.apply(lambda row:row[email_col].lower(),axis=1)
    return reviewers

def get_AC_data(path):
    AC_data = loadReviewerFiles( path,True)
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
