XML_FILE = 'data/dblp/dblp.xml'
DTD_FILE = 'data/dblp/dblp.dtd'
CURRENT_YEAR = 2020
ENCODING = 'utf-8'
STARTYEAR = 1950
ENDYEAR = 2269

POPULAR_PRIMARY_AREAS = ['Speech & Natural Language Processing (SNLP)', 'Machine Learning (ML)', 'Computer Vision (CV)', 'Data Mining & Knowledge Management (DMKM)','Application Domains (APP)']

IGNORE_DOMAINS = set(['gmail.com','hotmail.com','outlook.com','yahoo.com','foxmail.com','163.com','ieee.org','mail.ru',
                        'live.cn', 'live.com', 'live.fr', 'live.in', 'live.it',''])

IGNORE_DOMAINS.update(set(['ieee','live.fr','mail.ru','live.it','live.cn','foxmail','outlook','hotmail','yahoo','live','live.in','gmail','163']))

REMOVE_WORDS_IN_DOMAINS = ['.edu','.org','www.','webmail.','.com']

SENIORITY_YEAR_DIF = 5
SENIORITY_PAPER_DIF = 10

RULE_0 = '0-self'
RULE_1_2 = '1.2-Coauth'
RULE_1_3 = '1.3-Coauth'
RULE_1_4 = '1.4-Sup-Stdt'
RULE_1_5 = '1.5-Sibl'
RULE_1_1_LIMIT = 7
RULE_1_2_LIMIT = 5 
RULE_1_3_LIMIT = 6

RULE_1_4_LIMIT_TOPK = 100 #esentially making it void. Selecting only based on relaxed criteria: if 30% match in the top 10
RULE_1_4_LIMIT_PCT = 0.3
RULE_1_4_LIMIT_RELAX_TOPK = 10 

RULE_2_1 = '2.1-Domain'
RULE_2_2 = '2.2-Explicit'
RULE_3_1 = '3.1-Co-A3I21'


EXCEPTION_2_1 = 'E21_domains'
EXCEPTION_2_2 = 'E22_non-author-conflicts'
EXCEPTION_2_3 = 'E23_asymmetric-conflicts'

EXCEPTION_2_1_LIMIT = 8
EXCEPTION_2_2_LIMIT = 15
EXCEPTION_2_3_LIMIT = 10

h2n = {'First Name': 'fname',
 'Last Name': 'lname',
 'Email': 'email',
 'Google Scholar URL': 'gs_id',
 'Semantic Scholar URL': 'ss_id',
 'DBLP URL': 'dblp_id',
 'Publication Emails': 'pub_emails',
 'Conflict Domains': 'conflict_domains',
 'Write-in Conflicts': 'explicit_conflicts_not_in_cmt',
 'Individual Conflicts': 'explicit_conflicts',
 '# First Name': 'fname',
 'Middle Initial (optional)': 'mname',
 'Last Name': 'lname',
 'E-mail': 'email',
 'Domain Conflicts': 'conflict_domains',
 'Primary Subject Area': 'primary_area',
 'Q10 (How many times have you been part of the program committee (as PC/SPC/AC, etc) of AAAI, IJCAI, NeurIPS, ACL, SIGIR, WWW, RSS, NAACL, KDD, IROS, ICRA, ICML, ICCV, EMNLP, EC, CVPR, AAMAS, HCOMP, HRI, ICAPS, ICDM, ICLR, ICWSM, IUI, KR, SAT, WSDM, UAI, AISTATS, COLT, CORL, CP, CPAIOR, ECAI, OR ECML in the past?)': 'q10',
 'Secondary Subject Areas': 'secondary_area'
 } 

case2_subjects = ['Speech & Natural Language Processing (SNLP)', 'Machine Learning (ML)', 'Computer Vision (CV)', 'Data Mining & Knowledge Management (DMKM)']

n2h = {
    'fname': 'First Name',
    'lname': 'Last Name',
    'email': 'Email',
    'gs_id': 'Google Scholar URL',
    'ss_id': 'Semantic Scholar URL',
    'dblp_id': 'DBLP URL',
    'pub_emails':'Publication Emails',
    'conflict_domains': 'Conflict Domains',
    'explicit_conflicts_not_in_cmt': 'Write-in Conflicts',
    'explicit_conflicts': 'Individual Conflicts'
    }

"""
(1) Between every two authors compute DBLP co-authorship based COI:
1.1 Disregard papers with more than 7 authors
1.2 If two people co-authored a paper in last five years COI = 1
1.3 If two people co-authored more than 6 papers together at any time COI = 1
1.4 If Person 1 is senior to Person 2 and they co-authored several early papers of Person 2, COI = 1. (Here the guess is that Person 1 is the advisor)
1.5 We could conceivably also say that Person 2 has COI with all co-authors of Person 1… not sure if it becomes too extreme.

Else COI = 0

(2) Between every two authors COI = 1 if one of the authors expressed COI explicitly or with the email domain
Exceptions:
As an exception we will look at self-reported COIs, and check manually if
A user has an unreasonable number (8) of domains  as COI domains 
A user has an unreasonable number (15) of non co-authors as COIs 
A user has a large number of self-reported COIs (10) where the COIs’ don’t have this user as a COI in their self-reports

(3) Between each pair of co-authors in AAAI2021 submissions, declare COI=1
(4) If COI between two people is 1, then COI between one and papers written by the other is also 1
(5) Between every two authors compute the shortest distance in co-authorship graph. Is it too hard? Or at least check if the distance is 1 or 2 (Co-author distance).  This will help us in diversity constraint at time of reviewer assignment.

"""


