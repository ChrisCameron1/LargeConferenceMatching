import gzip
import collections
import xmltodict
import os
import csv
import copy
import pprint
import pickle
from  collections import defaultdict
from itertools import combinations
import pandas as pd
# for extracting key from dpbl websites
import urllib.request
import re
from lxml import etree as ElementTree
from IPython.core.debugger import Pdb
# Consider pubs in this range only.
import yaml
from CONSTANTS import *

import logging
logger = logging.getLogger(__file__)

def normalize_domain(x):
    y = x
    for repl in REMOVE_WORDS_IN_DOMAINS:
        x = x.replace(repl,'')
    #
    x = x.replace('amazom','amazon')
    
    if 'fudan' in x:
        return 'fudan'
    #
    if 'cuhk.' in x:
        return 'cuhk'
    #
    if 'washington' in x and x != 'medicine.washington':
        return 'washington'
    #
    if 'harvard' in x:
        return 'harvard'

    if 'tsinghua' in x and x != 'sz.tsinghua.cn': 
        return 'tsinghua'

    if 'pku' in x and 'pkusz' not in x:
        return 'pku'

    if x == 'pkusz.cm' or x == 'sz.pku.cn':
        return 'pkusz.cn'

    if 'utexas' in x:
        return 'utexas'

    if 'ox.ac.uk' in x:
        return 'ox.ac.uk'

    if 'upenn' in x:
        return 'upenn'

    if 'imperial' in x:
        return 'imperial'

    if x in set(['andrew.cmu', 'cmu', 'cs.cmu', 'alumni.cmu', 'ece.cmu', '.cs.cmu', 'cs.ccmu', 'west.cmu']):
        return 'cmu'

    if 'stanford' in x:
        return 'stanford'

    if 'ucla' in x:
        return 'ucla'

    if 'columbia' in x:
        return 'columbia'
    #if x != y:
    #    logger.info('Domain Norm: Replacing {} by {}'.format(y,x))
    return x



def is_email_valid(string):
    return ((len(string) > 0) and (string.find('@') != -1) and (not string.startswith('@')))

def extract_dblp_key_simple(dblp_url):
    if pd.isna(dblp_url):
        return None
    
    if dblp_url.endswith('#dblp'):
        dblp_url = dblp_url[:-5]
    if dblp_url.find("?") != -1:
        cut_from = dblp_url.find("?")
        dblp_url = dblp_url[:cut_from]
    if dblp_url.find("&") != -1:
        cut_from = dblp_url.find("&")
        dblp_url = dblp_url[:cut_from]
    if dblp_url.endswith('.html'):
        dblp_url = dblp_url.strip().replace('.html','')
    elif dblp_url.endswith('.htm'):
        dblp_url = dblp_url.strip().replace('.htm','')
        
    start_at = dblp_url.rfind('pid')
    if start_at != -1:
        return dblp_url[start_at + len('pid/'):].strip('/')
    else:
        return None


    #if dblp_url.startswith("https://dblp.org/pid/"):
    #    return dblp_url[21:]
    #else:
    #    return None
    # not a persistent key but the name-based one. Extract persistent one from
    # DBLP
    #resource = urllib.request.urlopen(dblp_url)
    #content =  resource.read().decode(resource.headers.get_content_charset())
    #m = re.search("<!-- head line --.*dblp key:.*homepages/(.*?)</small>", content)
    #return m.group(1)

def verify_dblp_key(key):
    return True
    #return key.split('/')[0].isdigit() 
    


def extract_gs_key(gs_url):
    if gs_url.find('user=') != -1:
        return gs_url[gs_url.find('user=')+5:]
    else:
        gs_url






#Functions

def get_other(pair, one):
    for other in pair:
        if one != other:
            return other

"""
def add_domain_user(domain, user, to_dict):
    if domain not in to_dict:
        to_dict[domain] = set()
    #
    to_dict[domain].add(user.email_id)
"""

def many_early_papers(junior, senior):
    topk_common = 0
    topk_relax_common = 0
    take_till = min(RULE_1_4_LIMIT_RELAX_TOPK, len(junior.paper_list)) - 1
    if take_till < 0:
        #no papers of the junior, vacous
        return False, {'topk_common': 0, 'topk_relax_common': 0, 'topk_relax_total': 0}
    till_year = junior.paper_list[take_till].year
    is_break = False 
    for i,this_paper in enumerate(junior.paper_list):
        if this_paper.year > till_year:
            is_break = True
            break 
    #
    take_till = i-int(is_break)+1
    #
    for i,this_paper in enumerate(junior.paper_list[:take_till]):
        if this_paper.id in senior.paper_ids:
            topk_relax_common += 1
            if topk_common == i:
                topk_common += 1
    #
    if (topk_common >= RULE_1_4_LIMIT_TOPK):
        #Pdb().set_trace()
        return True, {'topk_common': topk_common, 'topk_relax_common': topk_relax_common,'topk_relax_total': take_till}
    
    if (topk_relax_common >= RULE_1_4_LIMIT_PCT*max(take_till,RULE_1_4_LIMIT_RELAX_TOPK)):
        #qPdb().set_trace()
        return True, {'topk_common': topk_common, 'topk_relax_common': topk_relax_common,'topk_relax_total': take_till}
    
    return False, {'topk_common': topk_common, 'topk_relax_common': topk_relax_common,'topk_relax_total': take_till}
    
def populate_paper_coauthor_emails_in_users(pid2paper, mail2user):
    for pid,this_paper in pid2paper.items():
        for author_email in this_paper.emails:
            if author_email in mail2user:
                mail2user[author_email].paper_ids.add(pid)
                mail2user[author_email].coauthor_emails.update(this_paper.emails)
                for coemail in this_paper.emails:
                    if coemail in mail2user:
                        mail2user[author_email].coauthor_emails.update(mail2user[coemail].pub_emails)
                        if mail2user[coemail].dblp_ids is not None:
                            for codbid in mail2user[coemail].dblp_ids:
                                mail2user[author_email].coauthors.add(codbid)

    #
    for this_email,this_user in mail2user.items():
        this_user.coauthor_emails = this_user.coauthor_emails.difference(this_user.pub_emails)    


def populate_paper_coauthor_list_in_users(pid2paper, dbid2user):
    for pid,this_paper in pid2paper.items():
        for author in this_paper.authors:
            if author in dbid2user:
                dbid2user[author].paper_ids.add(pid)
                dbid2user[author].coauthors.update(this_paper.authors)
    #
    
    for pid, this_user in dbid2user.items():
        this_user.sort_paper_list(pid2paper)
        if pid in this_user.coauthors:
            this_user.coauthors.remove(pid)

"""
def add_pair(auth1,auth2,rule_id, pair2conflict,attributes = None):
    pair = frozenset([auth1,auth2])
    if pair not in pair2conflict:
        pair2conflict[pair] = Conflicts(pair)
    #
    pair2conflict[pair].rules.add(rule_id)
    if attributes is not None:
        pair2conflict[pair].__dict__.update(attributes)
            
"""     

class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]
