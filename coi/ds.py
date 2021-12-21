import yaml
import utils
from utils import *
from CONSTANTS import *

import logging
logger = logging.getLogger(__file__)

#fhandle = logging.FileHandler('notebook_coi.log')
#logger.addHandler(fhandle)
#logger.addHandler(console_handler)


class User:
    def __init__(self,row=None,author_info_map=None, name_to_ids = None,source=None,config=None):
        self.temp_cache = {}
        self.config = config
        self.all_primaries = set()
        self.paper_ids = set()
        self.coauthors = set()
        self.dblp_ids = None
        self.coauthor_emails = set()
        self.explicit_conflicts = []
        self.explicit_conflict_emails = set()  
        self.explicit_conflicts_not_in_cmt = []
        self.explicit_conflict_emails_not_in_cmt = set()  
        self.source = source
        self.is_popular = False 
        self.original_conflict_domains = set()
        self.domains_on_cmt = set()
        self.conflict_domains = set()

        if source == 'user_list':
            self.user_type = 'cmt'
            self.email_id = row.email.strip().lower()
            assert self.email_id != ''
            self.all_primaries.add(self.email_id)
            self.fname = row.fname.strip().encode(ENCODING)
            self.lname = row.lname.strip().encode(ENCODING)
            self.alias = (self.fname.strip()+ ' '.encode(ENCODING) + self.lname.strip())
            self.pub_emails = set([self.email_id])
            self.original_conflict_domains = set([x.strip().lower() for x in row.conflict_domains.replace(',',';').split(';')]) \
                                if not pd.isna(row.conflict_domains) else set()
            self.domains_on_cmt = set([x.strip().lower() for x in row.conflict_domains.replace(',',';').split(';')]) \
                                if not pd.isna(row.conflict_domains) else set()

            self.conflict_domains.update(self.original_conflict_domains) 
            self.add_domain_from_emails()
            self.conflict_domains = self.conflict_domains.difference(IGNORE_DOMAINS)
            if '' in self.pub_emails:
                self.pub_emails.remove('') 
            


        elif source == 'dblp':
            assert isinstance(row, str) 
            #we are given only dblp id . Initialize using this only. 
            self.dblp_ids = [row]
            self.user_type = 'dblp'
            self.pub_emails = set()
            self.alias = ''
            if author_info_map is not None:
                self.alias = author_info_map[row].get('aliases',[None])[0]
        elif source == 'user_info':
            self.initialize(row, author_info_map, name_to_ids)
            self.user_type = 'cmt'
        else:
            print(row)
            logger.error("source info missing while creating a user")
            raise 'source is mandatory'
        #
        self.reset_domains()
    #
    def sort_paper_list(self,pid2paper):
        self.paper_list = [pid2paper[x] for x in self.paper_ids]
        self.paper_list.sort(key= lambda x: x.year)


    def populate_dblp_keys(self, dblp_str, author_info_map, name_to_ids):
        def find_and_filter(dblp_str,author_info_map):
            if pd.isna(dblp_str):
                return  None
            #
            keys =  [extract_dblp_key_simple(x.strip()) for x in dblp_str.split()]
            filtered = [x for x in keys if (x is not None and x in author_info_map)]
            if len(filtered) == 0:
                return None
            else: 
                return filtered
        self.dblp_ids = find_and_filter(dblp_str, author_info_map)
        if (not pd.isna(dblp_str)) and dblp_str.strip() != '' and self.dblp_ids is None:
            logger.warning('{} Invalid dblp id: {}'.format(self.email_id,dblp_str))
        #if self.dblp_ids is None and self.alias in name_to_ids:
        #    self.dblp_ids = name_to_ids[self.alias]


    def initialize(self,row, author_info_map, name_to_ids): 
        self.email_id = row.email.strip().lower()
        assert self.email_id != ''
        self.all_primaries.add(self.email_id)
        self.primary_area = row.primary_area.split('->')[0].strip() if not pd.isna(row.primary_area) else ''
        self.is_popular = self.primary_area in POPULAR_PRIMARY_AREAS 
        self.fname = row.fname.strip().encode(ENCODING)
        self.lname = row.lname.strip().encode(ENCODING)
        self.alias = (self.fname.strip()+ ' '.encode(ENCODING) + self.lname.strip())
        self.populate_dblp_keys(row.dblp_id, author_info_map, name_to_ids)
        #
        self.pub_emails = set([x.strip().lower() for x in row.pub_emails.replace(',',';').split(';')]) \
                                if not pd.isna(row.pub_emails) else set()
        if self.email_id in self.config.IGNORE_PUB_MAILS_FOR_USERS:
            self.pub_emails = set(self.config.IGNORE_PUB_MAILS_FOR_USERS[self.email_id])

        self.pub_emails = set([x for x in self.pub_emails if is_email_valid(x)])

        self.original_conflict_domains = set([x.strip().lower() for x in row.conflict_domains.replace(',',';').split(';')]) \
                                if not pd.isna(row.conflict_domains) else set()
        
        self.domains_on_cmt = set([x.strip().lower() for x in row.conflict_domains.replace(',',';').split(';')]) \
                                if not pd.isna(row.conflict_domains) else set()
        
        self.conflict_domains.update(self.original_conflict_domains)
        self.explicit_conflicts = yaml.load(row.explicit_conflicts.replace('\\/','')) if not pd.isna(row.explicit_conflicts) else []
        self.explicit_conflicts_not_in_cmt = yaml.load(row.explicit_conflicts_not_in_cmt.replace('\\/','')) if not pd.isna(row.explicit_conflicts_not_in_cmt) else []
        self.explicit_conflicts += self.explicit_conflicts_not_in_cmt
        self.pub_emails.add(self.email_id)
        self.populate_explicit_conflict_emails()
        self.add_domain_from_emails()
        self.conflict_domains = self.conflict_domains.difference(IGNORE_DOMAINS)
        if '' in self.pub_emails:
            self.pub_emails.remove('') 
    
    def populate_explicit_conflict_emails(self):
        self.explicit_conflict_emails = set([x.get('email','-1').lower() for x in self.explicit_conflicts])
        if '-1' in self.explicit_conflict_emails:
            self.explicit_conflict_emails.remove('-1')
               
        self.explicit_conflict_emails_not_in_cmt = set([x.get('email','-1').lower() for x in self.explicit_conflicts_not_in_cmt])
        if '-1' in self.explicit_conflict_emails_not_in_cmt:
            self.explicit_conflict_emails_not_in_cmt.remove('-1')


    def remap_domains(self,old2new):
        self.original_conflict_domains = set([old2new[x] for x in self.original_conflict_domains])
        self.conflict_domains = set([old2new[x] for x in self.conflict_domains])

    def reset_domains(self):
        #self.conflict_domains.clear()
        self.original_conflict_domains = set([x[1:] if x.startswith('.') else x for x in self.original_conflict_domains])
        self.original_conflict_domains = self.original_conflict_domains.difference(IGNORE_DOMAINS)
        self.original_conflict_domains = set([normalize_domain(x) for x in self.original_conflict_domains])
        self.conflict_domains = set()
        self.conflict_domains.update(self.original_conflict_domains)
        self.add_domain_from_emails()
        self.conflict_domains = set([normalize_domain(x) for x in self.conflict_domains]) 
        self.conflict_domains = self.conflict_domains.difference(IGNORE_DOMAINS)

    def add_domain_from_emails(self):
        for x in self.all_primaries:
            self.conflict_domains.add(x.split('@')[-1].strip().lower())
        #
        is_primary_generic = all([x.split('@')[-1].strip().lower() in IGNORE_DOMAINS for x in self.all_primaries if x != ''])

        if self.config.SHOULD_PARSE_DOMAINS_FROM_PUB_EMAILS or is_primary_generic:
            for x in self.pub_emails:
                self.conflict_domains.add(x.split('@')[-1].strip().lower())
    
    def sanity_check_before_merging(self,other):
        rv = True
        if self.dblp_ids is not None and other.dblp_ids is not None:
            if len(set(self.dblp_ids).intersection(set(other.dblp_ids))) == 0:
                logger.warning("DBLP IDS don't match for the accounts to be merged:#1: {}, #2: {}".format(self.print_details(), other.print_details()))
                rv = False
        return rv
    
    def print_details(self):
        print_str = 'Alias: {}, Email: {}, Pub emails: {}, Primary mails: {}'.format(self.alias,self.email_id, self.pub_emails,self.all_primaries)
        if self.dblp_ids is not None:
            print_str = '{}, Dblp ids: {}'.format(print_str,';'.join(self.dblp_ids))
        return print_str


    def merge_details_from(self,other):
        rv = self.sanity_check_before_merging(other)
        # 
        if self.dblp_ids is None:
            self.dblp_ids = other.dblp_ids
        self.pub_emails.update(other.pub_emails)
        self.original_conflict_domains.update(other.original_conflict_domains)
        #self.conflict_domains.update(other.conflict_domains)

        self.domains_on_cmt.update(other.domains_on_cmt) 
        self.explicit_conflict_emails.update(other.explicit_conflict_emails)
        self.explicit_conflict_emails_not_in_cmt.update(other.explicit_conflict_emails_not_in_cmt)
        self.explicit_conflicts += other.explicit_conflicts
        self.explicit_conflicts_not_in_cmt += other.explicit_conflicts_not_in_cmt 
        self.is_popular = self.is_popular or other.is_popular
        self.all_primaries.update(other.all_primaries)
        self.reset_domains() 
        return rv  


    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.email_id == other.email_id
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash(self.email_id)
   
    def is_senior(self,other):
        #Pdb().set_trace()
        if len(self.paper_list) - len(other.paper_list) >= SENIORITY_PAPER_DIF:
            other_first_paper_year = other.paper_list[0].year if len(other.paper_list) > 0 else CURRENT_YEAR
            return (other_first_paper_year - self.paper_list[0].year >= SENIORITY_YEAR_DIF)
        return False



   #'explicit_conflicts': 'Write-in Conflicts'

        
class Paper:
    def __init__(self,article,pid,name_to_ids,source='dblp'):
        self.source = source #dblp or submissions?
        self.id = pid
        self.authors = set()
        self.emails = set() # for papers that are aaai21 submissions
        self.num_authors = 0
        self.num_dblp_ids = 0
        self.num_emails_not_in_users = 0 
        
        if source == 'dblp':
            self.year = int(article.get('year',-1))
            self.parse_authors_dblp(article, name_to_ids)
        else:
            self.year = int(CURRENT_YEAR +1)
            self.parse_authors_submission(article, name_to_ids)
        #
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.id == other.id
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash(self.id)
    
    def parse_authors_submission(self,row, mail2user):
        #Pdb().set_trace()
        emails = set([x.strip().lower() for x in row['Author Emails'].replace('*','').replace(',',';').split(';')]) if (not pd.isna(row['Author Emails'])) else set()        
        if not pd.isna(row['Primary Contact Author Email']):
            emails.add(row['Primary Contact Author Email'].strip().lower())
        self.authors = set()
        self.num_authors = len(emails)
        self.num_dblp_ids = 0
        self.emails = emails
        self.emails_in_users = set()
        for this_email in emails:
            if this_email in mail2user:
                self.emails_in_users.add(mail2user[this_email].email_id)
                if mail2user[this_email].dblp_ids is not None:
                    for this_dblp_id in mail2user[this_email].dblp_ids:
                        self.authors.add(this_dblp_id)
                        self.num_dblp_ids += 1
                else:
                    logger.warning('Parsing Submission: {}. Its dblp id not found in all users'.format(this_email))
            else:
                logger.warning("Author Email not in users!: {}. Paper ID: {}".format(this_email, self.id))
                self.num_emails_not_in_users += 1
                #self.emails_not_in_users.add(this_email)
       
    def parse_authors_dblp(self,article, name_to_ids):
        self.authors = set()
        self.num_authors = 0
        self.num_dblp_ids = 0
        if type(article['author']) != list:
            article['author'] = [article['author']]
        author_list = article['author']
        self.num_authors = len(author_list)
        for index, author_name in enumerate(author_list):
            if type(author_name) is collections.OrderedDict:
                author_name = author_name["#text"]
            author_name = author_name.encode("utf-8")
            for this_dblp_id in name_to_ids[author_name]: 
                self.authors.add(this_dblp_id)
        #
        self.num_dblp_ids = len(self.authors)
        
    def is_author_in(self,userdict):
        for this_author in self.authors:
            if this_author in userdict:
                return True
        #
        return False
                
       


class Conflicts:
    def __init__(self,pair):
        self.pair = pair
        self.rules = set()


#id_pair is a frozenset

