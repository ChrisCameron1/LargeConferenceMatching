

"""
This module extracts the conflicts between paper and reviewers. 

The following rules are implemented:

(1) Between every two authors compute DBLP co-authorship based COI:
Disregard papers with more than 7 authors
If two people co-authored a paper in last five years COI = 1
If two people co-authored more than 6 papers together at any time COI = 1
If Person 1 is senior to Person 2 and they co-authored several early papers of Person 2, COI = 1. (Here the guess is that Person 1 is the advisor)
We could conceivably also say that Person 2 has COI with all co-authors of Person 1… not sure if it becomes too extreme.
Else COI = 0

(2) Between every two authors COI = 1 if one of the authors expressed COI explicitly or with the email domain
As an exception we will look at self-reported COIs, and check manually if
A user has an unreasonable number (8) of domains  as COI domains 
A user has an unreasonable number (15) of non co-authors as COIs 
A user has a large number of self-reported COIs (10) where the COIs’ don’t have this user as a COI in their self-reports
(3) Between each pair of co-authors in AAAI2021 submissions, declare COI=1
(4) If COI between two people is 1, then COI between one and papers written by the other is also 1

"""




import argparse

import numpy as np
import pandas as pd
import sys
from ds import *
import utils 
from utils import *
import CONSTANTS
import identify_cois as icoi
import datetime as dt
from itertools import combinations
import logging
import copy

def add_pair(auth1,auth2,rule_id, pair2conflict, attributes = None):
    pair = frozenset([auth1,auth2])
    if pair not in pair2conflict:
        pair2conflict[pair] = Conflicts(pair)
    #   
    pair2conflict[pair].rules.add(rule_id)
    if attributes is not None:
        pair2conflict[pair].__dict__.update(attributes)

def add_senior_junior(junior, senior, to_dict):
    if senior.dblp_ids[0] not in to_dict:
        to_dict[senior.dblp_ids[0]] = set()
    #   
    to_dict[senior.dblp_ids[0]].add(junior.dblp_ids[0])


def add_domain_user(domain, user, to_dict):
    if domain not in to_dict:
        to_dict[domain] = set()
    #
    to_dict[domain].add(user.email_id)


class COI:
    def __init__(self,config,logger,stats_logger,load_light = False):
        self.config = config
        output_folder = config.OUTPUT_FOLDER
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        with open(os.path.join(self.config.OUTPUT_FOLDER,'config.yml'),'w') as fh:
            yaml.dump(self.config.__dict__,fh)
        
        self.fishy_accounts = defaultdict(list)

        self.logger = logger
        self.stats_logger = stats_logger
        
        self.explicit_different_from_primary = set()

        self.dblp_paper_counter = 0
        self.valid_dblp_paper_counter  = 0
        
        self.read_users_submissions_reviewers()
        self.read_dblp_id_universe()
        
        self.create_users()
        self.organize_and_deduplicate_users()
        self.run_sanity_checks()
        self.organize_reviewers()
        
        self.organize_submissions()        
        
        self.dump_reviewer_mail_clusters()
        self.dump_merged_user_information()
        self.domain2user = {}
        self.recreate_domain_to_users_map() 
       
        if load_light:
            self.pid2paper = {}
            self.pid2paper.update(self.subid2paper)
            return
        #
        self.read_dblp_papers()
        self.pid2paper.update(self.subid2paper)
        self.create_users_from_dblp_info()

        utils.populate_paper_coauthor_list_in_users(self.pid2paper, self.dbid2user)
        utils.populate_paper_coauthor_emails_in_users(self.subid2paper, self.mail2user)
        
        self.pair2conflict = {}
        self.senior2juniors = {}
        self.emailpair2conflict = {}
        
        self.exceptions = defaultdict(set)
        self.mail2exceptions = defaultdict(set)

        """
        self.submissions
        self.name_to_ids
        self.dbid2user
        self.pid2paper
        """
   #have all reviewers
    def dump_reviewer_mail_clusters(self):
        rmail2all_mails = {}
        for x,y in self.mail2reviewer.items():
            for z in y.pub_emails:
                if z not in rmail2all_mails:
                    rmail2all_mails[z] = (y.email_id,y.pub_emails)
            #
        if 'REVIEWER_MAIL_CLUSTERS_FILE' not in self.config:
            self.config.REVIEWER_MAIL_CLUSTERS_FILE = os.path.join(self.config.OUTPUT_FOLDER,'reviewer_mail_clusters.pkl')
        with open(self.config.REVIEWER_MAIL_CLUSTERS_FILE,'wb') as fh:
            pickle.dump(rmail2all_mails,fh)
        self.rmail2all_mails = rmail2all_mails
    
    def read_dblp_papers(self):
        self.pid2paper = {}
        if (not os.path.exists(self.config.RELEVANT_PAPERS_DBLP)) or self.config.SHOULD_NOT_USE_EXTRACTED_PAPERS:
            icoi.parse_xml(self.handle_relevant_article)
            self.stats_logger.info("papers processed from dblp, {}\nRelevant papers, {}".format
                              (self.dblp_paper_counter, self.valid_dblp_paper_counter))
            with open(self.config.RELEVANT_PAPERS_DBLP,'wb') as fh:
                pickle.dump(self.pid2paper,fh)
                self.stats_logger.info("dblp papers written to, {}".format
                              (self.config.RELEVANT_PAPERS_DBLP))
        else:
            with open(self.config.RELEVANT_PAPERS_DBLP,'rb') as fh:
                self.pid2paper = pickle.load(fh)
                self.stats_logger.info("{} dblp papers read from {}".format(len(self.pid2paper),self.config.RELEVANT_PAPERS_DBLP))
       
    #Read reviewers
    def organize_reviewers(self):
        self.reviewers_emails = set([x.lower().strip() for x in self.reviewers_emails])
        mail2reviewer = {}
        dbid2reviewer = {}
        mail2reviewer = dict([(x,self.mail2user.get(x,None)) for x in self.reviewers_emails])
        missing_reviewers = [k for k,v in mail2reviewer.items() if v is None]
        #
        for missing_reviewer in missing_reviewers:
            self.logger.warning("Reviewer missing from user list: {}".format(missing_reviewer))
            del mail2reviewer[missing_reviewer]
        #
        valid_reviewers = set(mail2reviewer.keys())
        #populate other mail ids of the reviewers
        for rmail in valid_reviewers:
            for this_mail in mail2reviewer[rmail].pub_emails:
                mail2reviewer[this_mail] = mail2reviewer[rmail]
        #
        #valid_reviewers = set(mail2reviewer.keys())
        missing_dbid_count = 0
        for mail_id,this_user in mail2reviewer.items():
            dbid_list = this_user.dblp_ids
            if dbid_list is not None:
                for this_dbid in dbid_list:
                    dbid2reviewer[this_dbid] = this_user
            else:
                missing_dbid_count += 1
                #self.logger.warning("Missing dbid for reviewer {}".format(mail_id))

        self.stats_logger.info('Total reviewers, {}\nMissing Reviewers, {}\nReviewers Missing dbids, {}'.format(
                                len(self.reviewers_emails), len(missing_reviewers), missing_dbid_count ))

        self.mail2reviewer = mail2reviewer
        self.dbid2reviewer = dbid2reviewer
        self.missing_reviewers = missing_reviewers
        self.valid_reviewers = valid_reviewers

    def organize_submissions(self):
        #Read submissions
        self.subid2paper = dict([(row['Paper ID'], Paper(row, row['Paper ID'], self.mail2user,'sub')) 
                            for index, row in self.submissions.iterrows()])
        
        total_mail_id_missing = 0
        self.paper_ids_missing_user = set()
        for x in self.subid2paper:
            if self.subid2paper[x].num_emails_not_in_users > 0:
                self.paper_ids_missing_user.add(x)
        self.stats_logger.info('submissions, {}\n#submissions missing user, {}'.format(
            len(self.subid2paper), len(self.paper_ids_missing_user)))
        
        with open(self.config.PAPER_IDS_WITH_AUTHOR_NOT_IN_USER,'w') as fh:
            for this_paper_id in self.paper_ids_missing_user:
                missing_emails = self.subid2paper[this_paper_id].emails.difference(self.subid2paper[this_paper_id].emails_in_users)
                print('{},{}'.format(this_paper_id,';'.join(list(missing_emails))), file = fh)

    


    def organize_and_deduplicate_users(self):
        #Create aux datastructures
        self.dbid2user = {}
        self.mail2user = {}
        self.key_mail2user = {}
        num_users_with_multiple_dbids = 0
        for i,this_user in enumerate(self.all_users):
            #key_mail2user[this_user.email_id] = this_user
            #skip = False
            dedup_user = this_user
            #dup_count = 0
            dup_emails = set()
            for pub_mail in this_user.pub_emails:
                if pub_mail in self.mail2user:
                    dup_emails.add(pub_mail)
                    
            #
            if len(dup_emails) > 0:
                dup_emails = list(dup_emails)
                dup_primaries = list(set([self.mail2user[x].email_id for x in dup_emails]))
                if i < self.user_info_count:
                    #self.logger.warning("Duplicate entries for: {}. Primary mail ids: {};{}".format
                    #               (';'.join(dup_emails), ';'.join(dup_primaries), this_user.email_id))
                    if len(dup_primaries) > 1:
                        warn_str = '##'.join(['Primary: {}, Pub: {}'.format(x,';'.join(self.mail2user[x].pub_emails)) for x in dup_primaries])
                        self.logger.warning("More than 3 duplicates: {}, ##, Primary: {}, Pub: {}".format(warn_str,this_user.email_id,';'.join(this_user.pub_emails)))
                dedup_user = self.mail2user[dup_primaries[0]]
                #
                
                if len(dup_primaries) > 1:
                    for existing_id in dup_primaries[1:]:
                        #this will always be true now
                        if dedup_user.email_id != self.mail2user[existing_id].email_id:
                            if i < self.user_info_count:
                                self.logger.warning("Delete one of the two from key_dict: primary1: {}, primary2: {} (delete). This primary: {}".format
                                           (dedup_user.email_id, self.mail2user[existing_id].email_id, this_user.email_id))
                            #
                            #they are actually different
                            is_dbid_unique = dedup_user.merge_details_from(self.mail2user[existing_id])
                            if not is_dbid_unique:
                                self.fishy_accounts['merged_users_with_dbid_mismatch'].append([self.mail2user[existing_id],dedup_user])

                            del self.key_mail2user[self.mail2user[existing_id].email_id]
                #
                #merge details from this_user as well:
                is_dbid_unique = dedup_user.merge_details_from(this_user)
                if not is_dbid_unique:
                    self.fishy_accounts['merged_users_with_dbid_mismatch'].append([this_user,dedup_user])

                
            #        
            self.key_mail2user[dedup_user.email_id] = dedup_user
            for pub_mail in dedup_user.pub_emails:
                self.mail2user[pub_mail] = dedup_user
        
            if dedup_user.dblp_ids is not None:
                if len(dedup_user.dblp_ids) > 1:
                    num_users_with_multiple_dbids += 1
                for k in dedup_user.dblp_ids:
                    self.dbid2user[k] = dedup_user
        
        self.stats_logger.info("Original users, {}\nAfter De-dup users: {}\nMail ids, {}\ndblp ids, {}\nusers with multiple dbids, {}".
                          format(len(self.all_users), len(self.key_mail2user), len(self.mail2user), len(self.dbid2user), num_users_with_multiple_dbids))



    def create_users_from_dblp_info(self):
        dbid_universe = set().union(*[v.authors for k,v in self.pid2paper.items()])
        alldbid2user = dict([ (x,self.dbid2user.get(x,User(x, author_info_map = self.author_info_map, source='dblp',config =self.config))) for x in dbid_universe])
        self.cmtdbid2user = self.dbid2user
        self.dbid2user = alldbid2user 

    def create_users(self):
        self.logger.info("Create User Objects")
        self.all_users = [User(row, self.author_info_map, self.name_to_ids,'user_info',config =self.config) for row in self.users.itertuples()]
        self.user_info_count = len(self.all_users)
        self.logger.info('User Information: {}'.format(len(self.all_users)))
        all_addn_users = [User(row,self.author_info_map,self.name_to_ids,'user_list',config = self.config) for row in self.addn_users_table.itertuples()]
        self.logger.info('User List: {}'.format(len(all_addn_users)))
        self.all_users.extend(all_addn_users)
        self.logger.info('Total: {}'.format(len(self.all_users)))

    def read_users_submissions_reviewers(self):
        self.logger.info("Read Users , submissions and reviewers from the given files")
        h2n = CONSTANTS.h2n
        self.users = pd.read_excel(self.config.USER_FILE)
        self.users.columns = [ h2n[x] if x in h2n else x for x in self.users.columns]
        
        self.submissions = pd.read_excel(self.config.SUBMISSIONS_FILE)
        self.reviewers_emails = set()
        reviewers_list = [pd.read_csv(x,sep='\t') for x in self.config.REVIEWERS_FILE_LIST]
        if len(reviewers_list) > 0:
            self.reviewers =  pd.concat(reviewers_list)
            self.reviewers_emails.update(set(self.reviewers['E-mail']))
            
        ac_reviewers_list = [pd.read_csv(x,sep='\t') for x in self.config.AC_REVIEWERS_FILE_LIST]
        if len(ac_reviewers_list) > 0:
            self.ac_reviewers =  pd.concat(ac_reviewers_list)
            ac_emails = set(self.ac_reviewers['Senior Meta-Reviewer Email'])
            self.reviewers_emails.update(ac_emails)

        self.addn_users_table = [pd.read_csv(x,sep='\t') for x in self.config.ADDITIONAL_USER_FILE_LIST]
        self.addn_users_table = pd.concat(self.addn_users_table)
        self.addn_users_table.columns = [h2n[x] if x in h2n else x for x in self.addn_users_table.columns]


    def read_dblp_id_universe(self):
        self.logger.info("Read the universe of dblp ids. Created externally")
        #name_to_ids: map from author alias to dblpid
        with open(self.config.PERSON_DATA_FILE, 'rb') as f:
            self.name_to_ids, self.author_info_map = pickle.load(f)


    #Functions            
    def handle_relevant_article(self, path, article):
        if path[1][0] == 'www':
            return True # not an article
        
        #dont need globals now. these are now  class variables.
        #global name_to_ids
        #global author_info_map
        #global dblp_paper_counter
        #global valid_dblp_paper_counter
        #global pid2paper
        #
        self.dblp_paper_counter += 1
        # extract some data, return True if entry not relevant
        
        if ('booktitle' not in article and
            'journal' not in article):
            return True
        year = int(article.get('year',"-1"))
        #if year < STARTYEAR or year > ENDYEAR:
        #    return True
        if self.dblp_paper_counter % 1000000 == 0:
            self.logger.info("{} {} papers processed. {} relevant".format(dt.datetime.now(), self.dblp_paper_counter, self.valid_dblp_paper_counter))
        if 'author' not in article:
            return True
    
        pid = path[1][1]['key']
        this_paper = Paper(article,pid,self.name_to_ids,'dblp')
        if this_paper.is_author_in(self.dbid2user) or (year >= (CURRENT_YEAR - 1000)):
            self.pid2paper[pid] = this_paper
            self.valid_dblp_paper_counter += 1
        return True

    def add_dblp_coauthorship_conflict(self):
        for pid,this_paper in self.pid2paper.items():
            if this_paper.num_authors >= RULE_1_1_LIMIT:
                continue
            if this_paper.year <= (CURRENT_YEAR - RULE_1_2_LIMIT):
                continue
            #
            #at a paper published in last 5 years
            for auth1, auth2 in combinations(this_paper.authors,2):
                add_pair(auth1,auth2,RULE_1_2, self.pair2conflict)
        #
        common_paper_added = 0
        for this_dbid,this_user in self.dbid2user.items():
            this_user_papers = this_user.paper_ids
            for coauthor_id in this_user.coauthors:
                if coauthor_id in self.dbid2user:
                    coauthor = self.dbid2user[coauthor_id]
                    #just a sanity check , in case 
                    if coauthor.dblp_ids[0] != this_dbid:
                        coauthor_papers  = coauthor.paper_ids
                        num_common = len(this_user_papers.intersection(coauthor_papers))
                        #rule 1.3
                        if  num_common >= RULE_1_3_LIMIT:
                            for author_id in this_user.dblp_ids:
                                if author_id != coauthor_id:
                                    common_paper_added += 1
                                    add_pair(coauthor_id, author_id, RULE_1_3, self.pair2conflict,{'total_common': num_common})
            
        self.stats_logger.info('rule1.3 additions: {}'.format(common_paper_added)) 

    def identify_student_supervisor_relationship(self):
        if len(self.senior2juniors) == 0:        
            for this_dbid,this_user in self.dbid2user.items():
                this_user_papers = this_user.paper_ids
                for coauthor_id in this_user.coauthors:
                    if coauthor_id in self.dbid2user:
                        coauthor = self.dbid2user[coauthor_id]
                        #just a sanity check , in case 
                        if coauthor.dblp_ids[0] != this_dbid:
                            coauthor_papers  = coauthor.paper_ids
                            #rule 1.4
                            #if this_dbid == senior_id and coauthor_id == junior_id:
                            #    Pdb().set_trace()
                            if this_user.is_senior(coauthor):
                                #print('Is senior', this_user.alias, coauthor.alias)
                                has_early,attributes  = utils.many_early_papers(junior=coauthor, senior=this_user)
                                if has_early:
                                    for author_id in this_user.dblp_ids:
                                        #print('supervisor',attributes,coauthor_id, author_id)
                                        attributes['senior'] = this_dbid
                                        add_senior_junior(senior=this_user,junior=coauthor,to_dict=self.senior2juniors)
                                        add_pair(coauthor_id, author_id, RULE_1_4, self.pair2conflict,attributes)
                                        

        self.stats_logger.info('unique supervisors, {}'.format(
                                    len(self.senior2juniors)))

    def add_conflicts_bw_siblings_and_student_supervisor(self):
        self.identify_student_supervisor_relationship()
        siblings_added = 0
        for supervisor_id, juniors in self.senior2juniors.items():
            for student_id in juniors:
                add_pair(student_id, supervisor_id, RULE_1_4, self.pair2conflict)
        
            for auth1, auth2 in combinations(juniors,2):
                siblings_added += 1 
                add_pair(auth1,auth2,RULE_1_5, self.pair2conflict,{'common_supervisor': supervisor_id})
        
        #pickle.dump(list(senior2juniors.keys()), open('seniors.pkl','wb'))
        self.stats_logger.info('siblings_added, {}'.format(siblings_added))

    def log_supervisor_students(self):
        #senior2juniors
        """log senior to juniors in SUPERVISOR_STUDENTS_OUTFILE"""
        all_supervisors = list(self.senior2juniors.keys())
        all_supervisors.sort(key=lambda x: len(self.senior2juniors[x]), reverse = True)
        with open(self.config.SUPERVISOR_STUDENTS_OUTFILE,'w') as fh:
            print('supervisor,#students,students(; seperated)',file =fh)
            for x in all_supervisors:
                student_dbids = list(self.senior2juniors[x])
                students = ';'.join(['{}({})'.format(
                    self.author_info_map[y].get('aliases',[None])[0], y) for y in student_dbids])
                print('{}({}),{},{}'.format(
                    self.author_info_map[x].get('aliases',[None])[0], self.dbid2user[x].dblp_ids[0], len(self.senior2juniors[x]), students), file = fh)
            
        #correct conflict domains
        #for mail_id,  this_user in mail2user.items():
        #    this_user.conflict_domains = set([x.strip() for x in list(this_user.conflict_domains)])
        #    this_user.conflict_domains = this_user.conflict_domains.difference(IGNORE_DOMAINS)
          

    def global_domain_normalization(self):
        old_domains = list(self.domain2user.keys())
        new_domains = copy.deepcopy(old_domains)
        
        for cpre in ['cs.','ee.','ece.','math.','maths.','cse.']:
            new_domains = [x[len(cpre):] if (x.startswith(cpre) and (x[len(cpre):] in self.domain2user)) else x for x in new_domains]
            self.logger.info('{}, #Earlier:  {}, #Now: {}'.format(cpre,len(self.domain2user),len(set(new_domains))))
        
        for cext in ['.cn','.ca','.in','.sg','.tw']:
            new_domains = [x+ cext if ((x+cext) in self.domain2user) else x for x in new_domains]
            self.logger.info('{}, #Earlier:  {}, #Now: {}'.format(cext,len(self.domain2user),len(set(new_domains))))
        
        old2new_domain = dict([(k,v) for k,v in zip(old_domains,new_domains)])
        for mail,this_user in self.key_mail2user.items():
            this_user.remap_domains(old2new_domain)
       
        #
    
    def recreate_domain_to_users_map(self):
        #Pdb().set_trace()
        #self.domain2user.clear() 
        self.domain2user = {} 
        self.reset_users_domains()
        #Pdb().set_trace()
        self.create_domain_to_users_map()
        self.logger.info("Total number of domains: {}".format(len(self.domain2user)))
        self.global_domain_normalization()
        self.create_domain_to_users_map()
        self.logger.info("#Domains after global normalization: {}".format(len(self.domain2user)))


    def reset_users_domains(self):
        ignore_domains_for = set(self.config.get('IGNORE_DOMAINS_FOR_USERS',[]))
        for mail,this_user in self.key_mail2user.items():
            this_user.reset_domains()
        
        for mail in ignore_domains_for:
            self.mail2user[mail].original_conflict_domains = set()
            logger.warning("Deleting all explicit conflict domains for : {}".format(mail))
            this_user.reset_domains()

        #

    def create_domain_to_users_map(self):
        self.domain2user = {}
        for mail_id, this_user in self.key_mail2user.items():
            for this_domain in this_user.conflict_domains:
                add_domain_user(domain=this_domain.strip().lower(), user = this_user, to_dict = self.domain2user)
            
        self.stats_logger.info('total domains, {}'.format(len(self.domain2user)))
        
        self.all_domains = list(self.domain2user.keys())
        self.all_domains.sort(key = lambda x: len(self.domain2user[x]), reverse=True)
        
        self.stats_logger.info('Top 10 domains: {}'.format(str([(x,len(self.domain2user[x])) for x in self.all_domains[:10]])))


    def add_same_domain_pairs(self):
        same_domain_pairs_added = 0
        #large_domains = set()
        for domain, domain_users in self.domain2user.items():
            for mail1, mail2 in combinations(domain_users,2):
                same_domain_pairs_added += 1
                add_pair(mail1,mail2,RULE_2_1, self.emailpair2conflict)

        self.stats_logger.info("same_domain_pairs_added, {}".format(same_domain_pairs_added))

    def add_explicit_conflicts(self):
        self.explicit_different_from_primary = set()
        missing_explicit_conflicts = []
        num_explicit_conflicts_added = 0
        for mail_id, this_user in self.key_mail2user.items():
            for this_explicit_conflict_email in this_user.explicit_conflict_emails:
                if this_explicit_conflict_email.lower() in self.mail2user:
                    num_explicit_conflicts_added += 1
                    #add_pair(mail_id, this_explicit_conflict_email, RULE_2_2, self.emailpair2conflict)
                    add_pair(mail_id, self.mail2user[this_explicit_conflict_email.lower()].email_id, RULE_2_2, self.emailpair2conflict)
                    if this_explicit_conflict_email.lower() != self.mail2user[this_explicit_conflict_email.lower()].email_id:
                        self.explicit_different_from_primary.add((mail_id, this_explicit_conflict_email.lower(), self.mail2user[this_explicit_conflict_email.lower()].email_id))
                        #self.logger.info("explicit different from primary: {},  {}".format(this_explicit_conflict_email, self.mail2user[this_explicit_conflict_email].email_id))
                else:
                    missing_explicit_conflicts.append([mail_id,this_explicit_conflict_email.lower()])
        #
        self.missing_explicit_conflicts = missing_explicit_conflicts
        self.stats_logger.info("num_explicit_conflicts_added, {}".format(num_explicit_conflicts_added))
        self.stats_logger.info("num missing explicit conflicts, {}".format(len(missing_explicit_conflicts)))

    #Exceptions
    """
    Exceptions from rule 2
    """
    def populate_exceptions_to_domain_and_explicit_conflicts(self):
        self.exceptions = defaultdict(set)
        self.mail2exceptions = defaultdict(set)

        exceptions = self.exceptions
        mail2exceptions = self.mail2exceptions
        mail2user = self.mail2user

        for mail_id, this_user in self.key_mail2user.items():
            explicit_conflict_emails = this_user.explicit_conflict_emails
            if len(this_user.original_conflict_domains) >= EXCEPTION_2_1_LIMIT:
                self.exceptions[EXCEPTION_2_1].add(mail_id)
            #
            #
            this_user.explicit_conflicts_in_users = set([x for x in explicit_conflict_emails if x in self.mail2user])
            this_user.coauthors_in_users = set([x for x in this_user.coauthors if x in self.dbid2user])
            #this_user.explicit_conflicts_in_users_non_coauthors = set()
            coauthor_mail_ids = [self.dbid2user[x].pub_emails for x in this_user.coauthors if x in self.dbid2user]

            #allpub mails TODO: what if couthor is not a user? 
            #Should we also take intersection of conflict mail ids with the universe of all mail ids that we have?
            #coauthor_mail_ids = [dbid2user[x].pub_emails for x in this_user.coauthors_in_users if x in dbid2user]
            coauthor_mail_ids = set()
            if len(coauthor_mail_ids) > 0:
                coauthor_mail_ids = set.union(*coauthor_mail_ids)

            #
            #if len(explicit_conflict_emails.difference(coauthor_mail_ids)) >= EXCEPTION_2_2_LIMIT:
            this_user.explicit_conflicts_in_users_non_coauthors = this_user.explicit_conflicts_in_users.difference(coauthor_mail_ids)
            if len(this_user.explicit_conflicts_in_users_non_coauthors) >= EXCEPTION_2_2_LIMIT:
                self.exceptions[EXCEPTION_2_2].add(mail_id)
                    #
            this_user.asymmetric_conflicts = set()
            for coi_email in explicit_conflict_emails:
                if coi_email in self.mail2user:
                    coi_user = self.mail2user[coi_email]
                    coi_explicit_conflict_emails = coi_user.explicit_conflict_emails
                    if len(this_user.pub_emails.intersection(coi_explicit_conflict_emails)) == 0:
                        this_user.asymmetric_conflicts.add(self.mail2user[coi_email].email_id)
                        #asymmetric_count += 1

            #
            if len(this_user.asymmetric_conflicts) >= EXCEPTION_2_3_LIMIT:
                self.exceptions[EXCEPTION_2_3].add(mail_id)

        for exception_type, mail_ids in self.exceptions.items():
            for this_mail in mail_ids:
                self.mail2exceptions[this_mail].add(exception_type)
        #

        e21_mails = list(exceptions[EXCEPTION_2_1])

        e21_mails.sort(key = lambda x: len(mail2user[x].original_conflict_domains), reverse=True)

        e21_details = [(x,len(mail2user[x].original_conflict_domains), ';'.join(list(mail2user[x].original_conflict_domains)))
                                for x in e21_mails]

        e22_mails = list(exceptions[EXCEPTION_2_2])
        e22_mails.sort(key = lambda x: len(mail2user[x].explicit_conflicts_in_users_non_coauthors), reverse=True)
        e22_details = [(x,len(mail2user[x].explicit_conflicts_in_users_non_coauthors),
                        len(mail2user[x].coauthors), len(mail2user[x].coauthors_in_users),
                         len(mail2user[x].explicit_conflict_emails),
                        len(mail2user[x].explicit_conflicts_in_users), 
                        ';'.join(list(mail2user[x].explicit_conflicts_in_users_non_coauthors)))  for x in e22_mails]

        e23_mails = list(exceptions[EXCEPTION_2_3])
        e23_mails.sort(key= lambda x: len(mail2user[x].asymmetric_conflicts), reverse=True)
        e23_details = [(x,len(mail2user[x].asymmetric_conflicts),
                        len(mail2user[x].explicit_conflict_emails),
                        len(mail2user[x].explicit_conflicts_in_users),
                       ';'.join(mail2user[x].asymmetric_conflicts)) for x in e23_mails]
        #
        self.e21_mails = e21_mails
        self.e21_details = e21_details
        self.e22_mails = e22_mails
        self.e22_details = e22_details
        self.e23_mails  = e23_mails
        self.e23_details = e23_details


    #mail2user['chinmoy.dutta@gmail.com'].conflict_domains
    #exceptions['E21']
    #exceptions
    def log_exceptions_to_domain_conflicts(self):
        exceptions = self.exceptions    
        with open(self.config.EXCEPTIONS_CSV_OUTFILE,'w') as fh:
            print("Different exception counts: ",file=fh)
            e_counts = [[x,len(exceptions[x])] for x in exceptions.keys()] 
            print('\n'.join(['{}, {}'.format(x[0],x[1])  for x in e_counts]), file = fh)
            self.stats_logger.info(str(e_counts))
            print("Limits:####### ",file =fh)
            print('{},{}'.format(EXCEPTION_2_1, EXCEPTION_2_1_LIMIT),file=fh)
            print('{},{}'.format(EXCEPTION_2_2, EXCEPTION_2_2_LIMIT),file=fh)
            print('{},{}'.format(EXCEPTION_2_3, EXCEPTION_2_3_LIMIT),file=fh)

            if len(self.e21_details) > 0:
                print("##############",file =fh)
                print('Mail ID, #Domains, Domains', file = fh)
                print('\n'.join([','.join(map(str,x)) for x in self.e21_details]),file = fh)
            #
            if len(self.e22_details) > 0:
                print("##############",file =fh)
                print('Mail ID, #non coauthor conflicts, #coauthors, #coauthors In Users, #explicit conflicts, #conflicts in users, non-coauth conf', file = fh)
                print('\n'.join([','.join(map(str,x)) for x in self.e22_details]),file = fh)

            if len(self.e23_details) > 0:
                print("##############",file =fh)
                print('Mail ID, #asymmetric conflicts, #explicit conflicts, #conflicts in users, asymm conflicts', file = fh)
                print('\n'.join([','.join(map(str,x)) for x in self.e23_details]),file = fh)


    #rule 3
    def add_coauthor_aaai21_conflicts(self):
        for pid, this_paper in self.subid2paper.items():
            for email1, email2 in combinations(this_paper.emails_in_users,2):
                #print(auth1,auth2)
                add_pair(email1,email2,RULE_3_1, self.emailpair2conflict)


    """
    For logging the final output. There are 2 dicts: 
    pair2conflict (maping dbid to conflicts) and
    emailpair2conflict(mapping email ids to conflicts)
    """
    def organize_all_conflicts(self):
        self.dbid2conflict_pair = defaultdict(set)
        for pair in self.pair2conflict:
            for dbid in pair:
                self.dbid2conflict_pair[dbid].add(pair)

        self.email2conflict_pair = defaultdict(set)
        for emailpair in self.emailpair2conflict:
            for this_email in emailpair:
                self.email2conflict_pair[this_email].add(emailpair)

    def attribution_of_conflicts(self):
        #self.organize_all_conflicts()
        mail2user  = self.mail2user
        valid_reviewers = self.valid_reviewers
        #output
        #iterate over papers. and take union of all conflicting mail ids
        num_missing_for_dbid_conflict = 0
        num_missing_for_email_conflict = 0
        table_of_conflicts = {'pid':[],'amail': [], 'remail':[], 'rules':[]}
        with open(self.config.CONFLICT_CSV_OUTFILE,'w') as fh:
            for pid, this_paper in self.subid2paper.items():
                conflicting_reviewers = defaultdict(set)
                #dict mapping conflicting reviewer to the set of rules for him
                #from dbid conflicts
                for auth_dbid in this_paper.authors:
                    conflicting_pairs = self.dbid2conflict_pair[auth_dbid]
                    #set of all conflicting pairs
                    for this_conflicting_pair in conflicting_pairs:
                        this_conflict = self.pair2conflict[this_conflicting_pair]
                        other_dbid = get_other(this_conflicting_pair,auth_dbid)
                        amail = self.dbid2user[auth_dbid].email_id
                        if other_dbid in self.dbid2reviewer:
                            for temp_id in self.dbid2user[other_dbid].pub_emails.intersection(valid_reviewers):
                                conflicting_reviewers[(amail,temp_id)].update(this_conflict.rules)
                        else:
                            #pass
                            num_missing_for_dbid_conflict += 1
                            #self.logger.warning("missing dbid for conflict, {}".format(other_dbid))
                    #
                #
                #from email conflicts
                for auth_mail_id in this_paper.emails_in_users:
                    #self conflicts:
                    amail = self.mail2user[auth_mail_id].email_id
                    for temp_id in self.mail2user[auth_mail_id].pub_emails.intersection(valid_reviewers):
                        conflicting_reviewers[(amail,temp_id)].add(RULE_0)

                    conflicting_emailpairs  = self.email2conflict_pair[auth_mail_id]
                    for this_conflicting_emailpair in conflicting_emailpairs:
                        this_emailconflict = self.emailpair2conflict[this_conflicting_emailpair]
                        other_email = get_other(this_conflicting_emailpair,auth_mail_id)
                        if other_email in self.mail2reviewer:
                            for temp_id in mail2user[other_email].pub_emails.intersection(valid_reviewers):
                                conflicting_reviewers[(amail,temp_id)].update(this_emailconflict.rules)
                        else:
                            num_missing_for_email_conflict += 1


                #conflicting_reviewers = conflicting_reviewers.intersection(all_reviewers) 
                for this_conflicting_reviewer in conflicting_reviewers:
                    this_rules = ';'.join(list(conflicting_reviewers[this_conflicting_reviewer]))
                    table_of_conflicts['pid'].append(pid)
                    table_of_conflicts['remail'].append(this_conflicting_reviewer[1])
                    table_of_conflicts['amail'].append(this_conflicting_reviewer[0])
                    table_of_conflicts['rules'].append(conflicting_reviewers[this_conflicting_reviewer])
                    print('{},{},{},{}'.format(pid,this_conflicting_reviewer[0],this_conflicting_reviewer[1],this_rules),file=fh)


        self.stats_logger.warning('number of conflicting dbid and email ids missing from reviewers list: {}, {}'.
                             format(num_missing_for_dbid_conflict, num_missing_for_email_conflict))

        #dbid2reviewer['142/1013'].dblp_ids
        table_of_conflicts = pd.DataFrame(table_of_conflicts)
        table_of_conflicts['reason'] = 2
        table_of_conflicts[['pid','remail','reason']].drop_duplicates().to_csv(self.config.CONFLICTS_FILE_FOR_DINESH,index=False)
        self.table_of_conflicts = table_of_conflicts
        return table_of_conflicts



    def write_and_return_all_conflicts(self):
        #self.organize_all_conflicts()
        mail2user  = self.mail2user
        valid_reviewers = self.valid_reviewers
        #output
        #iterate over papers. and take union of all conflicting mail ids
        num_missing_for_dbid_conflict = 0
        num_missing_for_email_conflict = 0
        table_of_conflicts = {'pid':[], 'remail':[], 'rules':[]}
        with open(self.config.CONFLICT_CSV_OUTFILE,'w') as fh:
            for pid, this_paper in self.subid2paper.items():
                conflicting_reviewers = defaultdict(set)
                #dict mapping conflicting reviewer to the set of rules for him
                #from dbid conflicts
                for auth_dbid in this_paper.authors:
                    conflicting_pairs = self.dbid2conflict_pair[auth_dbid]
                    #set of all conflicting pairs
                    for this_conflicting_pair in conflicting_pairs:
                        this_conflict = self.pair2conflict[this_conflicting_pair]
                        other_dbid = get_other(this_conflicting_pair,auth_dbid)
                        if other_dbid in self.dbid2reviewer:
                            for temp_id in self.dbid2user[other_dbid].pub_emails.intersection(valid_reviewers):
                                conflicting_reviewers[temp_id].update(this_conflict.rules)
                        else:
                            #pass
                            num_missing_for_dbid_conflict += 1
                            #self.logger.warning("missing dbid for conflict, {}".format(other_dbid))
                    #
                #
                #from email conflicts
                for auth_mail_id in this_paper.emails_in_users:
                    #self conflicts:
                    for temp_id in self.mail2user[auth_mail_id].pub_emails.intersection(valid_reviewers):
                        conflicting_reviewers[temp_id].add(RULE_0)

                    conflicting_emailpairs  = self.email2conflict_pair[auth_mail_id]
                    for this_conflicting_emailpair in conflicting_emailpairs:
                        this_emailconflict = self.emailpair2conflict[this_conflicting_emailpair]
                        other_email = get_other(this_conflicting_emailpair,auth_mail_id)
                        if other_email in self.mail2reviewer:
                            for temp_id in mail2user[other_email].pub_emails.intersection(valid_reviewers):
                                conflicting_reviewers[temp_id].update(this_emailconflict.rules)
                        else:
                            num_missing_for_email_conflict += 1


                #conflicting_reviewers = conflicting_reviewers.intersection(all_reviewers) 
                for this_conflicting_reviewer in conflicting_reviewers:
                    this_rules = ';'.join(list(conflicting_reviewers[this_conflicting_reviewer]))
                    table_of_conflicts['pid'].append(pid)
                    table_of_conflicts['remail'].append(this_conflicting_reviewer)
                    table_of_conflicts['rules'].append(conflicting_reviewers[this_conflicting_reviewer])
                    print('{},{},{}'.format(pid,this_conflicting_reviewer,this_rules),file=fh)


        self.stats_logger.warning('number of conflicting dbid and email ids missing from reviewers list: {}, {}'.
                             format(num_missing_for_dbid_conflict, num_missing_for_email_conflict))

        #dbid2reviewer['142/1013'].dblp_ids
        table_of_conflicts = pd.DataFrame(table_of_conflicts)
        table_of_conflicts['reason'] = 2
        table_of_conflicts[['pid','remail','reason']].to_csv(self.config.CONFLICTS_FILE_FOR_DINESH,index=False)
        self.table_of_conflicts = table_of_conflicts
        return table_of_conflicts



    def get_pids_with_high_conflicts(self):
        table_of_conflicts = self.table_of_conflicts
        with open(self.config.ANALYSIS_OF_CONFLICTS,'a') as fh:
            rcount = table_of_conflicts.groupby(['pid']).agg({'remail':'count'})
            table_of_conflicts['rules_count'] = table_of_conflicts['rules'].apply(len)
            set_of_all_conflicts = set().union(*table_of_conflicts['rules'])
            for x in set_of_all_conflicts:
                table_of_conflicts[x] = table_of_conflicts['rules'].apply(lambda y: float(x in y))

            rcount= rcount.reset_index()
            high_conflict_pids = rcount[rcount['remail'] >= 1000]['pid']

            print('{} Papers with more than 1000 conflicts: {}'.format(len(high_conflict_pids),
                                        ('; '.join(map(str,high_conflict_pids)))), file =fh)

            self.stats_logger.warning('#Papers with more than 1000 conflicts: {}'.format(
                                    len(high_conflict_pids)))
            print(table_of_conflicts.describe(),file =fh)
            print(rcount.describe(),file =fh)

        high_conflicts = table_of_conflicts.loc[table_of_conflicts['pid'].isin(high_conflict_pids)]
        high_conflicts.to_csv(self.config.HIGH_CONFLICTS_FILE)
        return high_conflict_pids

    #coauthor distance. we care about co-author and 1hop distances only
    def compute_coauthor_distance_for_ilp(self):
        self.logger.info("STart computing co-author distance between reviewers")
        reviewers_emails = self.reviewers_emails
        mail2user = self.mail2user
        dbid2user = self.dbid2user
        distance1 = defaultdict(set)
        distance2 = defaultdict(set)
        for r in reviewers_emails:
            this_user = mail2user.get(r,None)
            #if r == r1:
            #    Pdb().set_trace()
            if this_user is not None:
                #
                nhbrs_dbids = this_user.coauthors
                nhbrs_emails = this_user.coauthor_emails
                #there may be overlap b/w the two types of nhbrs
                #we will disambiguate based on user.email_id
                #

                #distance 2
                for this_nhbr_dbid in nhbrs_dbids:
                    distance1[r].update(dbid2user[this_nhbr_dbid].pub_emails.intersection(reviewers_emails))
                    for x in dbid2user[this_nhbr_dbid].coauthors:
                        distance2[r].update(dbid2user[x].pub_emails.intersection(reviewers_emails))
        #                 if r == r1 and r2 in distance2[r]:
        #                     Pdb().set_trace()
        #                     print('debugging')

                    for x in dbid2user[this_nhbr_dbid].coauthor_emails:
                        distance2[r].update(mail2user[x].pub_emails.intersection(reviewers_emails) if x in mail2user else set())
        #                 if r == r1 and r2 in distance2[r]:
        #                     Pdb().set_trace()
        #                     print('debugging')

                for this_nhbr_email in nhbrs_emails:                
                    if this_nhbr_email in mail2user:
                        distance1[r].update(mail2user[this_nhbr_email].pub_emails.intersection(
                                    reviewers_emails))
                        for x in mail2user[this_nhbr_email].coauthors:
                            distance2[r].update(dbid2user[x].pub_emails.intersection(reviewers_emails))
        #                     if r == r1 and r2 in distance2[r]:
        #                         Pdb().set_trace()
        #                         print('debugging')

                        for x in mail2user[this_nhbr_email].coauthor_emails:
                            distance2[r].update(mail2user[x].pub_emails.intersection(
                                reviewers_emails) if x in mail2user else set())
        #                     if r == r1 and r2 in distance2[r]:
        #                         Pdb().set_trace()
        #                         print('debugging')

                #
                distance2[r] = distance2[r].difference(distance1[r])
                distance1[r]= distance1[r].difference(set([r]))
                distance2[r] = distance2[r].difference(set([r]))
                #


        with open(self.config.COAUTHOR_DISTANCE_FILE,'wb') as fh:
            pickle.dump([distance1,distance2],fh)
        #
        self.logger.info("End computing co-author distance between reviewers")
        return distance1,distance2


    def get_reviewer_to_own_papers(self):
        r2pids = {}
        submission_pids = set(list(self.subid2paper.keys()))
        for rmail,this_user in self.mail2reviewer.items():
            r2pids[rmail] = this_user.paper_ids.intersection(submission_pids)

        with open(self.config.REVIEWER_TO_PID_FILE,'wb') as fh:
            pickle.dump(r2pids,fh)
        return r2pids


    def run_sanity_checks(self):
        key_mail2user = self.key_mail2user
        self.users_with_same_dbids = defaultdict(list)
        self.fishy_accounts['dif_users_with_same_dbids'] = []
        for k,v in self.key_mail2user.items():
            if v.dblp_ids is not None:
                self.users_with_same_dbids[v.dblp_ids[0]].append(k)
        #   
        for did,v in self.users_with_same_dbids.items():
            if len(v) > 1:
                self.fishy_accounts['dif_users_with_same_dbids'].append([key_mail2user[x] for x in v])


        def print_list_of_group_of_users_details(falist,fh):
            for fa in falist:
                msg = ',\t,'.join(['{}, {}, {}'.format(x.alias,x.email_id,x.dblp_ids[0]) for x in fa])
                #self.logger.warning(msg)
                print(msg,file =fh)

        with open(self.config.FISHY_ACCOUNTS_FILE,'w') as fh:
            tk = 'dif_users_with_same_dbids'
            msg1 = '{} Groups of accounts with same dblp id.'.format(
                len(self.fishy_accounts[tk]))
            self.logger.warning(msg1)
            print(msg1, file = fh)
            header = 'alias1, mail1 ,dblp id1,\t,alias1, mail2, dblp id2, ...'
            print(header,file =fh)
            #self.logger.warning(header)
            print_list_of_group_of_users_details(self.fishy_accounts[tk],fh)
            #
            tk = 'merged_users_with_dbid_mismatch'
            msg2 = '{} Pairs of accounts merged with different dblp ids.'.format(
                    len(self.fishy_accounts[tk]))
            self.logger.warning(msg2)
            print(msg2,file = fh)
            print(header,file=fh)
            #self.logger.warning(header)
            print_list_of_group_of_users_details(self.fishy_accounts[tk],fh)





    def get_all_paper_reviewer_conflicts(self):
        self.add_dblp_coauthorship_conflict()
        self.identify_student_supervisor_relationship()
        self.add_conflicts_bw_siblings_and_student_supervisor()
        self.log_supervisor_students()
        self.add_same_domain_pairs()
        self.add_explicit_conflicts()
        self.populate_exceptions_to_domain_and_explicit_conflicts()
        self.log_exceptions_to_domain_conflicts()
        self.add_coauthor_aaai21_conflicts()
        
        self.organize_all_conflicts()
        self.table_of_conflicts_woauthor = self.write_and_return_all_conflicts()
        self.table_of_conflicts_with_author = self.attribution_of_conflicts()
        _ = self.get_reviewer_to_own_papers()
        _ = self.compute_coauthor_distance_for_ilp()
        
        high_coi_pids = self.get_pids_with_high_conflicts()
        self.stats_logger.info('##### END AT {} #######'.format(dt.datetime.now()))
        self.logger.info('##### END AT {} #######'.format(dt.datetime.now()))



    def domain_conflict_stats_by_author(self):
        key_mail2user = self.key_mail2user
        for k,v in key_mail2user.items():
            #v.temp_cache.clear()
            v.temp_cache = {}
            v.temp_cache['domain_conflict_users'] = set()
    
        for domain,domain_users in self.domain2user.items():
            for mail1, mail2 in combinations(domain_users,2):
                u1pub_mails = key_mail2user[mail1].pub_emails
                u2pub_mails = key_mail2user[mail2].pub_emails
                key_mail2user[mail1].temp_cache['domain_conflict_users'].update(u2pub_mails)
                key_mail2user[mail2].temp_cache['domain_conflict_users'].update(u1pub_mails)
        #
        for mail,this_user in key_mail2user.items():
            tuc = this_user.temp_cache
            tuc['domain_conflict_users'] = tuc['domain_conflict_users'].intersection(
                self.reviewers_emails)
            #
            primary_conflicts = set([self.mail2user[x].email_id for x in tuc['domain_conflict_users']])
            #tuc['domain_conflict_users'].clear()
            tuc['domain_conflict_users'] = set()
            tuc['domain_conflict_users'].update(primary_conflicts)
    
        all_authors = set()
        for pid,this_paper in self.subid2paper.items():
            all_authors.update(this_paper.emails_in_users)
    
        act = {'mail':[],'num_dom_conflicts':[],'orig_num_domains':[],'num_domains': [], 'domains':[]}
    
        for this_author in all_authors:
            act['mail'].append(this_author)
            act['num_dom_conflicts'].append(len(key_mail2user[this_author].temp_cache['domain_conflict_users']))
            act['orig_num_domains'].append(len(key_mail2user[this_author].original_conflict_domains))
            act['num_domains'].append(len(key_mail2user[this_author].conflict_domains))
            act['domains'].append(copy.deepcopy(key_mail2user[this_author].conflict_domains))
    
        act = pd.DataFrame(act)
        act['avg_domain_size'] = act['domains'].apply(lambda x: np.mean([len(self.domain2user[y]) for y in x if y in self.domain2user]))
        print(act[act['num_domains'] == 0].shape)
        print(act[act['num_domains'] > 0].describe())
        return act

    def dump_merged_user_information(self):
        already_covered = set()
        mail2prim = {'email':[],'prim_email':[]}
        for x,y in self.mail2user.items():
            for z in y.pub_emails:
                if z not in already_covered:
                    already_covered.add(z)
                    mail2prim['prim_email'].append(y.email_id)
                    mail2prim['email'].append(z)

        #

        mail2prim = pd.DataFrame(mail2prim)

        h2n = CONSTANTS.h2n

        self.users['case_sensitive_email'] = self.users['email']
        self.users.columns = [ h2n[x] if x in h2n else x for x in self.users.columns]
        self.users['email'] = self.users['email'].apply(lambda x: x.lower())

        users = self.users.merge(mail2prim,how='left',on='email')
        unique_primary = users.groupby('prim_email').agg({'email':'count'}).sort_values('email',ascending=False)

        to_merge = ['gs_id','ss_id','dblp_id','primary_area','secondary_area','pub_emails','conflict_domains','q10']
        for this_col in to_merge:
            users['merged.'+this_col] = users[this_col]

        #ACL Scorer: Column F (Semantic Scholar URL)
        #Subject Area Scorer: Column L (Primary Subject Area) and Column M (Secondary Subject Areas)
        #Reviewer Info (A4): Column H (Publication Emails), Column J (Conflict Domains) and Column P (Q10)
        #'gs_id','ss_id','dblp_id','primary_area','secondary_area','pub_emails','conflict_domains','q10'


        def merge_info(sdf):
            if len(sdf) > 1:
                for this_col in to_merge:
                    merged_str = '##'.join(list(sdf[this_col].unique().astype(str)))
                    sdf['merged.'+this_col] = merged_str
            #
            return sdf

        merged_users  = users.groupby('prim_email').apply(merge_info)
        n2h = dict([(v,k) for k,v in h2n.items()])
        n2h['email'] = 'Email'
        n2h['fname'] = 'First Name'
        merged_users.columns = [ n2h[x] if x in n2h else x for x in merged_users.columns]
        merged_users.to_csv(os.path.join(self.config.OUTPUT_FOLDER,'merged_user_information.csv'),sep=',', index=False)
        return merged_users



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


def get_loggers(config):
    logging.basicConfig(filename=os.path.join(config.OUTPUT_FOLDER,'COI_NOTEBOOK.log'),
                        filemode= 'w', level=logging.INFO,format='%(asctime)s, %(levelname)s, %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger = logging.getLogger("NB")
    logger.addHandler(console_handler)
    
    stats_handler = logging.FileHandler(os.path.join(config.OUTPUT_FOLDER,'stats.log'))
    formatter = logging.Formatter('%(asctime)s, %(levelname)s, %(message)s')
    stats_handler.setFormatter(formatter)
    stats_logger = logging.getLogger('stats')
    stats_logger.addHandler(stats_handler)
    stats_logger.addHandler(console_handler)
    
    stats_logger.info('##### START AT {} #######'.format(dt.datetime.now()))
    logger.info('##### START AT {} #######'.format(dt.datetime.now()))
    logger.info(pformat(config))
    return logger, stats_logger

if __name__ == '__main__':
    args = get_args()
    config_file = args.config
    with open(config_file,'rb') as fh:
        config = yaml.safe_load(fh)
    
    config = Map(config)
    if not os.path.exists(config.OUTPUT_FOLDER):
        os.mkdir(config.OUTPUT_FOLDER)

    logger, stats_logger = get_loggers(config)
    coi = COI(config,logger,stats_logger,load_light  = False)
    coi.get_all_paper_reviewer_conflicts()
