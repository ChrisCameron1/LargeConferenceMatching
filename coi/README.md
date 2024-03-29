
# COI Computation

To start the COI computation, run the following command:
```
python coi_module.py --config_file config.yml
```

### Input:
A yaml file containing the following fields:
1. USER_FILE: xlsx file: contains the universe of users considered. Reviewers should be a subset of this.
2. REVIEWERS_FILE_LIST: xlsx file: list of file names containing reviewers data
3. SUBMISSIONS_FILE: xlsx file: contains list of all submissions
4. PERSON_DATA_FILE: pkl file:    list of:  
                                       A. alias to dblp id dict (name_to_ids) 
                                       B. dblp id to author info dict  (author_info_map). 
                                       Generated by ``python identify_cois.py --in_file data/dblp/dblp_person_data.pkl --out_file data/dblp/dblp_person_data.pkl``

5. CONFLICT_CSV_OUTFILE: filename where conflicts are written. 
        Format: Paper ID,Conflicting Reviewer Mail ID, Rules. 
                Rules: is a list of rules joined by ;
6. EXCEPTIONS_CSV_OUTFILE: filename where exceptions are written.
        Format: Reviewer Mail ID,Exceptions
                Exceptions: list of exception ids joined by ;

#### How to get input files from CMT?
1. REVIEWERS_FILE_LIST 
        User -> Reviewers Click. On the Manage Reviewers page, select AAAI2021 Track. Go to Actions -> Export Reviewers. This will download Reviewers-1.txt. Repeat for AISI Track. This will download Reviewers-2.txt

2. Repeat for Meta-Reviewers 
 
3. AC_REVIEWERS_FILE_LIST  
    Repeat for Senior Meta Reviewers: it downloads MetaReviewersSubjectAreas.txt. There are no subject areas for AISI track, hence only one file.

4. For User-Information.xls: User -> Conference Users Click. -> Actions -> Export -> User Information.xls and User List.txt
    Unfortunately,  pandas library doesn't read it as is. Hence, we open it in excel and save as .xlsx format.

5. For Papers.xlsx: Download papers from Submissions -> Actions -> Export -> Excel. 
    This will download Papers.xls. Before saving in .xlsx format, we first delete AISI papers sheet after copying paper details (from row 4 onwards) from AISI Track sheet to the AAAI2021 sheet. Finally, delete the first two rows so that the header is the first row and then save to .xlsx format.



### Output:
1. Written to files CONFLICT_CSV_OUTFILE and EXCEPTIONS_CSV_OUTFILE. In addition, a bunch of other files are also written. See ``coi_module.COI.get_all_paper_reviewer_conflicts()`` for more details.
2. Logging: file COI_NOTEBOOK.log

### Modules:
``ds.py``:  Contains three classes: 
        User, Paper, Conflicts
        
``CONSTANTS.py``: Similar to a config file.
              Contains exception ids, rule ids, limits 

``coi_module.py``: master class which binds everything together and has a function for each rule.

### Flow:

#### Data Structures:
All datastructures are defined in the class: coi_module.COI
1. all_users: Create a list of User objects. Each User contains:
        A email_id: primary email id using which user registerd.
        B. pub_emails: all publication emails provided by the user. Includes email_id as well.
        C. fname: first name; lname: last name; alias: fname + ' ' + lname.
        D. dblp_ids: list of all dblp ids for the user.
                    Provided by the user
                    Filtered using author_info_map that is extraced from dblp ids.
                    If not found, populated from name_to_ids by matching the alias: probably incorrect: Remove it.
       
      Fields below are populated after reading all relevant papers from  dblp.xml file and all submissions.
      Function 'populate_paper_coauthor_list_in_users'  in utils does the task
        E. paper_ids: set of all paper ids of the author. Extracted from Papers object. 
        F. paper_list: list of all the papers of the author, sorted by the year.  
        G. coauthors: Set of dblp ids of the couthors
        H. conflict_domains
        I. explicit conflicts

2. dbid2user, mail2user, key_mail2user: dictionaries containing different views of all users.
                 mail2user: all user.pub_mails are also keys
                 key_mail2user: only user.email_id are the keys
                 
3. subid2papers: dict storing all the submissions. key: Paper ID. value: object of class Paper, contains:
       A. source: 'sub' or 'dblp': whether current submission or fetched from dblp?
       B. id: Unique Paper ID
       C. authors: set of dblp ids of all the authors. 
               C.1. Populated by finding 'User' object for each email id provided in 'Author Emails' field.
               C.2. Warning is raised if 'User' not found in mail2user.
       D. year: Year of publication = CURRENT_YEAR + 1 for current submissions. 
       
4. pid2papers: Dict storing:
        B. All current submissions (taken from subid2papers)
        A. All papers in dblp files such that at least one author of the paper is in dbid2user.
             Exctracted by using callback function: handle_relevant_article, defined below.
        
5. pair2conflict: dict mapping dblp id pairs to an object of class 'Conflicts'. 
                 It is populated by running each rule seperately.        
                 key: frozenset(id1,id2). 
                 value: object of Class Conflicts, containing:
                     A. rules: set of rule ids that fired for this conflict
                     B. other rule specific attributes that may be required for verifying the rule, e.g. #of common papers in case of rule 1.2
                
6. exceptions: dict mapping each exception id to a set of reviewer mail ids for which the exception is raised.

#### Logic for each Rule.
1. Each rule is coded in a different function in coi_module.COI class
2. Limits, if required for any rule, are defined in CONSTANTS.py
3. Rule ids are also in CONSTANTS.py
4. Whenever comparing with any limit, >= or <= is used instead of strict inequalities < or >.

##### Rule 1
 - 1.1,  1.2:
    - Iterate over all papers using 'pid2papers'. 
    - Ignore 'Paper' if paper.year or paper.num_authors is beyond the limits. 
    - Create a conflict b/w all pairs in paper.authors

 - 1.3, 1.4, 1.5: 
    - Additional data structure required: senior2juniors: dict mapping a potential supervisor to list of its students
    - Iterate over all unique users in key_mail2user. 
    - For each user, iterate over user.coauthors. 
        - Ignore coauthor if not in dbid2user. 
        - Ignore coauthor if same as the author: CHECK why is this Happening.
        - Find intersection of user.paper_ids and coauthor.paper_ids. num_common = len(intersection).
        - (1.3) Add a conflict if intersection >= RULE_1_3_LIMIT. Add num_common attribute to Conflicts object.
        - Check if user is senior to coauthor (user.is_senior(coauthor). Logic for seniority: <br>
            (user.paper_list - coauthor.paper_list >= SENIORITY_PAPER_DIF and <br>
            (coauthor first paper year - user first paper year >= SENIORITY_YEAR_DIF <br>
            (remember that user.paper_list is sorted by year) <br>
        - If senior, then check if coauthor has many early year papers with the senior. Logic: <br>
            Iterate over first 10 (RULE_1_4_LIMIT_RELAX_TOPK) papers of the junior (junior.paper_list) <br>
            Check presence of paper.id in senior.paper_ids and increment number of common papers if present <br> 
            has_early_papers if more than 30% papers amongst first 10 are common (topk_relax_common) OR
            has_early_papers if first 3 are common (topk_common)
        - (1.4) If yes, then add to Conflicts along with the following attributes: 
            topk_common
            topk_relax_common
            topk_relax_total (should be 10 if junior has 10 papers, otherwise = number of papers of the junior
        - If yes, then add to senior2juniors dict as well. Map from user.email_id to set of all junior.email_id 
        
 - (1.5) Iterate over all the seniors found in senior2juniors.
    - For all the pairs of juniors for a given senior, add a Conflict, along with 'common supervisor' as an attribute. 

##### Rule 2

##### Rule 3

##### Rule 5


