# Profile Extraction
Extracts attributes for a user profile.

To execute, do the following:
- place the input data in ../data/input/ and the output data will be placed in ../data/output/
- check the input data paths in config.yaml
- change the seniority level of reviewer in line 112 of profile_extractor.py
- The code assumes input for two tracks - (variables for second track have '_t2' suffix)
- Steps for dblp file:
    - Download the latest dblp file (dblp.xml.gz) from [here](https://dblp.org/xml/)
    - Place it in the same folder as the profile extraction script
- run  below to generate output
```
cd profile_extraction
python profile_extractor.py
```

The .yaml must contain filepaths to all data files as so:

```
papers_path : "../data/input/Papers.xls"
paper_sheet : 1
info_path : "../data/input/User Information.xls"
users_path : "../data/input/Users.txt"
score_file : "../data/output/subject_scores.txt"
output_info_file : "../data/output/new_info.xls"
logname : "./dblp.log"
...
```

The `.csv` files should have the headers and data format similar to the one provided in scoring module. 
## Exports from CMT (in exports folder):

Download the following files from CMT and copy them to the `exports` folder:
1. `Papers.xls` (Submissions; Actions->Export to Excel->Submissions): This file contains the paper-id, abstract, number-of-files-submitted (to check if the full paper was submitted) and submission-status (to ignore withdrawn papers). 
2. `User Information.xls` (Users->Conference Users; Actions->Export->User Information): This file contains semantic scholar ids of all the PCs, SPCs and ACs.
3. `Reviewers-1.txt`,`Reviewers-2.txt` (Users->Reviewer; Select Track (AAAI2021, AISI); for-each-track: Actions->Export->Reviwers): These files contain the list of PCs (in each track).
4. `MetaReviewers-1.txt`, `MetaReviewers-2.txt` (Users->Meta-Reviewer; Select Track (AAAI2021, AISI); for-each-track: Actions->Export->Meta-Reviwers): These files contain the list of SPCs (in each track).
5. `MetaReviewerSubjectAreas-1.txt`, `MetaReviewerSubjectAreas-2.txt`: : These files contain the list of ACs (in each track).

`Papers.xls` and `User Information.xls` have to be opened in MS Excel Application and saved back with the same name. As CMT exports the `xls` file in a `xml` format, pandas package won't be able to read the file unless its saved in the right format.

This code also requires `Users.txt` file to be present in the `input` folder. That file is obtained from CMT and should have the following format:
- First Name
- Middle Initial (optional)	
- Last Name	
- E-mail	
- Organization	
- Country	
- Google Scholar URL	
- Semantic Scholar URL	
- DBLP URL	
- Domain Conflicts


# Code Structure & Important Files
- `A4_dblp_helper.py`: helper functions for processing dblp file
- `A4_helper.py`: helper functions for the profile extractor
- `profile_extractor.py`: main file for getting profile information

# Output Files

Output file is saved to `output_info_file` in the config.yaml file.

When the script executes, you will have the output with following columns:
- `Email`: email id of the user
- `Country`: Country is listed in the User.txt export from CMT
- `Region` : takes one of the following values:  Asia, Australia, Europe, America, and Unknown
- `Popular`: A reviewer is reviewing a popular area if percent of papers submitted in the L1 of the reviewer primary area > 8% of total papers in the conference
- `S1` = 1, if reviewer j has reviewed 3+ times or has written >10 papers in sister conferences
- `S2` = 1, if reviewer j has reviewed 3+ times or has written >4 papers in sister conferences
- `S3` = 1, if reviewer j has reviewed 1+ times or has written >2 papers in sister conferences

# Things to Note
- This scipt will create a pkl cache for the dblp file. In order to use a new dblp file, please delete the pkl file to refresh the cache.
- See the sample output file 'new_info.xlsx' for reference
- if country is not found for a user, then it is populated with empty string ''