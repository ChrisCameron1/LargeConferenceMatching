# keyword-scorer
Creates the keyword-scores based on primary and secondary subject areas

To execute, can run the following commands:
- In helper_data folder, place a list of secondary areas for each track (as in the example txt files provided)
- run below command
```
cd keyword-scorer
python keyword-scorer.py
```

The .yml must contain filepaths to data files as so for each track:

```
papers_path : "../exports/Papers.xlsx"
paper_sheet : 2 ##TODO: ADD A SANITY CHECK TO MAKE SURE IT IS TRACK 1
reviewers_path : "../exports/Reviewers-1.txt"
meta_reviewers_path : "../exports/MetaReviewers-1.txt"
subjects_track : "./helper_data/SecondaryAreas_track1_.txt" #Have MAX_SUBJECT_LENGTH
```


# Code Structure & Important Files
- `A4_helper.py`: data loading and processing functions
- `keyword-scorer.py`: main script to run the scorer
- `scores_helper.py`: helper functions for scores calculation, contains the main algorithm
## Exports from CMT (in exports folder):

Download the following files from CMT and copy them to the `exports` folder:
1. `Papers.xls` (Submissions; Actions->Export to Excel->Submissions): This file contains the paper-id, abstract, number-of-files-submitted (to check if the full paper was submitted) and submission-status (to ignore withdrawn papers). 
2. `User Information.xls` (Users->Conference Users; Actions->Export->User Information): This file contains semantic scholar ids of all the PCs, SPCs and ACs.
3. `Reviewers-1.txt`,`Reviewers-2.txt`,`Reviewers-3.txt` (Users->Reviewer; Select Track (AAAI2021, AISI); for-each-track: Actions->Export->Reviwers): These files contain the list of PCs (in each track).
4. `MetaReviewers-1.txt`, `MetaReviewers-2.txt`, `MetaReviewers-3.txt` (Users->Meta-Reviewer; Select Track (AAAI2021, AISI); for-each-track: Actions->Export->Meta-Reviwers): These files contain the list of SPCs (in each track).
5. `MetaReviewerSubjectAreas-1.txt`, `MetaReviewerSubjectAreas-2.txt`, `MetaReviewerSubjectAreas-3.txt` : These files contain the list of ACs (in each track).

`Papers.xls` and `User Information.xls` have to be opened in MS Excel Application and saved back with the same name. As CMT exports the `xls` file in a `xml` format, pandas package won't be able to read the file unless its saved in the right format.

# Output Files

`score_file` param to be set in [./config.yaml].

- score_file : columns are as follows:

    - paper ID
    - reviewer ID
    - keword score
    - overlap flag, true when there is an overlap of subject areas between the reviewer and the paper

# Things to Note when using for a different conference
- line 40 (A4_helper.py), definition of `rule2`, line 55 (A4_helper.py), definition of `inter_l1_mapping` : subject area specific
- line 285, `REVIEWER_VECTOR_get_other_foundation_weights` function in `A4_helper.py` : subject area specific filtering

