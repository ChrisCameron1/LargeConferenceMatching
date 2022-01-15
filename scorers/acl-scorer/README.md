## Overview

This code is based on the [ACL Reviewer Matching Code](https://github.com/acl-org/reviewer-paper-matching). The score is based on the textual similarity between the abstract of a submitted paper and the abstracts of a reviewer’s prior publications.

## Steps to Run
1. Download a trained model and a large corpus of abstract (to profile reviewers) from [here](https://drive.google.com/file/d/18yEg1CCJypeNxRv6orgpkJGOD5DLaP2c/view?usp=sharing) and extract the contents into the `scratch` folder. 
2. Download the necessary files from CMT to the `exports` folder. Some sample files are in `exports` folder.
3. Compute the scores for each (role, track) pair by setting the appropriate CMT export as the argument to `--reviewer_file`. Roles can be `PC`, `SPC` or `AC`. Tracks are the various tracks in the conference (like main-track, AISI track). To generate scores for PCs in Track-1, run the following command:

    ```
    python compute_scores_aaai.py --submission_file ../exports/Papers.xls --db_file scratch/aaai-s2-entries.json --reviewer_file ../exports/Reviewers-1.txt --user_info_file "../exports/User Information.xls" --model_file scratch/similarity-model.pt --abstracts_file "scratch/additional-abstracts.tsv"
    ```

4. Agggregate all score files to a single file (`.\output\acl_scores.csv`) by running `python merge_all_files.py`.

## Scratch Folder

The `scratch` folder should contain a large corpus of abstracts relevant to the conference filtered from semantic scholar (`--db_file`), a trained similarity model (`--model_file`) and other related files such as the vocabulary file used by the model, and a file with additional abstracts (`--abstracts_file`).

### Generate a corpus of abstracts relevant to the conference

These commands were used for AAAI. Please add/remove conference as you deem fit.

PLEASE NOTE: The corpus downloded along with the model file only contains papers till 2020. Please run the following script to include more recent abstracts for better reviewer profiling.

1. Download a dump of sematic scholar and filter the relevant entries (high recall, low precision) using the following script:
```
nohup zcat s2/s2-corpus*.gz | grep 'AAAI\|AAMAS\|ACL\|AISTATS\|COLT\|CoRL\|CPAIOR\|CVPR\|ECAI\|ECML\|EMNLP\|HCOMP\|ICAPS\|ICCV\|ICDM\|ICLR\|ICML\|ICRA\|ICWSM\|IJCAI\|International Conference on Human-Robot Interaction\|IROS\|IUI\|KDD\|NAACL\|NeurIPS\|NIPS\|Robotics: Science and Systems\|SIGIR\|WWW\|WSDM\|UAI\|AIES\|ACM Multimedia\|AIED\|CAV\|CHI\|CIKM\|CogSci\|COLING\|CSCW\|Ethics and Information Technology\|FATML\|FAT\*\|FOGA\|GECCO\|ICASSP\|ICDE\|IJCAR\|ISMAR\|ISWC\|J\. Mach\. Learn\. Res\.\|MICCAI\|Minds and Machines\|PODS\|SIGMOD\|TACL\|Transactions of the Association for Computational Linguistics\|UbiComp\|UIST\|VLDB\|CoNLL\|CV\|EACL\|ECCV\|ECIR\|IJCNN\|INTERSPEECH\|K-CAP\|PAKDD\|SDM\|WACV\|WISE\|\"venue\":\"CP\|\"venue\":\"RSS\|\"venue\":\"Machine Learning\"\|\"venue\":\"MM\|\"venue\":\"SAT\|\"venue\":\"EC\|\"venue\":\"KR\|Transactions of the Association for Computational Linguistics' > aaai-s2-entries-noisy.json &
```
2. Run the following script to remove false positives:
```
python filter_entries.py --infile scratch/aaai-s2-entries-noisy.json --outfile scratch/aaai-s2-entries-cleaned.json
```

### Training a Similarity Model

Please follow the instructions in [Part 1: Installing and Training Model (before review process)](https://github.com/acl-org/reviewer-paper-matching#part-1-installing-and-training-model-before-review-process) to train a model from scratch. In step 3, please use the corpus of abstracts relevant to your conference.

### Additional Abstracts:

- Many reviewers did not have a semantic scholar id. We requested such reviewers to submit URLs of their published papers in CMT.  The `abstracts_file` contains abstracts extracted from those paper URLs.
- Many reviewers did not have any papers in the filtered semantic scholar db (`aaai-s2-entries.json`). We identified such reviewers and for only these reviewers extracted papers published by relaxing the year constraint in `filter_entries.py ` to `year>2000`. These abstracts were also appended to the `abstracts_file`.

## Exports from CMT:

Download the following files from CMT and copy them to the `exports` folder:
1. `Papers.xls` (Submissions; Actions->Export to Excel->Submissions): This file contains the paper-id, abstract, number-of-files-submitted (to check if the full paper was submitted) and submission-status (to ignore withdrawn papers). 
2. `User Information.xls` (Users->Conference Users; Actions->Export->User Information): This file contains semantic scholar ids of all the PCs, SPCs and ACs.
3. `Reviewers-1.txt`,`Reviewers-2.txt` (Users->Reviewer; Select Track (AAAI2021, AISI); for-each-track: Actions->Export->Reviwers): These files contain the list of PCs (in each track).
4. `MetaReviewers-1.txt`, `MetaReviewers-2.txt` (Users->Meta-Reviewer; Select Track (AAAI2021, AISI); for-each-track: Actions->Export->Meta-Reviwers): These files contain the list of SPCs (in each track).
5. `MetaReviewerSubjectAreas-1.txt`, `MetaReviewerSubjectAreas-2.txt`: : These files contain the list of ACs (in each track).

`Papers.xls` and `User Information.xls` have to be opened in MS Excel Application and saved back with the same name. As CMT exports the `xls` file in a `xml` format, pandas package won't be able to read the file unless its saved in the right format.