# LargeConferenceMatching
This is code for the paper "Matching papers and reviewers at large conferences" (see [here](https://arxiv.org/abs/2202.12273))


Testing with python version `3.7.3`, pandas version `1.3.2`

# Quick Start

To test that the code runs on your system, you first need to create some toy data. Run:
```
cd LargeConferenceMatching
mkdir results
python data_generator.py
```
This creates the necessary data files in `./toy_data`. Then to run the matching on that data, execute:
```
python iter_solve.py --config_file toy_config.yml --add_soft_constraints --output_files_prefix ./results/toy
```
The file should terminate with the message:
```
No more coauthor constraints to add. Terminated at iteration [iter] with 0 d0 and 1 d1 coauthor constraints
```
The output matching file will be created in `./results/toy_[iter]_matching.csv`, where `[iter]` is the iteration of row generation. See the "Output Files" section for a description of each output file. Note that this toy data is very unrealistic and many soft constraints are never violated. You will not be able to fully test row generation without more realistic data.

# Running on your data

To run on your own data you must create the following files that the `config.yml` file points to:
```
RAW_SCORES_FILE: 'data/scores.csv'
BIDS_FILE: 'data/bids.csv'
REVIEWERS_FILE: 'data/reviewers.csv'
COAUTHOR_DISTANCE_FILE: 'data/distances.csv'
```

The `.csv` files should have the following headers and data format:
- `RAW_SCORES_FILE`
**Header**: `paper,reviewer,ntpms,nacl,nk`\
**Data types**:\
`paper/reviewer`: int (unique indentifier for every paper/reviewer)\
`ntpms,nacl,nk`: float (any real number. The `n` represents the normalized version of each of tpms,acl, and k (keywords). See paper for description of normalization.)
- `BIDS_FILE`\
**Header**: `paper,reviewer,bid`\
**Data types**:\
`paper/reviewer`: int (unique indentifier for every paper/reviewer)\
`bid`: float (0.05,1,2,4,6: not willing , not entered, in a pinch, willing, eager)
- `REVIEWERS_FILE`\
**Header**: `reviewer,role,seniority,conflict_papers,region,authored`\
**Data types**:\
`reviewer`: int (unique indentifier for every paper/reviewer)\
`role`: 'AC', 'SPC', or 'PC'\
`seniority`: 0,1,2,3\
`conflict_paper`: list of papers (e.g., [0,1,2])\
`region`: string representing region (e.g., US)\
`authored`: list of papers the reviewer authored (e.g., [0,1,2])
- `COAUTHOR_DISTANCE_FILE`\
**Header**: `reviewer_1,reviewer_2,distance`\
**Data types**:\
`reviewer_1/reviewer_2`: int (unique indentifier for every reviewer)\
`distance`: 0 for direct coauthors, 1 for once removed

There is an addition file that is created after the scores are created. This is used for caching on subsequent runs. Please delete this file is you want to recompute the scores for any reason:
```
CACHED_SCORES_FILE: 'data/cached_scores.csv'
```

After creating the necessary files, run:

```
cd LargeConferenceMatching
mkdir results
python iter_solve.py --config_file config.yml --add_soft_constraints --output_files_prefix ./results/test
```


# Code Structure & Important Files
- `iter_solve.py`: This is the main file for running (will call all of the below)

- `matching_ilp.py`: Code for building and writing the MIP
- `create_indicator.py`: Wrapper around `matching_ilp.py` that sets up the variable indicator matrix. We create variables as follows:
    - For each paper, create a variable for the K best PCs, SPCs, and ACs.
    - For each PC/SPC/AC member, create a variable for the K/5K/10K best papers. 
    - We do not create variables for really weak reviewer-paper pairs (as determined by a score threshold), as if these were set they would be poor matches.

- `analyze_sol.py`: Analyzes solution outputs
- `lp_solver.py`: Code for running a MIP. Note that CPLEX warm starting is used when available and that the MIP is optimized only to within some fixed absolute tolerance (configurable within this file).

# Constraint Overview

- **Regions:** Reward a paper for number of unique regions the reviewers come from.

- **Seniority:** Every reviewer is assigned seniority score in [0-3], with 3 being the most senior level. We bound a slack variable by cumulative seniority level: slack <= cumulative seniority. We then add to objective function `sen_reward` * slack. The higher the slack, the more senior the the reviewers are on that paper. We cap the slack with `target_seniority` to encourage more balance in seniority across papers.

- **Balance:** In order to balance reviewing load across ACs and SPCs, we add a penalty for every additional paper that the AC/SPC has to review beyond some initial number of "free" papers. The per-paper penalty increases as certain paper number thresholds get crossed. See `paper_distribution_pen`.

- **Cycle Constraints:** Prevent cycles

- **Conflicts:** These constraints are handled via column generation since we could not include all of them at once. We start with 0 coreview contraints. After ILP is solved, we check which co-review constraints were violated. We then add constraints to the ILP that restrict those co-review violations. We repeated this process until we were satisfied with the number of violated constraints (the number of violations tended to plateau around eight iterations).

# Important parameters

- `sparsity_k`: You should increase `sparsity_k` as high as your computation / memory requirments allow.
- `include_d1` does
    - penalize coreviewing couathors once removed. By default, only penalizes direct couathors.
- `remove_non_symmetric` :Removes reviewer pairs for coreview constraints if their coreview distances were not symmetric (i.e., d(i,j) \neq d(j,i)))
- `fixed_variable_solution_file` File format is a CSV of the form `paper,reviewer`. These pairings will be set to occur, no matter what. (e.g, for phase 2, would pass a file for fixed phase 1 matches)
- `relax_paper_capacity` change the reviewer capacity constraints for every paper to be an upper bound rather than equality. Useful, because papers without enough reviewers must have had very poor options for matches. Manual matching is more appropriate here if there a small number of such cases.
- `score_threshold` variables don't get created for any paper-reviewer pair below the `score_threshold`


# Output Files

`--output_files_prefix` param set to be [dir/experiment_name].

When each stage completes, you will find the following files in `dir` with prefix `[output_files_prefix]`\_iter\_`[iteration]`]: 

    `.lp` - MIP file that is passed to CPLEX    
    `_status.json` - dictionary of time (walltime), status (CPLEX status), objective (CPLEX objective), and full_objective (objective after adding full constraint set)
    `.sol` - CPLEX solution file (for warm starting / analyzing)
    `_cplex.log` - CPLEX log of solcing
    `.yml` - configuration used for that iteration
    `_per_paper_num_indicators.csv` - Number of indicators created for each paper, for each reviewer role
    `_coauthor_violations.csv` - Coauthor violations in the assignment
    `_indicator.pkl`- Indicator matrix representing that (paper,reviewer) varaiables that are created(cached if you rerun)
    `_matching.csv` - Main output file containing the complete matching. Columns are `paper,reviewer,role,score,seniority`.
    `_RESULTS.txt` - Solution analysis script 

# FAQ

- *CPLEX is taking a long time solve the `.lp` file?* Try setting `--abstol 10` in `iter_solver.py`. Often CPLEX quickly find a good enough solution but takes a long time proving optimality. CPLEX terminates if it finds a solution that it can prove is within `--abstol` units of objective function away from optimality. Change `10` as needed for your desired level of optimality.

# Paper

```
@misc{https://doi.org/10.48550/arxiv.2202.12273,
  doi = {10.48550/ARXIV.2202.12273},
  url = {https://arxiv.org/abs/2202.12273},
  author = {Leyton-Brown, Kevin and {Mausam} and Nandwani, Yatin and Zarkoob, Hedayat and Cameron, Chris and Newman, Neil and Raghu, Dinesh},
  keywords = {Artificial Intelligence (cs.AI), Information Retrieval (cs.IR), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Matching Papers and Reviewers at Large Conferences},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```