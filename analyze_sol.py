import sys
import os
import pickle
import pandas as pd
import argparse
import numpy as np
import xml.etree.ElementTree as et
from tqdm import tqdm
from collections import defaultdict
import yaml
import re
import itertools
from dataclasses import dataclass
from matching_data import get_data


@dataclass
class ParsedSolution:
    df: pd.DataFrame
    region_df: pd.DataFrame
    cycles: list
    region_stats: bool
    cycle_stats: bool
    full_cycles: list

def get_violation_records(distance_df, solution_df):
    
    d0_constraint_violations = 0
    d1_constraint_violations = 0
    violation_records = []

    for paper, group in solution_df.query('role != "AC"').groupby('paper'):
        try:
            rids = group['reviewer'].unique()
        except KeyError:
            rids = group['rid'].unique()
        for (a,b) in itertools.combinations(rids, 2):
            (a,b) = sorted((a,b))
            try:
                distance = distance_df.loc[(a,b)]['distance'].item()
            except KeyError:
                continue

            violation_records.append(
                    dict(reviewer_1=a, reviewer_2=b, paper=paper, d=distance)
                )

    df = pd.DataFrame.from_records(violation_records,columns=['reviewer_1', 'reviewer_2', 'paper', 'd'])
    print(f"Found {df.query('d == 0')} violations of co-author distance 0 and {df.query('d == 1')} of co-author distance 1")
    return df

        
def parse_solution(solution_file: str, paper_reviewer_df: pd.DataFrame, reviewer_df: pd.DataFrame) -> ParsedSolution:
    tree = et.parse(solution_file)

    region_stats = False
    regions = []
    cycle_stats = False
    cycles = []
    full_cycles = []
    records = []

    print(f"Parsing solution")
    root = tree.getroot()
    for variables in root.iter('variables'):
        for variable in variables.iter('variable'):
            attrib = variable.attrib
            name, value = attrib['name'], attrib['value']
            # x_ij means reviewer j reviews paper i
            # Small value in case cplex returns some stupid numbers close to 0
            if name.startswith('x') and float(value) > 1e-5:
                paper, reviewer = (int(x) for x in name.replace('x', '').split('_'))
                score = paper_reviewer_df.loc[(paper, reviewer)]['score'].item()
                r_info = reviewer_df.query(f'reviewer == {reviewer}')
                record = dict(
                    paper = paper,
                    reviewer = reviewer,
                    score = paper_reviewer_df.loc[(paper,reviewer)]['score'].item(),
                    role = r_info['role'].item(),
                    seniority = r_info['seniority'].item(),
                )
                records.append(record)
            elif re.match(r'^re\d+$', name):
                region_stats = True
                if float(value) > 1e-5:
                    paper = int(name.replace('re', ''))
                    regions.append(dict(
                        paper=paper,
                        regions=float(value)
                    ))
            elif name.startswith('cy'):
                cycle_stats = True
                if float(value) > 1e-5:
                    paper1, paper2 = name.replace('cy', '').split('_')
                    cycles.append((int(paper1), int(paper2)))

    for constraint in root.iter('constraint'):
        n = constraint.attrib['name']
        search = re.search(r'^cyc_ip(\d+)_jp(\d+)_i(\d+)_j(\d+)$', n)
        if search:
            pid1, rid1, pid2, rid2 = search.group(1), search.group(2), search.group(3), search.group(4)
            full_cycles.append(tuple(map(int, (pid1, rid1, pid2, rid2))))


    return ParsedSolution(
        df = pd.DataFrame.from_records(records),
        region_stats = region_stats,
        region_df = pd.DataFrame.from_records(regions),
        cycle_stats = cycle_stats,
        cycles = cycles,
        full_cycles = full_cycles,
    )

def vcs_normalized(s, sort_index=True):
    vc = s.value_counts().to_frame()
    vc['%'] = s.value_counts(normalize=True)
    return vc.sort_index()


def print_dict(d):
    for k, v in d.items():
        print(f"{k}\t{v}")


POSITIVE_BID = 2

def count_positive_bids_for_rid(expanded_bid_df, rid, thresh=POSITIVE_BID):
    bids = expanded_bid_df.query(f'reviewer == {rid}')['bid']
    positive_bids = bids[bids >= thresh]
    return len(positive_bids)

def distribution_of_positive_bids(expanded_bid_df, frame):
    UPPER_CLIP = 10
    vcs_dict = frame.reset_index()['rid'].apply(lambda x: count_positive_bids_for_rid(expanded_bid_df, x)).clip(upper=UPPER_CLIP).value_counts().sort_index().to_dict()
    vcs_dict[f'{UPPER_CLIP}+'] = vcs_dict[UPPER_CLIP]
    del vcs_dict[UPPER_CLIP]
    return vcs_dict

def parse_unassigned_papers(parsed_solution, per_paper_num_indicators, k=None, filename=None):

    print('Updating num indicators for unassigned papers...')

    df = parsed_solution.df

    under_capacity_pc = (2 - df.query('role == "PC"').groupby('paper').size()).to_dict()
    under_capacity_spc = (1 - df.query('role == "SPC"').groupby('paper').size()).to_dict()
    under_capacity_ac = (1 - df.query('role == "AC"').groupby('paper').size()).to_dict()

    papers = df.reset_index()['paper'].unique()
    pc_papers = set(df.query('role == "PC"')['paper'].unique())
    spc_papers = set(df.query('role == "SPC"')['paper'].unique())
    ac_papers = set(df.query('role == "AC"')['paper'].unique())
    for paper in papers:
        if paper not in pc_papers:
            per_paper_num_indicators.loc[paper,'pc_k'] += k
        if paper not in spc_papers:
            per_paper_num_indicators.loc[paper,'spc_k'] += int(k/2)
        if paper not in ac_papers:
            per_paper_num_indicators.loc[paper,'ac_k'] += int(k/2)


    for paper in under_capacity_pc:
        if under_capacity_pc[paper] > 0:
            per_paper_num_indicators.loc[paper,'pc_k'] += k
    for paper in under_capacity_spc:
        if under_capacity_spc[paper] > 0:
            per_paper_num_indicators.loc[paper,'spc_k'] += int(k/2)
    for paper in under_capacity_ac:
        if under_capacity_ac[paper] > 0:
            per_paper_num_indicators.loc[paper,'ac_k'] += int(k/2)
    
    per_paper_num_indicators.to_csv(filename, index=False)

    return per_paper_num_indicators
    
# TODO: Write output to results_file
def analyse_solution(config, solution_file: str , results_file: str):

    print(f"Reading solution file {solution_file}")

    matching_data = get_data(config=config)
    paper_reviewer_df = matching_data.paper_reviewer_df
    reviewer_df = matching_data.reviewer_df

    #%%
    parsed_solution = parse_solution(solution_file=solution_file, paper_reviewer_df=paper_reviewer_df, reviewer_df=reviewer_df)
    df = parsed_solution.df

    #%%
    n_papers = paper_reviewer_df.reset_index()['paper'].nunique()
    n_reviewers = paper_reviewer_df.reset_index()['reviewer'].nunique()
    print(f"Aggscore matrix is shaped for {n_papers} papers and {n_reviewers} reviewers (all roles). This may not reflect the final counts.")
    print()

    print("Reviewer info contains the following role distribution")
    print(reviewer_df['role'].value_counts())
    print()

    print(f"Assigned {df['reviewer'].nunique()} reviewers (all roles) to {df['paper'].nunique()} papers. Some reviewers may have gone unused - check below")
    print()

    unassigned_reviewers = set(reviewer_df.index.values) - set(df['reviewer'].unique())
    unassigned_reviewers_df = reviewer_df.loc[unassigned_reviewers]
    print("Reviewers by role assigned 0 papers")
    print(vcs_normalized(unassigned_reviewers_df['role']))
    print()


    print("Seniority distribution of eligible PCs")
    print(vcs_normalized(reviewer_df.query('role == "PC"')['seniority']))
    print()

    print("Seniority distribution of matched PCs")
    print(vcs_normalized(df.query('role == "PC"')['seniority']))
    print()

    print("Seniority of unassigned PCs")
    print(vcs_normalized(unassigned_reviewers_df.query('role == "PC"')['seniority']))
    print()

    print("Mean, median, min and max match score of each reviewer-paper assignment")
    print(df['score'].describe())
    print()


    for role, group in df.groupby('role'):
        print(f"Mean, median, min and max match score of each reviewer-paper assignment by role: {role}")    
        print(group['score'].describe())
        print()

    #%%
    print("Mean, median, min and max of mean match-score for each paper (PC-ONLY).")
    print(df.query('role == "PC"').groupby('paper')['score'].mean().describe())
    print()

    print("Mean, median, min and max of min match-score for each paper (PC-ONLY)")
    print(df.query('role == "PC"').groupby('paper')['score'].min().describe())
    print()

    print("Mean, median, min and max of mean match-score for each paper.")
    print(df.groupby('paper')['score'].mean().describe())
    print()

    print("Mean, median, min and max of min match-score for each paper.")
    print(df.groupby('paper')['score'].min().describe())
    print()

    for role in df['role'].unique():
        role_df = df.query(f'role == "{role}"')
        for thresh in [0.5, 0.3, 0.15, 0.000001]:
            min_scores = role_df.groupby('paper')['score'].min()
            val = (min_scores <= thresh).sum()
            print(f"Number of papers with a {role} with score <= {thresh}: {val}")    
    print()


    #%%
    print("Minimum seniority of papers distribution (PC)")
    min_pc_seniority = df.query('role == "PC"').groupby('paper')['seniority'].min()
    print(min_pc_seniority.value_counts().sort_index())
    print()

    print("Minimum seniority of papers distribution (PC + SPC)")
    min_seniority = df.groupby('paper')['seniority'].min()
    print(vcs_normalized(min_seniority))
    print()

    print(f"Papers with a min PC seniority of 3 or higher: {', '.join(map(str, min_pc_seniority[min_pc_seniority >= 3].index.values))}")
    print()

    #%%
    for role in reviewer_df['role'].unique():
        vc = df.query(f'role == "{role}"').groupby('reviewer').size().value_counts().sort_index()
        vcd = vc.to_dict()
        vcd[0] = reviewer_info.loc[unassigned_reviewers]['role'].value_counts().sort_index().to_dict().get(role, 0)
        print(f"Distribution of number of papers assigned to {role} reviewers")
        print_dict(vcd)
        print()


    def review_distribution(frame, role):
        print(f"Distribution of number of reviews per paper by role {role}")
        vcs = frame.groupby('paper').size().value_counts().sort_index().to_dict()
        vcs[0] = len(set(paper_reviewer_df.reset_index()['paper'].unique().values) - set(frame['paper'].unique()))
        print_dict(vcs)
        expected_reviews = n_papers * (config['HYPER_PARAMS'][f'max_papers_per_reviewer_{role}'])
        print(f"This role was assigned {len(frame)} reviews. Required: {expected_reviews}. Missing {expected_reviews - len(frame)}")
        print()

    for role in reviewer_df['role'].unique().tolist():
        review_distribution(df.query(f'role == "{role}"'), role) 


    violation_records_df = get_violation_records(info, df)
    couathor_violation_file = args.solution_file.replace('.sol', '_coauthor_violations.csv')
    print(f'Writing exact coauthor violations to {couathor_violation_file}')
    violation_records_df.to_csv(couathor_violation_file, index=False)

    if parsed_solution.region_stats:
        # Check if 'regions' columns exist in dataframe
        if 'regions' in parsed_solution.region_df.columns:
            # NOTE: Region constraints only get imposed on "popular" areas
            print("Distribution of unique region reviewers (all roles) for each popular area paper")
            print(parsed_solution.region_df['regions'].value_counts().sort_index())

            bad_region_pids = parsed_solution.region_df.query('regions <= 2')['paper'].values
            print(f"Papers with 2 or fewer regions: {', '.join(map(str, bad_region_pids))}")
            print()

    if parsed_solution.cycle_stats:
        print(f"{len(parsed_solution.cycles)} papers violate cyclic review constraint")
        print(f"Reviewer cycles: {parsed_solution.cycles}")
        print()
        for cycle in parsed_solution.full_cycles:
            pid1, rid1, pid2, rid2 = cycle
            if len(df.query(f'reviewer == {rid1} and paper == {pid1}')) > 0 and len(df.query(f'reviewer == {rid2} and paper == {pid2}')) > 0:
                print(f"VIOLATED CYCLE {cycle}")


    print("Hyper parameters used:")
    print_dict(config['HYPER_PARAMS'])

    # Write out final solution to file to send back
    final_solution_prefix = args.solution_file.replace('.sol', '_matching')
    paper_df.to_csv('%s.csv' % final_solution_prefix, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution_file", help="solution file to read in CPLEX format", default='/global/scratch/aaai_mip_data/instances/all_k=100.sol')
    parser.add_argument("--config_file", help="config that generated the solution", default=None)
    main(parser.parse_args())
    

