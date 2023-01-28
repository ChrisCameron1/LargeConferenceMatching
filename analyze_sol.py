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
import logging
import logging.handlers

logger = logging.getLogger(__name__)


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

    df = pd.DataFrame.from_records(violation_records,columns=['reviewer_1', 'reviewer_2', 'paper', 'd']).drop_duplicates()
    logger.info(f"Found {len(df.query('d == 0').index)} violations of co-author distance 0 and {len(df.query('d == 1').index)} of co-author distance 1")
    return df

        
def parse_solution(solution_file: str, paper_reviewer_df: pd.DataFrame, reviewer_df: pd.DataFrame) -> ParsedSolution:
    
    tree = et.parse(solution_file)

    region_stats = False
    regions = []
    cycle_stats = False
    cycles = []
    full_cycles = []
    records = []

    logger.info(f"Parsing solution")
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
                    score = score,
                    role = r_info['role'].item(),
                    seniority = r_info['seniority'].item(),
                    region = r_info['region'].item()
                )
                records.append(record)
            elif re.match(r'^region\d+$', name):
                region_stats = True
                if float(value) > 1e-5:
                    paper = int(name.replace('region', ''))
                    regions.append(dict(
                        paper=paper,
                        regions=int(np.round(float(value)))
                    ))
            elif name.startswith('cycle'):
                cycle_stats = True
                if float(value) > 1e-5:
                    paper1, paper2 = name.replace('cycle', '').split('_')
                    cycles.append((int(paper1), int(paper2)))

    for constraint in root.iter('constraint'):
        n = constraint.attrib['name']
        search = re.search(r'^cycle_ip(\d+)_jp(\d+)_i(\d+)_j(\d+)$', n)
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
        logger.info(f"{k}\t{v}")


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

    logger.info('Updating num indicators for unassigned papers...')

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
    
    per_paper_num_indicators.to_csv(filename, index=True)

    return per_paper_num_indicators
    
def analyse_solution(config, solution_file: str, matching_data=None):

    results_file = solution_file.replace('.sol', '_RESULTS.txt')

    handler = logging.FileHandler(filename=results_file)
    logger.addHandler(handler)

    logger.info(f"Reading solution file {solution_file}")

    if matching_data is None:
        matching_data = get_data(config=config)

    paper_reviewer_df = matching_data.paper_reviewer_df
    reviewer_df = matching_data.reviewer_df
    distance_df = matching_data.distance_df
    #%%
    parsed_solution = parse_solution(solution_file=solution_file, paper_reviewer_df=paper_reviewer_df, reviewer_df=reviewer_df)
    df = parsed_solution.df

    #%%
    n_papers = paper_reviewer_df.reset_index()['paper'].nunique()
    n_reviewers = paper_reviewer_df.reset_index()['reviewer'].nunique()
    logger.info(f"Aggscore matrix is shaped for {n_papers} papers and {n_reviewers} reviewers (all roles). This may not reflect the final counts.")
    logger.info('')

    logger.info("Reviewer info contains the following role distribution")
    logger.info(reviewer_df['role'].value_counts())
    logger.info('')

    logger.info(f"Assigned {df['reviewer'].nunique()} reviewers (all roles) to {df['paper'].nunique()} papers. Some reviewers may have gone unused - check below")
    logger.info('')

    unassigned_reviewers = set(reviewer_df.index.values) - set(df['reviewer'].unique())
    unassigned_reviewers_df = reviewer_df.loc[unassigned_reviewers]
    logger.info("Reviewers by role assigned 0 papers")
    logger.info(vcs_normalized(unassigned_reviewers_df['role']))
    logger.info('')


    logger.info("Seniority distribution of eligible PCs")
    logger.info(vcs_normalized(reviewer_df.query('role == "PC"')['seniority']))
    logger.info('')

    logger.info("Seniority distribution of matched PCs")
    logger.info(vcs_normalized(df.query('role == "PC"')['seniority']))
    logger.info('')

    logger.info("Seniority of unassigned PCs")
    logger.info(vcs_normalized(unassigned_reviewers_df.query('role == "PC"')['seniority']))
    logger.info('')

    logger.info("Mean, median, min and max match score of each reviewer-paper assignment")
    logger.info(df['score'].describe())
    logger.info('')


    for role, group in df.groupby('role'):
        logger.info(f"Mean, median, min and max match score of each reviewer-paper assignment by role: {role}")    
        logger.info(group['score'].describe())
        logger.info('')

    #%%
    logger.info("Mean, median, min and max of mean match-score for each paper (PC-ONLY).")
    logger.info(df.query('role == "PC"').groupby('paper')['score'].mean().describe())
    logger.info('')

    logger.info("Mean, median, min and max of min match-score for each paper (PC-ONLY)")
    logger.info(df.query('role == "PC"').groupby('paper')['score'].min().describe())
    logger.info('')

    logger.info("Mean, median, min and max of mean match-score for each paper.")
    logger.info(df.groupby('paper')['score'].mean().describe())
    logger.info('')

    logger.info("Mean, median, min and max of min match-score for each paper.")
    logger.info(df.groupby('paper')['score'].min().describe())
    logger.info('')

    for role in df['role'].unique():
        role_df = df.query(f'role == "{role}"')
        for thresh in [0.5, 0.3, 0.15, 0.000001]:
            min_scores = role_df.groupby('paper')['score'].min()
            val = (min_scores <= thresh).sum()
            logger.info(f"Number of papers with a {role} with score <= {thresh}: {val}")    
    logger.info('')


    #%%
    logger.info("Maximum seniority of papers distribution (PC)")
    max_pc_seniority = df.query('role == "PC"').groupby('paper')['seniority'].max()
    logger.info(max_pc_seniority.value_counts().sort_index())
    logger.info('')

    logger.info("Maximum seniority of papers distribution (PC + SPC)")
    max_seniority = df.groupby('paper')['seniority'].max()
    logger.info(vcs_normalized(max_seniority))
    logger.info('')

    logger.info(f"Papers with a max PC seniority of 1 or lower: {', '.join(map(str, max_pc_seniority[max_pc_seniority <= 1].index.values))}")
    logger.info('')

    #%%
    for role in reviewer_df['role'].unique():
        vc = df.query(f'role == "{role}"').groupby('reviewer').size().value_counts().sort_index()
        vcd = vc.to_dict()
        vcd[0] = reviewer_df.loc[unassigned_reviewers]['role'].value_counts().sort_index().to_dict().get(role, 0)
        logger.info(f"Distribution of number of papers assigned to {role} reviewers")
        print_dict(vcd)
        logger.info('')


    def review_distribution(frame, role):
        logger.info(f"Distribution of number of reviews per paper by role {role}")
        vcs = frame.groupby('paper').size().value_counts().sort_index().to_dict()
        vcs[0] = len(set(paper_reviewer_df.reset_index()['paper'].unique()) - set(frame['paper'].unique()))
        print_dict(vcs)
        expected_reviews = n_papers * (config['HYPER_PARAMS'][f'max_reviews_per_paper_{role}'])
        logger.info(f"This role was assigned {len(frame)} reviews. Required: {expected_reviews}. Missing {expected_reviews - len(frame)}")
        logger.info('')

    for role in reviewer_df['role'].unique().tolist():
        review_distribution(df.query(f'role == "{role}"'), role) 


    violation_records_df = get_violation_records(distance_df, df)
    couathor_violation_file = solution_file.replace('.sol', '_coauthor_violations.csv')
    logger.info(f'Writing exact coauthor violations to {couathor_violation_file}')
    violation_records_df.to_csv(couathor_violation_file, index=False)

    if parsed_solution.region_stats:
        # Check if 'regions' columns exist in dataframe
        if 'regions' in parsed_solution.region_df.columns:
            # NOTE: Region constraints only get imposed on "popular" areas
            logger.info("Distribution of unique regions represented across popular-area papers")
            logger.info(parsed_solution.region_df['regions'].value_counts().sort_index())

            bad_region_pids = parsed_solution.region_df.query('regions <= 1')['paper'].values
            logger.info(f"Papers with 1 or fewer regions: {', '.join(map(str, bad_region_pids))}")
            logger.info('')

    if parsed_solution.cycle_stats:
        logger.info(f"{len(parsed_solution.cycles)} papers violate cyclic review constraint")
        logger.info(f"Reviewer cycles: {parsed_solution.cycles}")
        logger.info('')
        for cycle in parsed_solution.full_cycles:
            pid1, rid1, pid2, rid2 = cycle
            if len(df.query(f'reviewer == {rid1} and paper == {pid1}')) > 0 and len(df.query(f'reviewer == {rid2} and paper == {pid2}')) > 0:
                logger.info(f"VIOLATED CYCLE {cycle}")


    logger.info("Hyper parameters used:")
    print_dict(config['HYPER_PARAMS'])

    # Write out final solution to file to send back
    final_solution_prefix = solution_file.replace('.sol', '_matching')
    df.to_csv('%s.csv' % final_solution_prefix, index=False)

    logger.removeHandler(handler)

    return parsed_solution, violation_records_df

    

