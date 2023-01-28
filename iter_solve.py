import pandas as pd
import sys
import argparse
import numpy as np
import xml.etree.ElementTree as et
from tqdm import tqdm
from collections import defaultdict
import os
import yaml
import json
from analyze_sol import parse_solution, ParsedSolution, parse_unassigned_papers, analyse_solution, get_violation_records
from lp_solver import solve
from matching_ilp import to_name, MatchingILP
from coreview_filter import get_coreview_vars
from collections import defaultdict
from matching_data import get_data
import logging
import logging.config

logger = logging.getLogger(__name__)

CONFLICT_SUFFIX = '_CONFLICTS.csv'
NUM_VARIABLE_INDICATORS_SUFFIX = '_per_paper_num_indicators.csv'
NUM_REVIEWER_SUFFIX = '_reviewer_windows.csv'

def update_unused_reviewers(per_reviewer_num, reviewer_df, matching_df, k, filename=None):
    # Find unused reviewers and increment their k
    all_rids = set(reviewer_df.index.values)
    used_rids = set(matching_df['reviewer'].unique())
    unused_rids = list(all_rids - used_rids)
    per_reviewer_num.loc[unused_rids, 'window_end'] += k
    if filename is not None:
        per_reviewer_num.to_csv(filename)

def create_initial_per_reviewer_num(reviewer_df, k):
    per_reviewer_num = reviewer_df.copy().reset_index()[['reviewer', 'role',]].set_index('reviewer')
    per_reviewer_num['window_start'] = 0
    role_2_multiplier = {
        'AC': 10,
        'SPC': 5,
        'PC': 1,
    }
    per_reviewer_num['window_end'] = per_reviewer_num['role'].map(role_2_multiplier) * k  - 1
    return per_reviewer_num

def setup_logging(filename):
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "console": {"format": "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"},
                "file": {"format": "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"},
            },
            "handlers": {
                "console": {"class": "logging.StreamHandler", "formatter": "console"},
                "file": {"class": "logging.FileHandler", "formatter": "file", "filename": filename},
            },
            "loggers": {"": {"level": "INFO", "handlers": ["console", "file"]}},
        }
    )

def main(output_files_prefix='itertest', 
        config=None,
        num_coreview_vars=None,
        master_conflicts_file=None,
        rebuild_scores_file=False,
        initial_solution=None,
        fixed_variable_solution_file=None,
        abstol=None,
        relative_mip_gap=None,
        valid_papers_file=None,
        valid_reviewers_file=None,
        add_soft_constraints=True,
        no_iterate=False,
        max_iter=1000000000):

    setup_logging(output_files_prefix + '.log')
    matching_data = get_data(config=config, rebuild_scores_file=rebuild_scores_file)
    paper_reviewer_df = matching_data.paper_reviewer_df
    reviewer_df = matching_data.reviewer_df
    distance_df = matching_data.distance_df


    if valid_reviewers_file:
        logger.info('Filtering by valid reviewers...')
        valid_reviewers = pd.read_csv(valid_reviewers_file)['reviewer'].values
        # Filter by valid reviewers
        reviewer_df = reviewer_df.query('reviewer in @valid_reviewers')
        distance_df = distance_df.query('reviewer_1 in @valid_reviewers and reviewer_2 in @valid_reviewers')
        paper_reviewer_df = paper_reviewer_df.query('reviewer in @valid_reviewers')

    if valid_papers_file:
        logger.info('Filtering by valid papers...')
        valid_papers = pd.read_csv(valid_papers_file)['paper'].values
        paper_reviewer_df = paper_reviewer_df.query('paper in @valid_papers')



    master_conflict_coreview_vars = set()
    if master_conflicts_file is not None:
        df = pd.read_csv(master_conflicts_file)
        master_conflict_coreview_vars = set(zip(df.rid1, df.rid2, df.pid))
        logger.info("Read %d conflicts from master conflict file: %s" % (len(df.index), master_conflicts_file))

    if num_coreview_vars is not None:
        subsample_co_review_vars = set(get_coreview_vars(config=config, distance_df=distance_df, paper_reviewer_df=paper_reviewer_df,num_coreview_vars=num_coreview_vars,d1=config['HYPER_PARAMS']['include_d1'],valid_papers=valid_papers, valid_reviewers=valid_reviewers))
    else:
        subsample_co_review_vars = set()


    d0_pairs = set()
    d1_pairs = set()
    existing_pairs = subsample_co_review_vars | master_conflict_coreview_vars
    per_paper_num_indicators = None
    per_reviewer_num = None

    # Step -1: Am I resuming?
    def gen_problem_name(q):
        return f"{output_files_prefix}_iter_{q}"

    def gen_problem_path(q, suffix=''):
        name = to_name(gen_problem_name(q))
        if suffix:
            name = name.replace('.lp', suffix)
        return name

    # Check to see if we can warm start from something old
    i = 0
    found = False

    # Step 0: Read in stuff
    papers = paper_reviewer_df.reset_index()['paper'].unique()
    records = []
    for paper in papers:
        records.append({'paper':paper,'pc_k': config['HYPER_PARAMS']['sparsity_k'],'spc_k': config['HYPER_PARAMS']['sparsity_k'],'ac_k': config['HYPER_PARAMS']['sparsity_k']})

    per_paper_num_indicators = pd.DataFrame.from_records(records).set_index('paper')
    per_paper_num_indicators.to_csv('per_paper_num_indicators.csv')

    for j in reversed(range(1000)):
        search_path = gen_problem_path(j)
        if os.path.exists(search_path):
            i = j
            found = True
            break
    if found:
        logger.info(f"Resuming from iteration {i}. Note that this might be 0 if you were part way through that iter")

        try:
            tuple_path = gen_problem_path(j, suffix=CONFLICT_SUFFIX)
            logger.info(f"Reading bad tuples from {tuple_path}")
            old_tuples = pd.read_csv(tuple_path)
            for r in old_tuples.itertuples():
                existing_pairs.add((r.j, r.jp, r.pid))
        except:
            logger.info("Could not read a previous conflict file. Starting up variables from scratch")

        try:
            per_paper_num_indicators_file = gen_problem_path(j, suffix=NUM_VARIABLE_INDICATORS_SUFFIX)
            per_paper_num_indicators = pd.read_csv(per_paper_num_indicators_file).set_index('paper')
        except:
            logger.info(f"Could not read a previous num variables indicators file. Starting up setting num indicators to {config['HYPER_PARAMS']['sparsity_k']}")

        per_reviewer_num_file = gen_problem_path(j, suffix=NUM_REVIEWER_SUFFIX)
        if os.path.isfile(per_reviewer_num_file):
            per_reviewer_num = pd.read_csv(per_reviewer_num_file)
        else:
            logger.info(f"Could not read a previous per_reviewer_num_file. Looked for {per_reviewer_num_file}")

    else:
        logger.info("No previous iterations found. Starting from the beginning at i=0")

    if per_reviewer_num is None:
        per_reviewer_num = create_initial_per_reviewer_num(reviewer_df, config['HYPER_PARAMS']['sparsity_k'])

    while True:
        logger.info(f"Starting iteration {i}")

        # Step 1: Create ILP
        logger.info(f"Step 1: Problem generation")
        ilp_file = gen_problem_path(i)
        if os.path.exists(ilp_file):
            logger.info(f"Skipping problem generation since path exists {ilp_file}")
        else:
            co_review_vars_to_use = existing_pairs|(d0_pairs|d1_pairs if config['HYPER_PARAMS']['include_d1'] else d0_pairs)

            logger.info(f"Writing out bad tuples of reviewer/reviewer/papers")
            # to_block = pd.DataFrame(list(co_review_vars_to_use), columns=['j', 'jp', 'pid'])
            # to_block.to_csv(gen_problem_path(i, suffix=CONFLICT_SUFFIX) , index=False)

            ilp = MatchingILP(paper_reviewer_df,reviewer_df,distance_df,config,co_review_vars_to_use,output_files_prefix=output_files_prefix,add_soft_constraints=add_soft_constraints,fixed_variable_solution_file=fixed_variable_solution_file)
            ilp.create_ilp(lp_filename=ilp_file)

        # Step 2 Solve ILP with warm start
        logger.info(f"Step 2: Solving problem")
        solution_file = ilp_file.replace('.lp', '.sol')
        if os.path.exists(solution_file):
            logger.info(f"Skipping problem solving since path exists: {solution_file}")
        else:
            if i == 0 and initial_solution is not None:
                warm_start = initial_solution
            elif i == 0:
                warm_start = None
            else:
                warm_start = gen_problem_path(i - 1, suffix='.sol')
            # TODO: paramterize abstol
            logger.info(f'Abstol:{abstol}')
            logger.info(f'Relative MIP Gap:{relative_mip_gap}')
            solution_file = solve(ilp_file, warm_start=warm_start,abstol=abstol, relative_mip_gap=relative_mip_gap)
        
        # Step 3: Analyze solution
        logger.info(f"Step 3: Parse and Analyze solution")
        parsed_solution, violation_df = analyse_solution(config, solution_file, matching_data=matching_data)

        # Step 4: Parse conflicts and update pairs
        logger.info(f"Step 4: Update constraints")
        violated_d0_pairs = list(violation_df.query('d == 0').drop('d',axis=1).itertuples(index=False))
        violated_d1_pairs = list(violation_df.query('d == 1').drop('d',axis=1).itertuples(index=False))
        logger.info(f'{len(violated_d0_pairs)} d0 and {len(violated_d1_pairs)} d1 coauthor constraints violated.')

        update_unused_reviewers(per_reviewer_num, reviewer_df, parsed_solution.df, config['HYPER_PARAMS']['sparsity_k'], filename=gen_problem_path(i, NUM_REVIEWER_SUFFIX))
        per_paper_num_indicators = parse_unassigned_papers(parsed_solution, per_paper_num_indicators, k=config['HYPER_PARAMS']['sparsity_k'], filename=gen_problem_path(i, NUM_VARIABLE_INDICATORS_SUFFIX))

        d0_pairs_size = len(d0_pairs)
        d1_pairs_size = len(d1_pairs)

        violated_unpenalized_d0_triples = set(violated_d0_pairs) - d0_pairs
        violated_unpenalized_d1_triples  = set(violated_d1_pairs) - d1_pairs

        violated_unpenalized_d0_pairs = set()
        violated_unpenalized_d1_pairs = set()


        for (r_i,r_j,p) in violated_unpenalized_d0_triples:
            reviewer_pair_set = frozenset([r_i,r_j])
            violated_unpenalized_d0_pairs |= reviewer_pair_set

        for (r_i,r_j,p) in violated_unpenalized_d1_triples:
            violated_unpenalized_d1_pairs |= set([(r_i,r_j)])

        with open(ilp_file.replace('.lp', '_status.json'),'r') as f:
            status_dict = json.load(f)

        full_objective = status_dict['objective'] + len(violated_unpenalized_d0_pairs) * config['HYPER_PARAMS']['coreview_dis0_pen'] + len(violated_unpenalized_d1_pairs) * config['HYPER_PARAMS']['coreview_dis1_pen']

        with open(ilp_file.replace('.lp', '_status.json'),'w') as f:
            status_dict['full_objective'] = full_objective
            f.write(json.dumps(status_dict))

        d0_pairs |= set(violated_d0_pairs)
        d1_pairs |= set(violated_d1_pairs)

        # Need to take into account existing vars

        if not add_soft_constraints:
            logger.info(f'Soft constraints turned off. No row generation. Terminating...')
            break
        if no_iterate:
            logger.info('Iterating turned off. Terminating...')
            break
        # Terminate if no new correview conflict were added
        if (len(d0_pairs) == d0_pairs_size and len(d1_pairs) == d1_pairs_size):
            logger.info(f'No more coauthor constraints to add. Terminated at iteration {i} with {len(d0_pairs)} d0 and {len(d1_pairs)} d1 coauthor constraints')
            break

        if i >= max_iter:
            logger.info(f'Exceeded max iteration of {max_iter}')
            break

        i += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #### Experiment parameters
    parser.add_argument('--output_files_prefix', type=str, default='itertest')
    parser.add_argument('--config_file', type=str, default='config.yml')
    parser.add_argument('--rebuild_scores_file', action='store_true')
    parser.add_argument('--fixed_variable_solution_file',type=str, default=None)
    parser.add_argument('--relax_paper_capacity', action='store_true')
    parser.add_argument('--valid_papers_file', type=str, default=None)
    parser.add_argument('--valid_reviewers_file', type=str, default=None)

    #### Row/column geneneration parameters ####
    parser.add_argument('--sparsity_k', type=int, default=50)
    parser.add_argument('--score_threshold', type=float, default=0.0)
    parser.add_argument('--num_coreview_vars', type=int, default=None)
    parser.add_argument('--master_conflicts_file', type=str, default=None)
    parser.add_argument('--initial_solution', type=str, default=None)
    parser.add_argument('--no_iterate', action='store_true')
    parser.add_argument('--max_iter', type=int, default=10000000)

    #### Parameters ####
    parser.add_argument('--add_soft_constraints', action='store_true')
    # Paper/reviewer capacity
    parser.add_argument('--max_reviews_per_paper_PC', type=int, default=None)
    parser.add_argument('--max_reviews_per_paper_SPC', type=int, default=None)
    parser.add_argument('--max_reviews_per_paper_AC', type=int, default=None)
    parser.add_argument('--max_papers_per_reviewer_PC', type=int, default=None)
    parser.add_argument('--max_papers_per_reviewer_SPC', type=int, default=None)
    parser.add_argument('--max_papers_per_reviewer_AC', type=int, default=None)
    # Region
    parser.add_argument('--region_reward', type=float, default=None)
    # Seniority
    parser.add_argument('--sen_reward', type=float, default=None)
    # Coreview
    parser.add_argument('--coreview_dis0_pen', type=float, default=None)
    parser.add_argument('--coreview_dis1_pen', type=float, default=None)
    parser.add_argument('--remove_non_symmetric', action='store_true', default=True)
    parser.add_argument('--include_d1', action='store_true', default=True)
    # Cycle
    parser.add_argument('--cycle_pen', type=float, default=None)
    # Paper distribution
    parser.add_argument('--no_paper_distribution_pen', action='store_true')

    # CPLEX params
    parser.add_argument('--abstol', type=float, default=None)
    parser.add_argument('--relative_mip_gap', type=float, default=None)

    args = parser.parse_args()

    with open(args.config_file, 'rb') as fh:
        config = yaml.load(fh,Loader=yaml.FullLoader)

    argv = [x.replace('--','') for x in sys.argv]

    for arg in vars(args):
        if arg in argv:
            name = arg
            value = getattr(args, arg)

            if name in config['HYPER_PARAMS']:
                config['HYPER_PARAMS'][name] = value

    main(output_files_prefix=args.output_files_prefix,
        config=config,
        num_coreview_vars=args.num_coreview_vars,
        master_conflicts_file=args.master_conflicts_file,
        initial_solution=args.initial_solution,
        rebuild_scores_file=args.rebuild_scores_file,
        fixed_variable_solution_file=args.fixed_variable_solution_file,
        abstol=args.abstol,
        relative_mip_gap=args.relative_mip_gap,
        valid_papers_file=args.valid_papers_file,
        valid_reviewers_file = args.valid_reviewers_file,
        add_soft_constraints=args.add_soft_constraints,
        no_iterate=args.no_iterate,
        max_iter=args.max_iter)
