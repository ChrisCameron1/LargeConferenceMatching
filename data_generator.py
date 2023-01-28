import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging
import logging.config
from iter_solve import setup_logging

logger = logging.getLogger(__name__)


def main(n_papers=1000, n_reviewers=3000, n_regions=5, output_dir='toy_data'):
    setup_logging('data_generator.log')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating fake conference data with {n_papers} papers, {n_reviewers} reviewers, and {n_regions} regions. Data will be stored in {output_dir}")

    papers = np.arange(n_papers)

    ### BIDS FILE
    records = []
    for reviewer_id in range(n_reviewers):
        bid_odds = np.random.binomial(1, 15/n_papers, n_papers)
        for paper in papers:
            if bid_odds[paper]:
                bid = np.random.choice([0.05, 1, 2, 4, 6])
                record = dict(paper=paper, reviewer=reviewer_id, bid=bid)
                records.append(record)
    pd.DataFrame.from_records(records).to_csv(f'{output_dir}/bids.csv', index=False)
    logger.info(f"Generated {len(records)} bids")

    ### REVIEWERS FILE
    records = []
    for reviewer_id in range(n_reviewers):
        record = dict(reviewer=reviewer_id)
        record['role'] = 'PC'
        record['seniority'] = np.random.choice([0, 1, 2, 3])
        conflict_odds = np.random.binomial(1, 1/n_papers, n_papers)
        authored_odds = np.random.binomial(1, 1/n_papers, n_papers)
        conflict_papers = [p for (i, p) in enumerate(papers) if conflict_odds[i]]
        authored_papers = [p for (i, p) in enumerate(papers) if authored_odds[i]]
        record['conflict_papers'] = conflict_papers
        record['authored'] = authored_papers
        record['region'] = np.random.choice([f'Region{i}' for i in range(n_regions)])
        records.append(record)
    pd.DataFrame.from_records(records).to_csv(f'{output_dir}/reviewers.csv', index=False)
    logger.info(f"Generated {len(records)} reviewers")

    ### COAUTHOR_DISTANCE_FILE
    records = []
    for reviewer_1_id in range(n_reviewers):
        distance_odds = np.random.binomial(1, 1/n_reviewers, n_reviewers)
        for reviewer_2_id in range(n_reviewers):
            if distance_odds[reviewer_2_id]:
                record = dict()
                record['reviewer_1'] = reviewer_1_id
                record['reviewer_2'] = reviewer_2_id
                record['distance'] = np.random.choice([0, 1])
                records.append(record)
    pd.DataFrame.from_records(records).to_csv(f'{output_dir}/distances.csv', index=False)
    logger.info(f"Generated {len(records)} coauthor distance entries")

    ### SCORES FILE
    reviewers = np.repeat(np.arange(n_reviewers), n_papers)
    papers = np.tile(papers, n_reviewers)
    ntmps = np.random.rand(n_reviewers * n_papers)
    nacl = np.random.rand(n_reviewers * n_papers)
    nk = np.random.rand(n_reviewers * n_papers)
    scores_df = pd.DataFrame({
        'ntpms': ntmps,
        'nacl': nacl,
        'nk': nk,
        'reviewer': reviewers,
        'paper': papers
    })
    # Zero out a bunch of entires to mimic unqualified reviewers. Additional reviewers will get zeroed out because of the threshold later too.
    score_columns = ['ntmps', 'nacl', 'nk']
    zero_percentage = 0.5
    should_zero = np.random.binomial(1, zero_percentage, n_reviewers * n_papers)
    scores_df.loc[should_zero, score_columns] = 0
    scores_df.to_csv(f'{output_dir}/scores.csv', index=False)
    logger.info(f"Generated {len(scores_df)} scores")

    logger.info("All done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #### Experiment parameters
    parser.add_argument('--n_papers', type=int, default=1000)
    parser.add_argument('--n_reviewers', type=int, default=3000)
    parser.add_argument('--n_regions', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='toy_data')

    args = parser.parse_args()

    main(n_papers=args.n_papers, n_reviewers=args.n_reviewers, output_dir=args.output_dir, n_regions=args.n_regions)
