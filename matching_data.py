import pandas as pd
from dataclasses import dataclass
from compute_scores import compute_scores
import json
from create_indicator import create_paper_reviewer_df
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class MatchingData:
	reviewer_df: pd.DataFrame
	paper_reviewer_df: pd.DataFrame
	distance_df: pd.DataFrame
	
def get_data(config, per_reviewer_num_indicators=None, per_paper_num_indicators=None, rebuild_scores_file=False):
	logger.info("Computing aggregate scores from raw scores...")

	if os.path.isfile(config['CACHED_SCORES_FILE']) and not rebuild_scores_file:
		logger.info(f"Cached file {config['CACHED_SCORES_FILE']} exists. Reading from file...")
		scores_df = pd.read_csv(config['CACHED_SCORES_FILE']).set_index(['paper','reviewer'])
	else:
		scores_df = compute_scores(config)

	bids_df = pd.read_csv(config['BIDS_FILE']).set_index(['paper','reviewer'])
	reviewer_df = pd.read_csv(config['REVIEWERS_FILE']).set_index('reviewer')
	reviewer_df['conflict_papers'] = reviewer_df['conflict_papers'].apply(lambda x: json.loads(x))
	reviewer_df['authored'] = reviewer_df['authored'].apply(lambda x: json.loads(x))
	reviewer_df['authored_any'] = reviewer_df['authored'].apply(lambda x: len(x) > 0)
	
	reviewers = reviewer_df.index.values
	missing_reviewers = set(scores_df.index.unique('reviewer')) - set(reviewers)

	scores_df = scores_df[~scores_df.index.get_level_values('reviewer').isin(list(missing_reviewers))]

	distance_df = pd.read_csv(config['COAUTHOR_DISTANCE_FILE']).set_index(['reviewer_1','reviewer_2'])

	logger.info("Sparsifying problem...")
	paper_reviewer_df = create_paper_reviewer_df(config=config,
			scores_df=scores_df, 
            reviewer_df=reviewer_df,
            bids_df=bids_df, 
            k=config['HYPER_PARAMS']['sparsity_k'],
            score_threshold=config['HYPER_PARAMS']['score_threshold'],
            per_reviewer_num_indicators=per_reviewer_num_indicators,
			per_paper_num_indicators=per_paper_num_indicators)

	# Filter out any reviewers from reviewer_df that are not in paper_reviewer_df
	existing_reviewers = paper_reviewer_df.index.get_level_values('reviewer').unique()
	reviewer_df = reviewer_df.reset_index().query('reviewer in @existing_reviewers').set_index('reviewer')

	return MatchingData(
        reviewer_df=reviewer_df,
        paper_reviewer_df=paper_reviewer_df,
    	distance_df=distance_df)