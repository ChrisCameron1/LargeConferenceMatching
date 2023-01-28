import pandas as pd
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

ROLE_2_MULTIPLER= {
    'AC': 10,
    'SPC': 5,
    'PC': 1,
}

def create_paper_reviewer_df(config=None,
							scores_df=None, 
							reviewer_df=None,
							bids_df=None, 
							per_reviewer_num_indicators=None,
							per_paper_num_indicators=None,
							k=10,
							score_threshold=0.15):

	# Filter out scores below threshold
	logger.info(f'Filtering scores below threshold {score_threshold}')
	num_entries_before = scores_df.size
	scores_df = scores_df.query(f'score > {score_threshold}').copy()
	num_entries_after = scores_df.size
	logger.info(f'Keeping {num_entries_after/num_entries_before} fraction of scores after filtering below threshold {score_threshold}...')
	# Add role column
	role_dict = reviewer_df['role'].to_dict()
	scores_df['role'] = scores_df.reset_index()['reviewer'].map(role_dict).values

	reviewers = scores_df.index.unique('reviewer').values
	papers = scores_df.index.unique('paper').values

	if per_reviewer_num_indicators is None:
		records = []
		for reviewer in reviewers:
			role = reviewer_df.loc[reviewer]['role']
			records.append({'reviewer': reviewer, 'k': k * ROLE_2_MULTIPLER[role]})
		per_reviewer_num_indicators = pd.DataFrame.from_records(records).set_index('reviewer')

	if per_paper_num_indicators is None:
		records = []
		for paper in papers:
			records.append({'paper':paper,'PC_k':k,'SPC_k':k,'AC_k':k})
		per_paper_num_indicators = pd.DataFrame.from_records(records).set_index('paper')

	logger.info('Deleting (paper,reviewer) pairs that appear in conflicts...')
	# Filter out conflicts
	to_delete = []
	for reviewer in tqdm(reviewers,desc='Getting conflict papers'):
		to_delete += [(paper, reviewer) for paper in reviewer_df.loc[reviewer]['conflict_papers']]

	logger.info(f'Dropping {len(to_delete)} conflict pairs...')
	# TODO: very slow!
	scores_df = scores_df.drop(index=to_delete, errors='ignore') # Only existing labels are dropped

	reviewers = scores_df.index.unique('reviewer').values
	papers = scores_df.index.unique('paper').values

	# Sort dataframe by scores
	logger.info('Sorting scores...')
	scores_df = scores_df.sort_values(by=['score'],ascending=False)
	# Set scores_df index to reviewer
	scores_df = scores_df.reset_index().set_index('reviewer')
	dfs=[]
	# Add k best papers per reviewer
	missing_count = {'PC': 0, 'SPC':0, 'AC':0}

	logger.info(f"Adding best {k* ROLE_2_MULTIPLER['PC']}, {k* ROLE_2_MULTIPLER['SPC']}, {k* ROLE_2_MULTIPLER['AC']} papers for each PC, SPC, and AC reviewer respectively...")
	for reviewer in tqdm(reviewers, desc='Adding best papers for reviewers'):
		reviewer_k = per_reviewer_num_indicators.loc[reviewer, 'k']
		role = reviewer_df.loc[reviewer]['role']
		k_reviewers_to_add = scores_df.loc[reviewer].head(reviewer_k)
		dfs.append(k_reviewers_to_add.reset_index())

	for role in ["PC","SPC","AC"]:
		logger.info(f'{missing_count[role]} {role} reviewers with no papers')

	# Set scores_df index to paper
	scores_df = scores_df.reset_index().set_index('paper')
	# Add k best reviewers per paper
	papers_to_delete=[]
	missing_count = {'PC': 0, 'SPC':0, 'AC':0}
	logger.info(f"Adding best {k} reviewers per paper...")
	for paper in tqdm(papers,desc='Adding best reviewers for papers'):
		paper_df = scores_df.loc[paper]
		for role in ["PC","SPC","AC"]:
			paper_k = per_paper_num_indicators.loc[paper][f'{role}_k']
			k_reviewers_to_add = paper_df.query(f'role == "{role}"').head(paper_k)
			if role =='PC' and len(k_reviewers_to_add) == 0:
				papers_to_delete.append(paper)
			if len(k_reviewers_to_add) == 0:
				missing_count[role]+=1
			dfs.append(k_reviewers_to_add.reset_index())
	for role in ["PC","SPC","AC"]:
		logger.info(f'{missing_count[role]} papers with no {role} reviewers')

	paper_reviewer_df = pd.concat(dfs)
	paper_reviewer_df = paper_reviewer_df.reset_index(drop=True).drop_duplicates().set_index(['paper','reviewer'])
	size_before = paper_reviewer_df.size
	logger.info(f'{len(papers_to_delete)} papers to delete: {papers_to_delete}...')
	paper_reviewer_df = paper_reviewer_df.drop(index=papers_to_delete, level='paper', errors='ignore')
	logger.info(f'{size_before - paper_reviewer_df.size} entries deleted')
	logger.info(f'Keeping {paper_reviewer_df.size/num_entries_after} fraction of scores after removing low-scoring (paper,reviewer) matches...')

	# Add bids
	logger.info('Adding bids...')
	paper_reviewer_df = paper_reviewer_df.join(bids_df)
	paper_reviewer_df['bid'] = paper_reviewer_df['bid'].fillna(config['DEFAULT_BID_WHEN_NO_BIDS'])

	return paper_reviewer_df