import pandas as pd
import numpy as np
from tqdm import tqdm

ROLE_2_MULTIPLER= {
    'AC': 10,
    'SPC': 5,
    'PC': 1,
}

def create_paper_reviewer_df(scores_df=None, 
							reviewer_df=None,
							bids_df=None, 
							per_reviewer_num_indicators=None,
							per_paper_num_indicators=None,
							k=10,
							score_threshold=0.15):

	# Filter out scores below threshold
	scores_df = scores_df.query(f'agg_score > {score_threshold}')
	# Add role column
	scores_df['role'] = scores_df.apply(lambda row: reviewer_df.loc[row.name[1],'role'],axis=1)


	reviewers = reviewer_df.index.values
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

	# Filter out conflicts
	to_delete = []
	for reviewer in reviewers:
		to_delete += [(paper, reviewer) for paper in reviewer_df.loc[reviewer]['conflict_papers']]

	
	scores_df = scores_df.drop(index=to_delete, errors='ignore') # Only existing labels are dropped

	# Sort dataframe by scores
	scores_df = scores_df.sort_values(by=['agg_score'],ascending=False)
	dfs=[]
	# Add k best papers per reviewer
	for reviewer in tqdm(reviewers, desc='Adding best papers for reviewers'):
		reviewer_k = per_reviewer_num_indicators.loc[reviewer, 'k']
		dfs.append(scores_df.query(f'reviewer == {reviewer}').head(reviewer_k))

	# Add k best reviewers per paper
	for paper in tqdm(papers,desc='Adding best reviewers for papers'):
		paper_df = scores_df.query(f"paper == {paper}")
		for role in ["PC","SPC","AC"]:
			paper_k = per_paper_num_indicators.loc[paper][f'{role}_k']
			dfs.append(paper_df.query(f'role == "{role}"').head(paper_k))


	paper_reviewer_df = pd.concat(dfs).drop_duplicates()
	# print(paper_reviewer_df.columns)
	# paper_reviewer_df = paper_reviewer_df.set_index(['paper','reviewer'])

	# Add bids
	paper_reviewer_df = paper_reviewer_df.join(bids_df)#.merge(bids_df, how='left')#, validate='1:1') #.join(bids_df)
	paper_reviewer_df['bid'].fillna(0)

	# Add indicator, whether reviewer is author on given paper
	paper_reviewer_df['authored'] = paper_reviewer_df.apply(lambda row: 1 if row.name[0] in reviewer_df.loc[row.name[1]]['authored'] else 0, axis=1)
	# for (paper,reviewer), row in paper_reviewer_df.iterrows():
	# 	if paper in reviewer_df.loc[reviewer]['authored'].values:
	# 		paper_reviewer_df
	# paper_reviewer_df.merge(reviewer_as_author, how='left', validate='1:1')
	# paper_reviewer_df['authored'].fillna(0)

	return paper_reviewer_df