import pandas as pd
import itertools
from tqdm import tqdm

def get_coreview_vars(distance_df=None,
						paper_reviewer_df=None,
						num_coreview_vars=1000,
						d0=True, 
						d1=False,
						valid_papers=None,
						valid_reviewers=None):

	print('Generating %d coreview vars...' % num_coreview_vars)

	records = []

	for paper in tqdm(valid_papers):

		for (a,b) in itertools.combinations(valid_reviewers, 2):
			sorted_tuple = sorted((a,b))
			try:
				distance = distance_df.loc[sorted_tuple]['distance'].item()
			except ValueError:
				print(sorted_tuple)
				print(paper)
			except KeyError:
				print(sorted_tuple)
				print(paper)
				continue

			min_score = min(paper_reviewer_df.loc[(paper,a)]['score'].item(), paper_reviewer_df.loc[(paper,b)]['score'].item())

			if (distance == 0 and d0) or (distance == 1 and d1):
				record = dict(
						paper = paper,
						reviewer_i = sorted_tuple[0],
						reviewer_j = sorted_tuple[1],
						min_score = min_score)
				records.append(record)

	df = pd.DataFrame.from_records(records)

	num_records = len(records)
	if num_records < num_coreview_vars:
		print('Warning: Num eligible coreviews %d is below requested limit %d. Taking all...' % (num_records, num_coreview_vars))

	num_coreview_vars = min(len(records)-1, num_coreview_vars)

	top_k = df.sort_values('min_score')[-num_coreview_vars:]

	return list(zip(top_k.reviewer_i, top_k.reviewer_j, top_k.paper))


if __name__ == "__main__":

    get_coreview_vars()





