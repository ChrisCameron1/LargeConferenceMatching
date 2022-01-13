import sys
import csv
import logging
from collections import defaultdict

import numpy as np

def calc_reviewer_id_mapping(reviewers, author_col, names_to_ignore=set([])):
    assert(author_col == 'name' or author_col == 'id')
    reviewer_id_map = defaultdict(lambda: [])
    author_col_plural = author_col+'s'
    for j, x in enumerate(reviewers):
        for reviewer in x[author_col_plural]:
            reviewer_id_map[reviewer].append(j)
    
    if author_col == 'name':
        # removing the entires (names) that maps to multiple reviewers.
        clean_reviewer_id_map = defaultdict(lambda: [])
        for reviewer_name in reviewer_id_map.keys():
            if reviewer_name.lower() not in names_to_ignore:
                clean_reviewer_id_map[reviewer_name] = reviewer_id_map[reviewer_name]
        return clean_reviewer_id_map
    else:
        return reviewer_id_map    


def calc_reviewer_db_mapping(reviewers, db, names_to_ignore, author_field='authors', warn_under=1, print_warnings=False, abstract_id_pairs=None):
    """ Calculate correspondence between reviewers and papers

    :param reviewers: A list of reviewer names, or reviewer IDs
    :param db: A DB with papers, and a field `author_field` for authors
    :param author_col: The column in the author field to check in the DB
    :param author_field: The field to look for in the DB
    :param warn_under: Throw a warning if a reviewer has few papers under this value
    :return: an NP array with rows as reviewers, columns as entries in the DB
    """
    print(f'Calculating reviewer-paper mapping for {len(reviewers)} reviewers and {len(db)} papers', file=sys.stderr)
    reviewer_id_map_name = calc_reviewer_id_mapping(reviewers, 'name', names_to_ignore=names_to_ignore)
    reviewer_id_map_id = calc_reviewer_id_mapping(reviewers, 'id')

    #out_csvfile = open("abstracts/2000.tsv", 'a') 
    #out_csvwriter = csv.writer(out_csvfile, delimiter="\t")
    # lets do this for just when the db size is reduced

    new_mapping = {}
    if abstract_id_pairs == None:
        mapping = np.zeros( (len(db), len(reviewers)) )
    else:
        mapping = np.zeros( (len(db) + len(abstract_id_pairs), len(reviewers)) )
    for i, entry in enumerate(db):
        for cols in entry[author_field]:
            js = []
            author_col = 'id'
            reviewer_id_map = reviewer_id_map_id
            if author_col in cols:
                if cols[author_col] in reviewer_id_map:
                    js = reviewer_id_map[cols[author_col]]
            else:
                for x in cols[author_col+'s']:
                    if x in reviewer_id_map:
                        js.extend(reviewer_id_map[x])

            author_col = 'name'
            reviewer_id_map = reviewer_id_map_name
            if author_col in cols: 
                if cols[author_col] in reviewer_id_map:
                    js = reviewer_id_map[cols[author_col]]
            else:
                for x in cols[author_col+'s']:
                    if x in reviewer_id_map:
                        js.extend(reviewer_id_map[x])

            for j in js:
                mapping[i,j] = 1
                if j not in new_mapping:
                    new_mapping[j] = []
                new_mapping[j].append(i)
                #abstracts = ' '.join(entry['paperAbstract'].split())
                #if abstracts.strip() != "":
                #    out_csvwriter.writerow([reviewers[j]['email'], abstracts.strip()])
    #out_csvfile.close()
    
    # add the extracted abstracts and make the corresponding reviewer id column 1. 
    if abstract_id_pairs != None:
        for i, (abstract, reviewer_id) in enumerate(abstract_id_pairs):
            mapping[len(db)+i,reviewer_id] = 1
            if reviewer_id not in new_mapping:
                new_mapping[reviewer_id] = []
            new_mapping[reviewer_id].append(len(db)+i)

    num_papers = mapping.sum(axis=0)
    num_reviewers_with_no_papers = 0

    '''
    out_csvfile_path = "logs/no_profile_revs.csv"
    with open(out_csvfile_path, 'a') as out_csvfile:
        out_csvwriter = csv.writer(out_csvfile)
        index = 0
        for rev, num in zip(reviewers, num_papers):
            name = rev['names'][0]
            if print_warnings and num < warn_under:
                num_reviewers_with_no_papers += 1
                logging.warn(f'Reviewer {name} ({rev["email"]}) has {num} papers in the database')
                out_csvwriter.writerow([rev["email"], rev["ids"]])
            
            index += 1
        if print_warnings:    
            logging.warn(f'In total {num_reviewers_with_no_papers} reviewers have no papers in the database')
    '''
    for rev, num in zip(reviewers, num_papers):
        name = rev['names'][0]
        if print_warnings and num < warn_under:
            num_reviewers_with_no_papers += 1
            logging.warn(f'Reviewer {name} ({rev["email"]}) has {num} papers in the database')
    if print_warnings:    
        logging.warn(f'In total {num_reviewers_with_no_papers} reviewers have no papers in the database')

    return mapping, new_mapping

def print_text_report(query, file):
    print('----------------------------------------------', file=file)
    print('*** Paper Title', file=file)
    print(query['title'], file=file)
    print('*** Paper Abstract', file=file)
    print(query['paperAbstract'], file=file)
    print('\n*** Similar Papers', file=file)

    for x in query['similarPapers']:
        my_title, my_abs, my_score = x['title'], x['paperAbstract'], x['score']
        print(f'# {my_title} (Score {my_score})\n{my_abs}', file=file)
    print('', file=file)
    print('\n*** Best Matched Reviewers', file=file)
    for x in query['topSimReviewers']:
        my_name, my_score = x['names'][0], x['score']
        print(f'# {my_name} (Score {my_score})', file=file)
    print('\n*** Assigned Reviewers', file=file)
    for x in query['assignedReviewers']:
        my_name, my_score = x['names'][0], x['score']
        print(f'# {my_name} (Score {my_score})', file=file)
    print('', file=file)


def print_progress(i, mod_size):
    if i != 0 and i % mod_size == 0:
        sys.stderr.write('.')
        if int(i/mod_size) % 50 == 0:
            print(i, file=sys.stderr)
        sys.stderr.flush()