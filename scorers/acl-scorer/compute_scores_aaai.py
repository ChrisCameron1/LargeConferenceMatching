import csv
import sys
import math
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
from models import load_model
from collections import defaultdict
from sacremoses import MosesTokenizer

from model_utils import Example, unk_string
from suggest_utils import calc_reviewer_db_mapping, print_text_report, print_progress

BATCH_SIZE = 128
entok = MosesTokenizer(lang='en')

def create_embeddings(model, examps):
    """Embed textual examples

    :param examps: A list of text to embed
    :return: A len(examps) by embedding size numpy matrix of embeddings
    """
    # Preprocess examples
    print(f'Preprocessing {len(examps)} examples (.={BATCH_SIZE} examples)', file=sys.stderr)
    data = []
    for i, line in enumerate(examps):
        p1 = " ".join(entok.tokenize(line, escape=False)).lower()
        if model.sp is not None:
            p1 = model.sp.EncodeAsPieces(p1)
            p1 = " ".join(p1)
        wp1 = Example(p1)
        wp1.populate_embeddings(model.vocab, model.zero_unk, model.args.ngrams)
        if len(wp1.embeddings) == 0:
            wp1.embeddings.append(model.vocab[unk_string])
        data.append(wp1)
        print_progress(i, BATCH_SIZE)
    print("", file=sys.stderr)
    # Create embeddings
    print(f'Embedding {len(examps)} examples (.={BATCH_SIZE} examples)', file=sys.stderr)
    embeddings = np.zeros( (len(examps), model.args.dim) )
    for i in range(0, len(data), BATCH_SIZE):
        max_idx = min(i+BATCH_SIZE,len(data))
        curr_batch = data[i:max_idx]
        wx1, wl1 = model.torchify_batch(curr_batch)
        vecs = model.encode(wx1, wl1)
        vecs = vecs.detach().cpu().numpy()
        vecs = vecs / np.sqrt((vecs * vecs).sum(axis=1))[:, None] #normalize for NN search
        embeddings[i:max_idx] = vecs
        print_progress(i, BATCH_SIZE)
    print("", file=sys.stderr)
    return embeddings

def parseSubmissions(submissions_file, track_name):

    df = pd.read_excel(submissions_file, sheet_name=track_name)
    
    # assumes:
    # submission-id is in column 0
    # abstract is in column 4
    # submission status is in column 19
    # no of files  uploaded in column 25
    excel_submission_ids = [str(i) for i in df[df.columns[0]].values.tolist()][2:]
    excel_submission_abs = [str(i) for i in df[df.columns[4]].values.tolist()][2:]
    excel_submission_status = [str(i) for i in df[df.columns[19]].values.tolist()][2:]
    excel_submission_files_counts = [str(i) for i in df[df.columns[25]].values.tolist()][2:]

    submission_ids = []
    submission_abs = []
    for id, abs, status, file_count in zip(excel_submission_ids, excel_submission_abs, excel_submission_status, excel_submission_files_counts):
        # skip a submission if it has been withdrawn/desk-rejected OR has no files uploaded
        if status == "Awaiting Decision" and int(file_count) >= 1:
            submission_ids.append(id)
            submission_abs.append(abs)
        else:
            logging.warning("Paper skipped: " + id + " status: " + status)
    
    return submission_ids, submission_abs

def parseReviewers(reviewer_file, user_info_file, infered_ss_ids_map, ac=False):
    
    df = pd.read_excel(user_info_file)
    email_ssid_map = {}
    revs_with_inferred_ss_id = set([])
    for ind in df.index: 
        ss_url = df['Semantic Scholar URL'][ind]
        rev_email = df['Email'][ind]
        if str(ss_url) != "nan":
            if "https://www.semanticscholar.org/author/" not in ss_url:
                logging.warning("Reviewer has specified incorrect SS URL: " + rev_email)
            email_ssid_map[rev_email] = ss_url.replace("https://www.semanticscholar.org/author/", "")
        else:
            if rev_email in infered_ss_ids_map:
                inferred_ss_id = infered_ss_ids_map[rev_email]
                if "https://www.semanticscholar.org/author/" not in inferred_ss_id:
                    logging.warning("Reviewer has specified incorrect SS URL: " + rev_email)
                email_ssid_map[rev_email] = inferred_ss_id.replace("https://www.semanticscholar.org/author/", "")
                revs_with_inferred_ss_id.add(rev_email)
            else:
                email_ssid_map[rev_email] = None

    reviewers = []
    
    tsv_file = open(reviewer_file)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    no_user_details_count = 0
    no_ssid_count = 0
    ssid_count = 0
    revs_added_so_far = set([])
    for idx, row in enumerate(read_tsv):
        if idx == 0:
            continue
        
        if ac:
            reviewer_email = row[0].strip()
        else:
            reviewer_email = row[3]
        
        if ac and reviewer_email in revs_added_so_far:
            continue
        revs_added_so_far.add(reviewer_email)

        s2ids = []
        if ac:
            reviewer_org= ""
        else:
            reviewer_org= row[4]

        if reviewer_email in email_ssid_map:
            if email_ssid_map[reviewer_email] == None:
                no_ssid_count += 1
            else:
                if reviewer_email in revs_with_inferred_ss_id:
                    logging.info("Used inferred ss-id for " + reviewer_email)
                ssid_count += 1
                s2ids.append(email_ssid_map[reviewer_email])
        else:
            no_user_details_count += 1
            logging.warning(reviewer_email + ' missing in user information export')
        name = (row[0]).strip() + " " + (row[2]).strip()

        rev_data = {'name': name, 'ids': s2ids, 'email': reviewer_email, 'org': reviewer_org}
        reviewers.append(rev_data)

    for data in reviewers:
        if 'name' in data:
            data['names'] = [data['name']]
            del data['name']
    reviewer_names = [x['names'][0] for x in reviewers]

    logging.info('No. of Reviewers with s2id: ' + str(ssid_count))
    logging.info('No. of Reviewers with no s2id: '+ str(no_ssid_count))
    logging.info('No. of Reviewers with no info in user information export: '+ str(no_user_details_count))

    return reviewers, reviewer_names

# merged_csv: this file was provided by the COI owners (Yatin and Gabi)
# this file contains an entry of every user in CMT
# for certain users they had infered their semantic scholar ids
def get_infered_ss_ids(merged_csv):
    infered_ss_ids_map = {}

    if merged_csv == None:
        return infered_ss_ids_map

    csv_file = open(merged_csv)
    read_csv = csv.reader(csv_file)
    old_rev_mails = set([])
    for idx, row in enumerate(read_csv):
        if idx == 0:
            continue
        rev_email = row[2].strip()
        ss_id = row[5].strip()
        merged_ss_id = row[23].strip()
        if ss_id == "" and merged_ss_id != "":
            if merged_ss_id == "nan":
                continue
            merged_ss_id_split = merged_ss_id.split("##")
            infered_ss_ids = set([])
            for id in merged_ss_id_split:
                if id != "nan":
                    infered_ss_ids.add(id)
            if len(infered_ss_ids) == 1:
                infered_ss_ids_map[rev_email] = list(infered_ss_ids)[0]
    return infered_ss_ids_map

# we match certain users by using their names
# this method identifies the names to be ignored while matching
def getNamesToIgnore(user_info_file, reviewer_file, ac=False):
    
    names_to_ignore = set([])
    email_to_name_map = {}

    df = pd.read_excel(user_info_file)
    email_ssid_map = {}
    users = []
    name_2_idx_map = {}
    for ind in df.index: 
        name = (df['First Name'][ind].strip() + " " + df['Last Name'][ind].strip()).lower()
        email = df['Email'][ind].strip()
        email_to_name_map[email] = name
        ss_url = df['Semantic Scholar URL'][ind]
        if str(ss_url) == "nan":
            ss_url = ""

        gs_url = df['Google Scholar URL'][ind]
        if str(gs_url) == "nan":
            gs_url = ""

        dblp_url = df['DBLP URL'][ind]
        if str(dblp_url) == "nan":
            dblp_url = ""

        conflicts = df['Conflict Domains'][ind]
        conflicts_set = set([])
        if str(conflicts) != "nan":
            conflicts_set = set([x.strip() for x in conflicts.split(";")])

        email = df['Email'][ind]
        pub_emails = df['Publication Emails'][ind]
        pub_emails_set = set([])
        if str(pub_emails) != "nan":
            pub_emails_set = set([x.strip() for x in pub_emails.split(";")])
        
        users.append({"name":name, "s2id":ss_url.strip().lower(), "email":email, "pub_emails": pub_emails_set, "dblp_url": dblp_url, "gs_url":gs_url, "conflicts":conflicts_set})
        
        if name not in name_2_idx_map:
            name_2_idx_map[name] = []
        name_2_idx_map[name].append(ind)
    
    # these are clusters of users who share the same name.
    clusters_to_analyze = []

    already_added_reviewers = set([])
    tsv_file = open(reviewer_file)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    for idx, row in enumerate(read_tsv):
        if idx == 0:
            continue
        
        if ac:
            reviewer_email = row[0].strip()
            name = email_to_name_map[reviewer_email]
        else:
            reviewer_email = row[3]
            name = ((row[0]).strip() + " " + (row[2]).strip()).lower()

        if ac and reviewer_email in already_added_reviewers:
            continue
        already_added_reviewers.add(reviewer_email)
        
        if name in name_2_idx_map:
            if len(name_2_idx_map[name]) > 1:
                cluster = []
                for idx in name_2_idx_map[name]:
                    cluster.append(users[idx])
                clusters_to_analyze.append(cluster)
            else:
                # the reviewer doesnt have an entry in the user-information.xls (not regsitered) 
                # but there is another person with the same name in user information
                if (not ac) and  (row[5]).strip() == "":
                    names_to_ignore.add(name)
    
    # ignore names that have 2 ss-ids
    for cluster in clusters_to_analyze:
        if len(cluster) == 2:
            # check if they are the same based on email and pub-emails
            if cluster[0]["email"] in cluster[1]["pub_emails"] or cluster[1]["email"] in cluster[0]["pub_emails"]:
                pass
            elif cluster[0]["s2id"] != "" and cluster[0]["s2id"] == cluster[1]["s2id"]:
                pass
            elif cluster[0]["dblp_url"] != "" and cluster[0]["dblp_url"] == cluster[1]["dblp_url"]:
                pass
            elif cluster[0]["gs_url"] != "" and cluster[0]["gs_url"] == cluster[1]["gs_url"]:
                pass
            elif len(cluster[0]["conflicts"]) > 0 and cluster[0]["conflicts"] == cluster[1]["conflicts"]:
                pass
            else:
                names_to_ignore.add(cluster[0]["name"])
        else:
            same_s2_ids = True
            s2_id = cluster[0]["s2id"]
            if s2_id == "":
                same_s2_ids = False

            same_gs_ids = True
            gs_id = cluster[0]["gs_url"]
            if gs_id == "":
                same_gs_ids = False

            same_dblp_ids = True
            dblp_id = cluster[0]["dblp_url"]
            if dblp_id == "":
                same_dblp_ids = False

            for entry in cluster:
                if s2_id != entry["s2id"]:
                    same_s2_ids = False
                if gs_id != entry["gs_url"]:
                    same_gs_ids = False
                if dblp_id != entry["dblp_url"]:
                    same_dblp_ids = False
            
            if same_dblp_ids or same_s2_ids or same_gs_ids:
                pass
            else:
                names_to_ignore.add(cluster[0]["name"])
    
    return names_to_ignore

def calc_similarity_matrix(model, db, quer):
    db_emb = create_embeddings(model, db)
    quer_emb = create_embeddings(model, quer)
    print(f'Performing similarity calculation', file=sys.stderr)
    return np.matmul(quer_emb, np.transpose(db_emb))


def calc_aggregate_reviewer_score(rdb, rdb_new, all_scores, operator='max'):
    """Calculate the aggregate reviewer score for one paper

    :param rdb: Reviewer DB. NP matrix of DB papers by reviewers
    :param scores: NP matrix of similarity scores between the current papers (rows) and the DB papers (columns)
    :param operator: Which operator to apply (max, weighted_topK)
    :return: Numpy matrix of length reviewers indicating the score for that reviewer
    """
    agg = np.zeros( (all_scores.shape[0], rdb.shape[1]) )

    # this logic has been re-implemented to fit the scale for AAAI
    #** Re-implememntation ** 
    weighted_top = False
    max_weight = False
    if operator.startswith('weighted_top'):
        weighted_top = True
        k = int(operator[12:])
        weighting = np.reshape(1/np.array(range(1, k+1)), (k,1))
    if operator == 'max':
        max_weight = True


    print(f'Calculating aggregate scores for {all_scores.shape[0]} examples (.=10 examples)', file=sys.stderr)

    for i in range(all_scores.shape[0]):
        scores = all_scores[i]
        for rev_id, rdb_entries in rdb_new.items():
            rev_scores = scores[rdb_entries]
            if len(rev_scores) < k:
                b = np.array([0]*(k-len(rev_scores)))
                topk = np.concatenate((rev_scores,b))
            else:
                topk = np.partition(rev_scores, -k)
                topk = topk[-k:]
            topk.sort()
            #topk = topk[::-1]
            agg_score = (topk.reshape((len(topk), 1))*weighting).sum(axis=0)
            agg[i][rev_id] = agg_score[0]
        print_progress(i, mod_size=10)
    
    #** Original Code ** 
    '''
    for i in range(all_scores.shape[0]):
        scores = all_scores[i]
        INVALID_SCORE = 0
        # slow -- 2-3 secs
        scored_rdb = rdb * scores.reshape((len(scores), 1)) + (1-rdb) * INVALID_SCORE
        #backup_scored_rdb = rdb * scores.reshape((len(scores), 1)) + (1-rdb) * INVALID_SCORE
        if max_weight:
            agg[i] = np.amax(scored_rdb, axis=0)
        elif weighted_top:
            #k = int(operator[12:])
            #weighting = np.reshape(1/np.array(range(1, k+1)), (k,1))
            # slow -- 2-3 secs
            #scored_rdb.sort(axis=0)
            #topk = scored_rdb[-k:,:]
            #agg[i] = (topk*weighting).sum(axis=0)
            
            topk_unsorted = np.partition(scored_rdb, -k, axis=0)
            topk_unsorted = topk_unsorted[-k:,:]
            topk_unsorted.sort(axis=0)
            agg[i] = (topk_unsorted*weighting).sum(axis=0)
            
        else:
            raise ValueError(f'Unknown operator {operator}')
        print_progress(i, mod_size=10)
        '''
    print('', file=sys.stderr)
    return agg

def create_email_to_id_mapping(reviewers):
    email_to_id_mapping = {}
    for j, x in enumerate(reviewers):
        email_to_id_mapping[x['email']] = j
    return email_to_id_mapping

def create_abstracts_to_id_pairs(abstract_file, email_to_id_mapping):
    
    abstract_id_pairs = []
    tsv_file = open(abstract_file)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    unique_reviewers = set([])
    for idx, row in enumerate(read_tsv):
        reviewer_email = row[0]
        abstract = row[1]
        if reviewer_email in email_to_id_mapping:
            abstract_id_pairs.append((abstract, email_to_id_mapping[reviewer_email]))
            unique_reviewers.add(reviewer_email)
    
    return abstract_id_pairs, unique_reviewers

def load_coi_matrix(coi_file, email_to_id_mapping, submission_ids):
    
    paper_id_to_paper_idx_mapping = {}
    for idx, paper_id in enumerate(submission_ids):
        paper_id_to_paper_idx_mapping[paper_id] = idx
    
    coi_matrix = np.zeros((len(submission_ids), len(email_to_id_mapping)))
    csv_file = open(coi_file)
    read_csv = csv.reader(csv_file)
    no_of_conflicts_loaded = 0
    total_conflicts = 0
    for idx, row in enumerate(read_csv):
        if idx == 0:
            continue
        paper_id = row[1]
        reviewer_email = row[2]
        if reviewer_email in email_to_id_mapping and paper_id in paper_id_to_paper_idx_mapping:
            coi_matrix[paper_id_to_paper_idx_mapping[paper_id]][email_to_id_mapping[reviewer_email]] = 1
            no_of_conflicts_loaded += 1
        total_conflicts += 1
    logging.info(str(no_of_conflicts_loaded) + " out of " + str(total_conflicts) + " conflicts were relevent")
    
    return coi_matrix

def log_time_taken(event_str, start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    timestr = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    logging.info("Time taken to " + event_str + ": " + timestr)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--submission_file", type=str, required=True, help="export of submissions")
    parser.add_argument("--reviewer_file", type=str, required=True, help="export of reviewer names and IDs that can review this time")
    parser.add_argument("--user_info_file", type=str, required=True, help="export of user names and and SSIDs")
    parser.add_argument("--db_file", type=str, required=True, help="File (in s2 json format) of relevant papers from reviewers")
    parser.add_argument("--abstracts_file", type=str, required=True, help="file with abstracts of papers authored by reviewers")
    parser.add_argument("--merged_user_info_file", type=str, default=None, help="export of user names and and SSIDs that contains the merged columns")

    parser.add_argument("--model_file", help="filename to load the pre-trained semantic similarity file.")
    parser.add_argument("--aggregator", type=str, default="weighted_top3", help="Aggregation type (max, weighted_topN where N is a number)")
    
    parser.add_argument("--save_paper_matrix", help="A filename for where to save the paper similarity matrix")
    parser.add_argument("--load_paper_matrix", help="A filename for where to load the cached paper similarity matrix")
    parser.add_argument("--save_aggregate_matrix", help="A filename for where to save the reviewer-paper aggregate matrix")
    parser.add_argument("--load_aggregate_matrix", help="A filename for where to load the cached reviewer-paper aggregate matrix")

    args = parser.parse_args()

    # there were three tracks in AAAI 2021
    # 1 - Main Track
    # 2 - AI for Social Impact
    # 3 - NeurIPS EMNLP Fast Track
    # if the reviewer file is named "Reviewers-1.txt", it means role is "Reviewer" and track is "main track"

    track_name = "AAAI2021"
    if "2" in args.reviewer_file:
        track_name = "AI for Social Impact"
    if "3" in args.reviewer_file:
        track_name = "NeurIPS EMNLP Fast Track"

    # This code assumes --reviewer_file argument points to an export from CMT
    # according to CMT
    # Reviewer-1.txt is the export of all PCs in track-1
    # MetaReviewers-2.txt is the export of all SPCs in track-2
    # MetaReviewerSubjectAreas-1.txt is the export of all ACs in track-1
    
    role = "PC"
    ac_flag = False
    if "SubjectAreas" in args.reviewer_file:
        role = "AC"
        ac_flag = True
    elif "Meta" in args.reviewer_file:
        role = "SPC"

    timestr = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename='logs/acl-scores-logger-' + track_name + "-" + role + "-" + args.reviewer_file.replace("exports/", "") + "-" + timestr + '.log',level=logging.DEBUG, format='%(asctime)s %(message)s')

    logging.info("Track: " + track_name)
    logging.info("Role: " + role)
    logging.info("File: " + args.reviewer_file)

    start = time.time()
    submission_ids, submission_abs = parseSubmissions(args.submission_file, track_name)
    logging.info("No. of Submissions: " + str(len(submission_ids)))

    infered_ss_ids_map = get_infered_ss_ids(args.merged_user_info_file)
    reviewer_data, reviewer_names = parseReviewers(args.reviewer_file, args.user_info_file, infered_ss_ids_map, ac=ac_flag)
    logging.info("No. of Reviewers: " + str(len(reviewer_data)))
    
    # read the whole db file
    with open(args.db_file, "r") as f:
        db = [json.loads(x) for x in f]  # for debug
        db_abs = [x['paperAbstract'] for x in db]
    end = time.time()
    log_time_taken("read data files", start, end)

    start = time.time()
    names_to_ignore = getNamesToIgnore(args.user_info_file, args.reviewer_file, ac=ac_flag)
    rdb, rdb_new = calc_reviewer_db_mapping(reviewer_data, db, names_to_ignore, author_field='authors')
    # filter papers authored by reviewers
    # includes_reviewer is a array of length equal to len(db). 
    # The array element is greater than 1, if there is atleast one reviewer is a author of the paper
    includes_reviewer = rdb.sum(axis=1)
    new_db = []
    for i, paper in enumerate(db):
        if includes_reviewer[i] >= 1:
            new_db.append(paper)
    db = new_db
    db_abs = [x['paperAbstract'] for x in db]
    
    # read abstracts extracted from user profiles
    email_to_id_mapping = create_email_to_id_mapping(reviewer_data)
    abstract_id_pairs, unique_reviewers = create_abstracts_to_id_pairs(args.abstracts_file, email_to_id_mapping)
    logging.info("No of unique reviewers for whom abstracts were found: " + str(len(unique_reviewers)))

    # adding the extracted abstracts of reviewer-papers (from User Information.xls) to reduced db
    rdb, rdb_new = calc_reviewer_db_mapping(reviewer_data, db, names_to_ignore, author_field='authors',  print_warnings=True, abstract_id_pairs=abstract_id_pairs)
    if abstract_id_pairs != None:
        for (abstract, reviewer_id) in abstract_id_pairs:
            db_abs.append(abstract)
    end = time.time()
    log_time_taken("identify relevant abstracts from s2 db", start, end)

    # Calculate or load paper similarity matrix
    start = time.time()
    if args.load_paper_matrix:
        mat = np.load(args.load_paper_matrix)
        assert(mat.shape[0] == len(submission_abs) and mat.shape[1] == len(db_abs))
    else:
        print('Loading model', file=sys.stderr)
        model, epoch = load_model(None, args.model_file, force_cpu=True)
        model.eval()
        assert not model.training
        mat = calc_similarity_matrix(model, db_abs, submission_abs)
        mat = np.clip(mat, a_min = 0, a_max = 2) 
        if args.save_paper_matrix:
            np.save(args.save_paper_matrix, mat)
    end = time.time()
    log_time_taken("compute similarity matrix", start, end)
    
    start = time.time()
    # Calculate reviewer scores based on paper similarity scores
    if args.load_aggregate_matrix:
        reviewer_scores = np.load(args.load_aggregate_matrix)
        assert(reviewer_scores.shape[0] == len(submission_abs) and reviewer_scores.shape[1] == len(reviewer_names))
    else:
        print('Calculating aggregate reviewer scores', file=sys.stderr)
        reviewer_scores = calc_aggregate_reviewer_score(rdb, rdb_new, mat, args.aggregator)
        if args.save_aggregate_matrix:
            np.save(args.save_aggregate_matrix, reviewer_scores)
    end = time.time()
    log_time_taken("compute aggregate scores", start, end)

    # compute percentile
    #start = time.time()
    #reviewer_scores = convert_to_percentile(reviewer_scores)
    #end = time.time()
    #log_time_taken("compute percentile scores", start, end)
    
    start = time.time()

    ## COI is handled by the ILP input generator
    
    # use coi matrix as a mask
    # load cois generated by module A1
    #coi_matrix = load_coi_matrix(args.coi_file, email_to_id_mapping, submission_ids)

    # print to a flat file to be sent to CMT
    outfilename = "output/acl-scores-output-" + track_name + "-" + role + ".txt"
    with open(outfilename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        for i in range (len(reviewer_scores)):
            for j in range(len(reviewer_scores[i])):
                # ignore the entries with COI set to 1
                #if coi_matrix[i][j] == 0:
                csvwriter.writerow([submission_ids[i], reviewer_data[j]['email'], reviewer_scores[i][j]])
    end = time.time()
    log_time_taken("write to file", start, end)