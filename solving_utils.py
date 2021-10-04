import sys
import os
#sys.path.insert(0,'scripts')
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,'%s/scripts' % dir_path)
from dataclasses import dataclass
import pandas as pd
import numpy as np
import yaml
import pickle
from ilp_utils import read_pc_spc_table, read_ac_table
from ilp_CONSTANTS import *

@dataclass
class ExtraInfo:
    reviewer_info: pd.DataFrame
    agg_score_matrix: np.ndarray
    config: dict
    primary_area: pd.DataFrame
    mail2reviewers: dict
    pids_df: pd.DataFrame

def create_extra_info(config_file='ilp_config_20200922.yml'):
    with open(config_file, 'rb') as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)

    pids_df = pd.read_excel(config['SUBMISSIONS_FILE'])[['Paper ID', 'Track Name']].rename(columns={'Paper ID': 'paper'}).rename(columns={'Track Name': 'track'})

    with open(config['MAIL2REVIEWERS_FILE'], 'rb') as fh:
        mail2reviewers = pickle.load(fh)
    records = []
    for rtype,rfile_list in config['REVIEWERS_FILES'].items():
        role,track = rtype.split('_')
        role_id = role2id[role]
        track_id = track2id[track]
        if role_id == role2id['AC']:
            rtable = read_ac_table(rfile_list)
        else:
            rtable = read_pc_spc_table(rfile_list)

        for index,row in rtable.iterrows():
            rmail = row['email'].strip().lower()
            primary_area = row['primary_area']
            rid = mail2reviewers.get(rmail,[])
            record = dict(
                        rid=rid,
                        email=rmail,
                        primary_area=primary_area
                        )
            records.append(record)
    df = pd.DataFrame.from_records(records)

    reviewer_info = pd.read_csv(config['REVIEWER_ID_FILE'])
    # TODO: Code reuse from other scripts, should be a util function
    AGGSCORE_MATRIX_FILE = config['SCORE_MATRIX']
    if os.path.isfile(AGGSCORE_MATRIX_FILE):
        with open(AGGSCORE_MATRIX_FILE, 'rb') as fh:
            agg_score_matrix = pickle.load(fh)
    else:
        print(f'Warning: aggscore matrix file {AGGSCORE_MATRIX_FILE} doesnt exist')
        agg_score_matrix = None

    return ExtraInfo(
        reviewer_info=reviewer_info,
        agg_score_matrix=agg_score_matrix,
        config=config,
        primary_area=df,
        mail2reviewers=mail2reviewers,
        pids_df=pids_df
    )


def create_initial_per_reviewer_num(info, k):
    per_reviewer_num = info.reviewer_info.copy()[['rid', 'role', 'rmail']].set_index('rid')
    per_reviewer_num['window_start'] = 0
    role_2_multiplier = {
        'AC': 10,
        'SPC': 5,
        'PC': 1,
    }
    per_reviewer_num['window_end'] = per_reviewer_num['role'].map(role_2_multiplier) * k  - 1
    return per_reviewer_num