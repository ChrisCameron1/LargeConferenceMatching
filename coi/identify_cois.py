import argparse

import gzip
import collections
import xmltodict
import os
import csv
import copy
import pprint
import pickle
from  collections import defaultdict
from itertools import combinations
import pandas as pd
# for extracting key from dpbl websites
import urllib.request
import re
from lxml import etree as ElementTree
from IPython.core.debugger import Pdb
# Consider pubs in this range only.
from CONSTANTS import *

#PERSON_DATA_FILE =  'data/dblp/dblp_person_data.pkl'
#CONFLICT_DATA_FILE = 'data/dblp/dblp_conflicts.pkl'

paper_counter = 0
person_counter = 0
name_to_ids = defaultdict(list)
author_info_map = {}
conflicts = defaultdict(set)


def parse_xml(in_file, handle):
    """ parses dblp.xml.gz and extracts information on persons and conferences
    (depending on handle)."""

    #if not os.path.exists('dblp.xml.gz'):
    if not os.path.exists(in_file):
        print("\nPlease download the latest file from download dblp.xml and dblp.dtd file from https://dblp.uni-trier.de/xml/\n")
        exit(0)

    #gz = gzip.GzipFile('dblp.xml.gz')
    with open(in_file,'rb') as fh: 
        xmltodict.parse(fh, item_depth=2, item_callback=handle)


def handle_entry(path, entry):
    if path[1][0] == 'www':
        return handle_person(path, entry)
    else:
        return handle_article(path, entry)


def handle_person(path, entry):
    if path[1][0] != 'www': # not a person entry
        return True
    global name_to_ids
    global person_counter
    global conflicts
    

    key = path[1][1].get("key","")
    if not key.startswith('homepages'):
        return True

    author_info = {}
    key = key.replace("homepages/", "")
    author_info['key'] = key

    # collect author aliases
    author_aliases = []
    if 'author' not in entry: # a crossref
        return True
    
    person_counter += 1
    if person_counter % 1000 == 0:
        print(str(person_counter)+ " persons processed.")
    
    assert 'author' in entry
    if type(entry['author']) != list:
        entry['author'] = [entry['author']]
    author_info['aliases'] = []
    for item in entry['author']:
        if isinstance(item, collections.OrderedDict):
            author_info['aliases'].append(item["#text"].encode("utf-8"))
        else:
            author_info['aliases'].append(item.encode("utf-8"))

    for alias in author_info['aliases']:
        name_to_ids[alias].append(key)

    # collect urls 
    author_info['urls'] = []
    if "url" in entry:
        author_info["urls"] = list(entry["url"])

    author_info_map[key] = author_info
    return True


def handle_article(path, article):
    if path[1][0] == 'www':
        return True # not an article
    global name_to_ids
    global author_info_map
    global paper_counter
    paper_counter += 1
    
    # extract some data, return True if entry not relevant
    try:
        if ('booktitle' not in article and
            'journal' not in article):
            return True
        year = int(article.get('year',"-1"))
        if year < STARTYEAR or year > ENDYEAR:
            return True
        if paper_counter % 100000 == 0:
            print(str(paper_counter)+ " relevant papers processed.")
        if 'author' in article:
            # Fix if there is just one author.
            if type(article['author']) != list:
                article['author'] = [article['author']]
            author_list = article['author']
        else:
            return True
    except TypeError:
        raise
    except:
        print(sys.exc_info()[0])
        raise

    Pdb().set_trace()
    for index, author_name in enumerate(author_list):
        if type(author_name) is collections.OrderedDict:
            author_name = author_name["#text"]
        author_name = author_name.encode("utf-8")
        author_list[index] = author_name
    for author1, author2 in combinations(author_list, 2):
        # we could be cheaper with an explicit nested loop
        for id1 in name_to_ids[author1]:
            for id2 in name_to_ids[author2]:
                conflicts[id1].add(id2)
                conflicts[id2].add(id1)

    return True



def parse_persons(csv_filename):
    person_by_email = dict()

    # we use information from the pickle file to avoid repeated crawls of the
    # dblp websites, from which we extract the persistent key if we only have
    # the name key
    if os.path.isfile('persons.pkl'):
        with open('persons.pkl', 'rb') as f:
            person_by_email = pickle.load(f)

    with open(csv_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', quotechar='"')
        for person in reader:
            email = person['Wmail']
            if email not in person_by_email:
                try:
                    person["dblp_key"] = extract_dblp_key(person["DBLP URL"])
                except:
                    print("no valid dblp key?")
                    print(person)
                    continue
                person_by_email[email] = person 
                # we currently store everything about the person; if memory
                # becomes an issue, we can be more restrictive
    with open('persons.pkl', 'wb') as f:
        pickle.dump(person_by_email, f)
    return person_by_email


def extract_dblp_key(dblp_url):
    dblp_url = dblp_url.strip().lower()
    if dblp_url.startswith("https://dblp.org/pid/"):
        return dblp_url[21:]

    # not a persistent key but the name-based one. Extract persistent one from
    # DBLP
    resource = urllib.request.urlopen(dblp_url)
    content =  resource.read().decode(resource.headers.get_content_charset())
    m = re.search("<!-- head line --.*dblp key:.*homepages/(.*?)</small>", content)
    return m.group(1)


def set_parser():
    #XML_FILE = 'data/dblp/dblp.xml'
    #DTD_FILE = 'data/dblp/dblp.dtd'
    #PERSON_DATA_FILE =  'data/dblp/dblp_person_data.pkl'
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file',type=str,default = 'data/dblp/dblp.xml')
    parser.add_argument('--out_file',type=str,default = 'data/dblp/dblp_person_data.pkl')
    return parser




def get_args(arg_str=None):
    parser = set_parser()
    if arg_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(arg_str.split())
    #   
    print(args)

    return args



if __name__ == "__main__":
    args = get_args()
    # extract ALL co-authorship conflicts from DBLP
    parse_xml(args.in_file, handle_person)
    print("Serializing parsed data")
    # dump to pickle file
    with open(args.out_file, 'wb') as f:
        pickle.dump([name_to_ids, author_info_map], f)

    """
    if not os.path.isfile(CONFLICT_DATA_FILE):
        parse_xml(handle_article)
        print("Serializing parsed data")
        # dump to pickle file
        with open(CONFLICT_DATA_FILE, 'wb') as f:
            pickle.dump(conflicts, f)
    else:
        with open(CONFLICT_DATA_FILE, 'rb') as f:
            conflicts = pickle.load(f)
    """
    # parse all persons (authors, SPC, PC, area chairs)
    # Store as dict by email address (it's no problem if the same person is
    # registered several times). We need to extract the dblp key. If we don't
    # have the persistent key, we extract it from dblp.
#    person_by_email = parse_persons('persons.csv')
