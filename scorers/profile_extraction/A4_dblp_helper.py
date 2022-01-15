from  collections import defaultdict, OrderedDict
import csv
import gzip
import os
import sys
import pickle
import xmltodict

# for extracting keys from dpbl websites
import urllib.request
import re

# Consider pubs in this range only.
STARTYEAR = 1970
ENDYEAR = 2269

name_to_ids = defaultdict(list)
totalPapers = 0 # for statistics reporting purposes only
authlogs = defaultdict(list)
coauthors = {}
papersWritten = {}
paper_counter = 0
person_counter = 0
confdict = {}
author_info_map = {}

bucket_by_keyword_area = {
        "Domain(s) of Application" : "General",
        "Cognitive Modeling & Cognitive Systems" : "COG",
        "Constraint Satisfaction and Optimization" : "CSP",
        "Game Theory and Economic Paradigms" : "GT",
        "Humans and AI" : "HAC",
        "Human-Computation and Crowd Sourcing" : "HC&CS",
        "Search and Optimization" : "CSP",
        "Knowledge Representation and Reasoning" : "KRR",
        "Multiagent Systems" : "GT",
        "Machine Learning" : "ML",
        "Speech & Natural Language Processing" : "NLP",
        "Planning, Routing, Scheduling" : "P&S",
        "Robotics" : "Rob",
        "Reasoning under Uncertainty" : "UAI",
        "Computer Vision" : "CV",
        "Data Mining & Knowledge Management" : "DM",
        "Philosophy and Ethics of AI" : "Ethics",
        "Focus Area: AI for Conference Organization and Delivery" : "Focus",
        "Focus Area: AI Responses to the COVID-19 Pandemic (Covid19)" : "Focus",
        "Focus Area: AI for Conference Organization and Delivery (AICOD)": "Focus"
        }

bucket_by_conference = {
        'nips': "ML",
        'acl' : "NLP",
        'sigir': "DM",
        'www': "DM",
        'rss': "Rob",
        'naacl' : "NLP",
        'kdd' : "ML",
        'iros': "Rob",
        'icra': "Rob",
        'icml': "ML",
        'iccv': "CV",
        'emnlp': "NLP",
        'cvpr': "CV",
        'hcomp' : "HC&CS",
        'hri' : "HAC",
        'icaps' : "P&S",
        'icdm' : "DM",
        'iclr' : "ML",
        'kr' : "KRR", 
        'sat' : "KRR",
        'wsdm' : "DM",
        'uai' : "UAI",
        'colt' : "ML",
        'corl' : "Rob",
        'cp' : "CSP", 
        'cpaior' : "CSP",
        'ecml' : "ML",
        'ismar' : "CV",
        'jmlr' : "ML",
        'miccai' : "ML",
        'tacl' : "NLP",
        'coling' : "NLP",
        'icassp' : "NLP",
        'iswc' : "DM",
        'icde' : "DM",
        'cikm' : "DM",
        'pure DB' : "DM",
        # round 2
        'conll': "NLP",
        'eacl': "NLP",
        'interspeech' : "NLP",
        'sdm' : "DM",
        'ecir' : "DM",
        'wise' : "DM",
        'kcap' : "DM",
        'pacdd': "DM",
        'pkdd' : "ML", 
        'ijcnn' : "ML",
        'eccv' : "CV",
        'wacv' : "CV"
        }

# for other other conference aliases, please refer to "areadict" variable in line 70 of the script below
# https://github.com/emeryberger/CSrankings/blob/gh-pages/util/csrankings.py
conf_to_aliases = {
    'aaai': ['AAAI', 'AAAI/IAAI'],
    'ijcai': ['IJCAI'],
    'nips': ['NIPS', 'NeurIPS'],
    'acl' : ['ACL', 'ACL (1)', 'ACL (2)', 'ACL/IJCNLP', 'COLING-ACL'],
    'sigir': ['SIGIR'],
    'www': ['WWW'],
    'rss': ['Robotics - Science and Systems', 'Robotics: Science and Systems'],
    'naacl' : ['NAACL', 'HLT-NAACL', 'NAACL-HLT', 'NAACL-HLT (1)'],
    'kdd' : ['KDD'],
    'iros': ['IROS'],
    'icra': ['ICRA', 'ICRA (1)', 'ICRA (2)'],
    'icml': ['ICML', 'ICML (1)', 'ICML (2)', 'ICML (3)'],
    'iccv': ['ICCV'],
    'emnlp': ['EMNLP', 'EMNLP-CoNLL', 'HLT/EMNLP', 'EMNLP-IJCNLP', 'EMNLP/IJCNLP (1)'],
    'ec' : ['EC'],
    'cvpr': ['CVPR', 'CVPR (1)', 'CVPR (2)'],
    'aamas': ['AAMAS'],
    'hcomp' : ["HCOMP"],
    'hri' : ["HRI"],
    'icaps' : ["ICAPS"],
    'icdm' : ["ICDM", "Industrial Conference on Data Mining"],
    'iclr' : ["ICLR"],
    'icwsm' : ["ICWSM"], 
    'iui' : ["IUI"],
    'kr' : ["KR"], 
    'sat' : ["SAT"], 
    'wsdm' : ["WSDM"],
    'uai' : ["UAI"],
    'aistats' : ["AISTATS"], 
    'colt' : ["COLT", "COLT/EuroCOLT", "EuroCOLT"], 
    'corl' : ["CoRL"], 
    'cp' : ["CP"], 
    'cpaior' : ["CPAIOR"],
    'ecai' : ["ECAI"], 
    'ecml' : ["ECML", "ECML/PKDD (1)", "ECML/PKDD (2)", "ECML/PKDD (3)"],
    'cscw' : ["CSCW Companion", "CSCW"],
    'chi' : ["CHI"],
    'uist' : ["UIST"],
    'ismar' : ["ISMAR"],
    'jmlr' : ["J. Mach. Learn. Res."],
    'miccai' : ["MICCAI (1)", "MICCAI (2)", "MICCAI (3)", "MICCAI (4)", "MICCAI (5)", "MICCAI (6)", "MICCAI"],
    'tacl' : ["Trans. Assoc. Comput. Linguistics"],
    'coling' : ["COLING"],
    'icassp' : ["ICASSP"],
    'iswc' : ["ISWC (1)", "ISWC (2)", "International Semantic Web Conference (1)", "International Semantic Web Conference (2)", "International Semantic Web Conference", "ISWC"],
    'icde' : ["ICDE"],
    'cikm' : ["CIKM"],
    'pure DB (VLDB, SIGMOD, PODS)' : ["VLDB J.", "SIGMOD Conference", "PODS"],
    'acm multimedia' : ["ACM Multimedia"],
    'cav' : ["CAV (1)", "CAV (2)", "CAV"],
    'genetic (FOGA, GECCO)' : ["FOGA", "GECCO"],
    'ijcar' : ["IJCAR (1)", "IJCAR (2)", "IJCAR"],
    'ubicomp' : ["UbiComp"],
    'aied' : ["AIED", "AIED (1)", "AIED (2)", "AIED"],
    'cogsci' : ["CogSci"],
    'ethics' : ["AIES", "FAT*", "Ethics Inf. Technol.", "Minds Mach."],
    # round 2
    'conll': ['CoNLL'],
    'eacl': ['EACL'],
    'interspeech' : ['INTERSPEECH', 'EUROSPEECH', 'ICSLP'],
    'sdm' : ["SDM"],
    'ecir' : ["ECIR (1)", "ECIR (2)", "ECIR"],
    'wise' : ["WISE", "WISE (1)", "WISE (2)"],
    'kcap' : ["K-CAP"],
    'pacdd': ["PAKDD (1)", "PAKDD (2)", "PAKDD"],
    'pkdd' : ["PKDD"], # ECML/PKDD covered by ECML
    'ijcnn' : ['IJCNN', 'IJCNN (1)', 'IJCNN (2)', 'IJCNN (3)', 'IJCNN (4)', 'IJCNN (5)', 'IJCNN (6)'],
    'eccv' : [ "ECCV (1)", "ECCV (2)", "ECCV (3)", "ECCV (4)", "ECCV (5)",
            "ECCV (6)", "ECCV (7)", "ECCV (8)", "ECCV (9)", "ECCV (10)",
            "ECCV (11)", "ECCV (12)", "ECCV (13)", "ECCV (14)", "ECCV (15)",
            "ECCV (16)", "ECCV"],
    'wacv' : ["WACV"]
}

confdict = {}
for conf, aliases in conf_to_aliases.items():
    for alias in aliases:
        confdict[alias] = conf


def parse_xml():
    """ parses dblp.xml.gz and extracts information on persons and conferences.
        
        We store all person information but only information on conferences
        listed in conf_to_aliases.
    """

    if not os.path.exists('dblp.xml.gz'):
        print("\nPlease download the latest file from download dblp.xml.gz file from https://dblp.uni-trier.de/xml/\n")
        exit(0)

    gz = gzip.GzipFile('dblp.xml.gz')
    xmltodict.parse(gz, item_depth=2, item_callback=handle_entry)
    print("Serializing parsed data")
    with open('parsed_data.pkl', 'wb') as f:
        pickle.dump([authlogs, name_to_ids, author_info_map], f)


def handle_entry(path, entry):
    if path[1][0] == 'www':
        return handle_person(path, entry)
    else:
        return handle_article(path, entry)


def handle_person(path, entry):
    global name_to_ids
    global person_counter
    global author_info_map
    
    person_counter += 1
    if person_counter % 1000 == 0:
        print(str(person_counter)+ " persons processed.")

    # print(path, entry)
    key = path[1][1].get("key","")
    if not key.startswith('homepages'):
        return True

    author_info = {}
    key = key.replace("homepages/", "")
    author_info['key'] = key

    # collect author aliases
    author_aliases = []
    if 'author' not in entry: # typically a crossref
        return True
    assert 'author' in entry
    if type(entry['author']) != list:
        entry['author'] = [entry['author']]
    author_info['aliases'] = []
    for item in entry['author']:
        if isinstance(item, OrderedDict):
            author_info['aliases'].append(item["#text"].encode("utf-8"))
        else:
            author_info['aliases'].append(item.encode("utf-8"))

    for alias in author_info['aliases']:
        name_to_ids[alias].append(key)
    
    # collect urls 
    author_info['urls'] = []
    if "url" in entry:
        if type(entry['url']) != list:
            entry['url'] = [entry['url']]
        author_info["urls"] = entry["url"]

    author_info_map[key] = author_info
    return True


def handle_article(_, article):
    global totalPapers
    global confdict
    global paper_counter
    global authlogs
    paper_counter += 1
    
    # extract some data, return True if entry not relevant
    try:
        if paper_counter % 100000 == 0:
            print(str(paper_counter)+ " papers processed.")
        if 'author' in article:
            # Fix if there is just one author.
            if type(article['author']) != list:
                article['author'] = [article['author']]
            authorList = article['author']
            authorsOnPaper = len(authorList)

        else:
            return True
        if 'booktitle' in article:
            confname = article['booktitle']
        elif 'journal' in article:
            confname = article['journal']
        else:
            return True

        volume = article.get('volume',"0")
        number = article.get('number',"0")
        url    = article.get('url',"")
        year   = int(article.get('year',"-1"))
        pages  = ""
        
        if confname not in confdict:
            return True
        
        if 'title' in article:
            title = article['title']
            if type(title) is OrderedDict:
                title = title["#text"]
    except TypeError:
        raise
    except:
        print(sys.exc_info()[0])
        raise

    if countPaper(confname, year, volume, number, url, title):
        totalPapers += 1
        for authorName in authorList:
            if type(authorName) is OrderedDict:
                authorName = authorName["#text"]
            authorName = authorName.encode("utf-8")
            log = { 'name' : authorName,
                    'year' : year,
#                    'title' : title.encode('utf-8'),
                    'conf' : confname,
#                    'area' : areaname,
                    'numauthors' : authorsOnPaper }
            authlogs[authorName].append(log)
    return True


def countPaper(confname, year, volume, number, url, title):
    """Returns true iff this paper will be included in the rankings."""
    if year < STARTYEAR or year > ENDYEAR:
        return False
    return True


def parse_nominees(csv_filename, dblp_column, email_column, read_pickle=True):
    nominees_by_email = dict()

    # we use information from the pickle file to avoid repeated crawls of the
    # dblp websites, from which we extract the key
    if read_pickle and os.path.isfile('nominees.pkl'):
        with open('nominees.pkl', 'rb') as f:
            nominees_by_email = pickle.load(f)

    with open(csv_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', quotechar='"')
        for nominee in reader:
            email = nominee[email_column]
            if email not in nominees_by_email:
                "not yet included"
                try:
                    nominee["dblp_key"] = extract_dblp_key(nominee[dblp_column])
                except:
                    print("no valid dblp key?", nominee[email_column],
                          nominee[dblp_column])
                    continue
                if "area" in nominee:
                    nominee["area"] = nominee["area"].replace(";", ",")
                nominees_by_email[email] = nominee
    with open('nominees.pkl', 'wb') as f:
        pickle.dump(nominees_by_email, f)
    return nominees_by_email


def extract_dblp_key(dblp_url):
    if dblp_url == "":
        raise Exception
    m = re.search(".*dblp.*", dblp_url)
    if m and not dblp_url.endswith(".html"):
        dblp_url += ".html"
    resource = urllib.request.urlopen(dblp_url)
    content =  resource.read().decode(resource.headers.get_content_charset())
    m = re.search("<!-- head line --.*dblp key:.*homepages/(.*?)</small>", content)
    return m.group(1)

def write_publications_as_csv(nominees_by_email, author_info_map, authlogs,
        dblp_column, filename):
    print("Populating fields")
    conferences = conf_to_aliases.keys()
    data_to_write = []

    some_nominee = next(iter(nominees_by_email.values()))
    header = list(some_nominee.keys())
    header.extend(["related 10y normalized", "related 10", "related total"])
    header.extend(conferences)
    for email, nominee in nominees_by_email.items():
        dblp_key = nominee["dblp_key"]
        if dblp_key not in author_info_map:
            print("WARNING: no information on", nominee["First Name"],
                    nominee["Last Name"])
            print("DBLP:", nominee[dblp_column])
            print("DBLP key:", nominee["dblp_key"])
            continue
        aliases = author_info_map[dblp_key]["aliases"]
        publications_per_conference = defaultdict(int)
        publications_per_conference_past5 = defaultdict(int)
        area_buckets_past5 = defaultdict(int)
        no_related_publications_past10_normalized = 0
        no_related_publications_total = 0
        no_related_publications_past5 = 0
        no_related_publications_past10 = 0
        for alias in aliases: 
            for pub in authlogs[alias]:
                area = confdict[pub["conf"]]
                publications_per_conference[area] += 1
                no_related_publications_total += 1
                if 2020-int(pub["year"]) <= 5:
                    no_related_publications_past5 += 1
                    publications_per_conference_past5[area] += 1
                    if confdict[pub["conf"]] in bucket_by_conference:
                        area_buckets_past5[bucket_by_conference[area]] += 1
                if 2020-int(pub["year"]) <= 10:
                    no_related_publications_past10 += 1
                    no_related_publications_past10_normalized += 1.0 / pub["numauthors"]

        
        entry = list(nominee.values())
        entry.append(no_related_publications_past10_normalized)
        entry.append(no_related_publications_past10)
        entry.append(no_related_publications_total)
        for conference in conferences:
            entry.append(str(publications_per_conference[conference]))
        data_to_write.append(entry)
    with open(filename,'w') as f:
        mywriter = csv.writer(f, delimiter='\t', quotechar='"')
        mywriter.writerow(header)
        for data_row in data_to_write:
            mywriter.writerow(data_row)

def write_publications_for_nominees_as_csv(nominees_by_email, author_info_map, authlogs):
    print("Populating fields")
    conferences = conf_to_aliases.keys()
    data_to_write = []
    header = ["firstname", "familyname", "email", "affiliation", "area",
            "comment", "bucket", "related 10y normalized", "related 10", "related total"]
    header.extend(conferences)
    for email, nominee in nominees_by_email.items():
        dblp_key = nominee["dblp_key"]
        if dblp_key not in author_info_map:
            print("WARNING: no information on", nominee["firstname"],
                    nominee["familyname"])
            print("DBLP:", nominee["dblp"])
            print("DBLP key:", nominee["dblp_key"])
            continue
        aliases = author_info_map[dblp_key]["aliases"]
        publications_per_conference = defaultdict(int)
        publications_per_conference_past5 = defaultdict(int)
        area_buckets_past5 = defaultdict(int)
        no_related_publications_past10_normalized = 0
        no_related_publications_total = 0
        no_related_publications_past5 = 0
        no_related_publications_past10 = 0
        for alias in aliases: 
            for pub in authlogs[alias]:
                area = confdict[pub["conf"]]
                publications_per_conference[area] += 1
                no_related_publications_total += 1
                if 2020-int(pub["year"]) <= 5:
                    no_related_publications_past5 += 1
                    publications_per_conference_past5[area] += 1
                    if confdict[pub["conf"]] in bucket_by_conference:
                        area_buckets_past5[bucket_by_conference[area]] += 1
                if 2020-int(pub["year"]) <= 10:
                    no_related_publications_past10 += 1
                    no_related_publications_past10_normalized += 1.0 / pub["numauthors"]

        
        # determine buckets
        buckets = set()
        if nominee["area"] in bucket_by_keyword_area:
            buckets.add(bucket_by_keyword_area[nominee["area"]])
        for area, no in area_buckets_past5.items():
            if no > 8 or (no > 3 and float(no)/no_related_publications_past5 > 0.25):
                buckets.add(area)
        
        entry = [nominee["firstname"], nominee["familyname"], email,
                 nominee["affiliation"], nominee["area"], nominee["comment"],
                 ",".join(buckets),
                 no_related_publications_past10_normalized,
                 no_related_publications_past10, no_related_publications_total]
        for conference in conferences:
            entry.append(str(publications_per_conference[conference]))
        data_to_write.append(entry)
    with open('SPC-nominations.csv','w') as f:
        mywriter = csv.writer(f, delimiter='\t', quotechar='"')
        mywriter.writerow(header)
        for data_row in data_to_write:
            mywriter.writerow(data_row)

        
if __name__ == "__main__":


    # this script collects all papers for all authors from a fixed set of
    # conferences (cf. conf_to_aliases map above) and extracts for every author
    # some information on their publications numbers.
    #
    # If there is a new dblp dump, delete file parsed_data.pkl, which caches
    # extracted dblp information.
    #
    # We also cache information on authors in nominees.pkl and skip new
    # information if we already have some for the same email address. Delete
    # this if existing information changed.
    #
    # Requires dblp.xml.gz, 

    if not os.path.isfile('parsed_data.pkl'):
        parse_xml()
    with open('parsed_data.pkl', 'rb') as f:
        (authlogs, #_,
        name_to_ids, author_info_map) = pickle.load(f)

    # call this in a loop
    dblp_key = "72/11205"

    # name just for logging
    # please log this
    first_name = "Dinesh"
    last_name = "Raghu"

    if dblp_key not in author_info_map:
        print("WARNING: no information on", first_name,
                last_name)
    else:
        aliases = author_info_map[dblp_key]["aliases"]
        no_related_publications_total = 0
        no_related_publications_past10 = 0
        for alias in aliases: 
            for pub in authlogs[alias]:
                no_related_publications_total += 1
                if 2020-int(pub["year"]) <= 10:
                    no_related_publications_past10 += 1
        print("No of conf related pubs         (all):", no_related_publications_total)
        print("No of conf related pubs (last 10 yrs):", no_related_publications_past10)

#get dblp info
def get_dblp_info(info,logger):
    if not os.path.isfile('parsed_data.pkl'):
        parse_xml()
    with open('parsed_data.pkl', 'rb') as f:
        (authlogs, #_,
        name_to_ids, author_info_map) = pickle.load(f)

    info['DBLP ID'] = info.apply(lambda row: row['DBLP URL'].strip().split("pid/")[-1].split('.html')[0] if type(row['DBLP URL'])!=float else "NA", axis=1)
    def get_n_papers(row):
        # call this in a loop
        dblp_key = row['DBLP ID']

        # name just for logging
        # please log this
        first_name = row['First Name']
        last_name = row['Last Name']

        if dblp_key not in author_info_map:
            logger.info("No information on " + first_name + " " + last_name + " " + row['Email'])
            return 0
        else:
            aliases = author_info_map[dblp_key]["aliases"]
            no_related_publications_total = 0
            for alias in aliases: 
                no_related_publications_total += len(authlogs[alias])
        return no_related_publications_total
    info['n_DBLP'] = info.apply(get_n_papers,axis=1)
    return info