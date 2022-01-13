import json
import argparse
#from sacremoses import MosesTokenizer
#import suggest_utils

parser = argparse.ArgumentParser()

parser.add_argument("--infile", help="unfiltered json file: high recall but low precision")
parser.add_argument("--outfile", help="filtered json file: high precision")

args = parser.parse_args()

with open(args.infile, "r") as f:
    data = [json.loads(x) for x in f]

#entok = MosesTokenizer(lang='en')

round_1_conf = ["AIES","AIED","CAV","CHI","CIKM","CogSci","COLING","CSCW","Ethics and Information Technology","FATML","FAT*","FOGA","GECCO","ICASSP","ICDE","IJCAR","ISMAR","ISWC","J. Mach. Learn. Res.","MICCAI","Minds and Machines","PODS","SIGMOD","TACL","Transactions of the Association for Computational Linguistics","UbiComp","UIST","VLDB"]
round_2_conf = ["CoNLL","CV","EACL","ECCV","ECIR","IJCNN","INTERSPEECH","K-CAP","PAKDD","SDM","WACV","WISE"]
sister_conf_excluded = ["CP","EC","HRI","KR","RSS","SAT","Machine Learning", "MM", "FAT"]
sister_conferences = ["AAAI","AAMAS","ACL","AISTATS","COLT","CoRL","CPAIOR","CVPR","ECAI","ECML","EMNLP","HCOMP","ICAPS","ICCV","ICDM","ICLR","ICML","ICRA","ICWSM","IJCAI","International Conference on Human-Robot Interaction","IROS","IUI","KDD","NAACL","NeurIPS","NIPS","Robotics: Science and Systems","SIGIR","WWW","WSDM","UAI"]

all_details = {}

conf_maps = {
    "sister_conferences":sister_conferences,
    "sister_conf_excluded":sister_conf_excluded,
    "round_1_conf": round_1_conf,
    "round_2_conf": round_2_conf
}

def print_all_details():
    for conf_list_name, conf_list in conf_maps.items():
        for conf in conf_list:
            if conf in all_details[conf_list_name]:
                print(conf_list_name, ":", conf, "(" + str(all_details[conf_list_name][conf]['TOTAL']) + ")")
            
                for venue, count in all_details[conf_list_name][conf].items():
                    if venue == 'TOTAL':
                        continue
                    print("\t", ' '*(4-len(str(count)))+str(count), "\t", venue)
            else:
                print(conf_list_name, ":", conf)
                print("\t FIX THIS: HAS ZERO PAPERS")
            print("")

def is_in_conf_list(venue, conf_list):
    for conf in conf_list:

        if conf==venue or venue.startswith(conf+" ") or ((conf=="ICDE" or conf == "ISMAR" or conf == "ECML" or conf == "WACV" or conf == "IJCNN" or conf == "IROS" or conf=="ICCV" or conf=="ICDM" or conf=="ICRA" or conf=="NAACL" or conf=="IJCNN") and conf in venue):

            if "ICDEc" in venue or "ICDEW" in venue or "ICDECS" in venue or "ICDEL" in venue or "ICCVW" in venue or "ICCVG" in venue or "ICCVBIC" in venue or "ICCVE" in venue or "ISICDM" in venue or "ICDMW" in venue or "ICDMML" in venue or "ICDMAI" in venue or "ICRAE" in venue or "ICRAMET" in venue or "ICRAS" in venue or "ICRAI" in venue or "ICRAMET" in venue or "ICRAAE" in venue or "WACVW" in venue or "pharmacology" in venue or "anaesthesia" in venue or "Machine Learning and Knowledge Extraction" in venue or "Machine Learning Paradigms" in venue:
                continue
            return conf
    return None

def is_related_conf(venue):

    for conf_list_name, conf_list in conf_maps.items():
        conf = is_in_conf_list(venue, conf_list)
        if conf != None:
            if conf_list_name not in all_details:
                all_details[conf_list_name] = {}
                all_details[conf_list_name]["TOTAL"] = 0
            all_details[conf_list_name]["TOTAL"] += 1

            if conf not in all_details[conf_list_name]:
                all_details[conf_list_name][conf] = {}
                all_details[conf_list_name][conf]["TOTAL"] = 0
            all_details[conf_list_name][conf]["TOTAL"] += 1
            
            if venue not in all_details[conf_list_name][conf]:
                all_details[conf_list_name][conf][venue] = 0
            all_details[conf_list_name][conf][venue] += 1

            return True
    return False

unique_venues = set([])
year_map = {}
venue_map = {}
field_map = {}

count = 0
abstracts = []
outfile = open(args.outfile, 'w')
for i in data:
    abstracts.append(i['paperAbstract'])

    year = None
    if 'year' in i:
        year = i['year']
    
    fields = []
    if 'fieldsOfStudy' in i:
        fields = i['fieldsOfStudy']

    venue = ""
    if 'venue' in i:
        venue = i['venue']

    if 'Computer Science' in fields and (year != None and year > 2015) and is_related_conf(venue):
        outfile.write(json.dumps(i) + "\n")
        unique_venues.add(venue)
        if venue not in venue_map:
            venue_map[venue] = 0
        venue_map[venue] += 1

        count += 1
        if year not in year_map:
            year_map[year] = 0
        year_map[year] += 1
        
        for field in fields:
            if field not in field_map:
                field_map[field] = 0
            field_map[field] += 1

outfile.close()

#print("\n-------------------\n")
#[ print(key , " : " , value) for (key, value) in sorted(year_map.items()) ]
#print("\n-------------------\n")
#[ print(key , " : " , value) for (key, value) in sorted(field_map.items()) ]
#print("\n-------------------\n")

print_all_details()

print("TOTAL", count)
