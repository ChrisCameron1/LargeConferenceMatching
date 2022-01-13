import csv

if __name__ == "__main__":

    input_file_prefix = "output/"

    tracks = [
                "AAAI2021",
                "AI for Social Impact"
             ]


    input_file_names = []
    for track in tracks:
        for role in ["AC", "SPC", "PC"]:
            input_file_names.append("acl-scores-output-"+track+"-"+role+".txt")
   
    out_csvfile_path = "output/acl_scores.csv"
    present_in_spc = set([])
    with open(out_csvfile_path, 'w') as out_csvfile:
        
        out_csvwriter = csv.writer(out_csvfile)
        
        for input_file_name in input_file_names:

            ## only use this in phase-2 
            # as many SPCs take additional roles to review papers
            # we should not allcoate new papers to them by considering them as reviewers
            # not sure if this is handled by the ILP
            if "SPC" in input_file_name:
                present_in_spc = set([])
            skipped_pcs = set([])

            print("Processing", input_file_name)
            input_filepath = input_file_prefix + input_file_name
            csv_file = open(input_filepath)
            read_csv = csv.reader(csv_file)
            for row in read_csv:
                paperid = row[0]
                rev_email = row[1]
                score = row[2]

                if "SPC" in input_file_name:
                    present_in_spc.add(rev_email)
                if "SPC" not in input_file_name and rev_email in present_in_spc:
                    skipped_pcs.add(rev_email)
                    continue

                out_csvwriter.writerow(row)

            if "SPC" not in input_file_name:
                for email in skipped_pcs:
                    print("Skipped " + email)
                    