import csv
import sys
from pathlib import Path
import pandas as pd


def extract_triplets_from_line(line):
    triplets = []
    parts = line.strip().split(' | ')
    if len(parts) < 2:
        return triplets  
    for i in range(0, len(parts)):
        triplet_text = parts[len(parts) - 1 - i]
        triplet_parts = triplet_text.split(' ; ')
        if len(triplet_parts) == 3:
            head = triplet_parts[0].strip()
            relation = triplet_parts[2].strip()
            tail = triplet_parts[1].strip()
            triplets.append((head, relation, tail))
        else:
            break
    return triplets 

def main():
    input_filenames = ["train.txt", "test.txt", "dev.txt"]
    output_filename = "tripletas_FinRED.tsv"
    output_file = Path(output_filename)
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(outfile)
        for input_filename in input_filenames:
            input_file = Path(input_filename)
            if not input_file.exists():
                sys.exit(1)    
            triplets_count = 0
            lines_processed = 0
            with open(input_file, 'r', encoding='utf-8') as infile:
                for line_num, line in enumerate(infile, 1):
                    line = line.strip()
                    if not line:  
                        continue
                    lines_processed += 1
                    triplets = extract_triplets_from_line(line)
                    for triplet in triplets:
                        csv_writer.writerow(triplet)
                        triplets_count += 1     
    pd.read_csv(output_filename).drop_duplicates().to_csv(output_filename,index=False, sep="\t")
    
    # #sanity check
    # train_df = pd.read_csv("tripletas_train.csv").drop_duplicates()
    # test_df = pd.read_csv("tripletas_test.csv").drop_duplicates()
    # eval_df = pd.read_csv("tripletas_dev.csv").drop_duplicates()
    
    # train_entities = set(pd.concat([train_df["head"],train_df["tail"]]).drop_duplicates().unique())
    # test_entities = set(pd.concat([test_df["head"],test_df["tail"]]).drop_duplicates().unique())
    # eval_entities = set(pd.concat([eval_df["head"],eval_df["tail"]]).drop_duplicates().unique())

    # assert(len(test_entities - train_entities) == 0)
    # assert(len(eval_entities - train_entities) == 0)
    
        
        

if __name__ == "__main__":
    main() 