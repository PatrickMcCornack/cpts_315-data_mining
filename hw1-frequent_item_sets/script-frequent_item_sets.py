"""
Author: Pat McCornack
Date: 9/27/2022

Adapted from base code provided in Panopto lectures.
"""

from apyori import apriori
import csv
import pandas as pd
import time
import numpy as np


def get_max_length(data_file):  # Returns the maximum length of a basket in a set of baskets
    index = 0
    temp = 0
    with open(data_file, 'r') as csvfile:  # Get maximum length --------- Need to trim down
        hw1reader = csv.reader(csvfile, delimiter=' ')
        for row in hw1reader:
            if index == 0:
                temp = len(row)
            elif len(row) > temp:
                temp = len(row)
            index = index + 1

    return temp


def comma_delimit(data_file, csv_file):  # Creates a csv from a file that's space delimited
    print("Formatting data file...")
    max_length = get_max_length(data_file)
    with open(data_file, 'r') as csvfile:
        index = 0
        new_file = ''
        hw1reader = csv.reader(csvfile, delimiter=' ')
        for row in hw1reader:
            new_str = ''
            index += 1

            for x in range(len(row)):
                if row[x] != '':
                    new_str += row[x]
                    new_str += ','

            if len(row) < max_length:  # Adds commas to create 'tidy' data to be converted to dataframe
                for x in range(len(row), max_length):
                    new_str += ','

            new_file = new_file + '\n' + new_str + '\n'

    with open("output", "w+") as f:  # Creates a new csv file
        f.write(new_file)


def run_apriori(data_file, output_file):
    print("Generating List...")
    start_time = time.time()  # Used to record run-time
    data = pd.read_csv(data_file, header=None)  # Create dataframe
    data.dropna()  # Drops any column that's all NaN
    data = data.replace(np.nan, 'nan')  # Converts nan objects to strings. Cannot convert apriori to list otherwise

    rows = data.shape[0]
    s = 100 / rows  # We want absolute support to be 100

    records = data.values.tolist()  # Convert df to list of lists for apriori algorithm

    for i, row in enumerate(records):  # Removes the 'nan' values from every string so apriori doesn't count them
        records[i] = [x for x in row if x != 'nan']

    print("List generation time: {:.2f}s".format(time.time() - start_time))

    print("Starting Apriori...")
    hw1rules = apriori(records, min_support=s, min_confidence=0.65, min_lift=3, min_length=2, max_length=3)
    hw1results = list(hw1rules)
    print("Apriori Time: {:.2f}s".format(time.time() - start_time))

    pair_list = []
    triple_list = []
    for item in hw1results:
        if (len(item[0])) == 2:
            pair = item[0]
            items = [x for x in pair]
            pair_list.append([items[0], items[1], item[2][0][2]])
        else:
            triple = item[0]
            items = [x for x in triple]
            triple_list.append([items[0], items[1], items[2], item[2][0][2]])

    df_pairs = pd.DataFrame(pair_list)
    df_triples = pd.DataFrame(triple_list)

    new_file = ''
    new_file += 'OUTPUT A' + '\n'
    new_file += df_pairs.sort_values(by=[2, 0], ascending=[False, True])[:5].to_string(header=False, index=False) + '\n'
    new_file += 'OUTPUT B' + '\n'
    new_file += df_triples.sort_values(by=[3, 0], ascending=[False, True])[:5].to_string(header=False, index=False)

    # print(new_file)

    with open(output_file, "w+") as f:
        f.write(str(new_file))


# Run the algorithm
write_file = "results.txt"
unformatted_file = 'browsing-data.txt'
csv_file = "csv-data.txt"

comma_delimit(unformatted_file, csv_file)
run_apriori(csv_file, write_file)
