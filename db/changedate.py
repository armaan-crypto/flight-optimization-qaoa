#!/usr/bin/env python3
import csv
import re
import os
import tempfile
import shutil

INFILE = 'final.csv'  # adjust path if needed
PAT = re.compile(r'(\d{1,2}:\d{2}\s*(?:AM|PM))', re.IGNORECASE)

def clean_time(cell):
    if not cell:
        return cell
    m = PAT.search(cell)
    return m.group(1).upper() if m else cell

def process(infile):
    tmp_fd, tmp_path = tempfile.mkstemp(prefix='final_', suffix='.csv')
    os.close(tmp_fd)
    with open(infile, newline='', encoding='utf-8') as inf, \
         open(tmp_path, 'w', newline='', encoding='utf-8') as outf:
        reader = csv.reader(inf)
        writer = csv.writer(outf, quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if len(row) > 5:
                row[5] = clean_time(row[5])
            if len(row) > 6:
                row[6] = clean_time(row[6])
            writer.writerow(row)
    shutil.move(tmp_path, infile)

if __name__ == '__main__':
    process(INFILE)
