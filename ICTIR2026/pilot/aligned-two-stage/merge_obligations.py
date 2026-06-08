import sys
import csv
import json

def parseObls(obls, gold, pred):
    unmatchedGold = [gold[idx] for idx in obls['unmatchedList1']]
    unmatchedThis = [pred[idx] for idx in obls['unmatchedList2']]
    matched = [[gold[idx1],pred[idx2]] for idx1,idx2 in obls['identical']]
    opposite = [[gold[idx1],pred[idx2]] for idx1,idx2 in obls['opposite']]
    return matched,unmatchedGold,unmatchedThis,opposite


obs = dict()
with open(sys.argv[1]) as obligation_file:
    reader = csv.reader(obligation_file)
    for row in reader:
        obs[row[0]] = json.loads(row[1])

clauses = dict()
with open(sys.argv[3]) as clause_file:
    reader = csv.reader(clause_file)
    for row in reader:
        clauses[row[0]] = row[1]

rows = [['Doc','Clause','Score','Rank','Matched','Unmatched Ref','Unmatched Clause','Opposite']]
gold = ""
with open(sys.argv[2]) as aligned_file:
    reader = csv.reader(aligned_file)
    for idx,row in enumerate(reader):
        if idx == 0:
            continue
        if idx == 1:
            gold = row[0]
            rows.append([row[0],clauses[row[0]],row[2],-1,"N","N","N","N"])
            continue
        obls = json.loads(row[1].replace('\'','"'))
        matched, unmatchedGold, unmatchedThis, opposite = parseObls(obls, obs[gold],obs[row[0]])
        rows.append([row[0],clauses[row[0]],row[2],row[3],matched,unmatchedGold,unmatchedThis,opposite])

with open(sys.argv[4],'w') as outfile:
    writer = csv.writer(outfile)
    for row in rows:
        writer.writerow(row)