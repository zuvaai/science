import csv
import sys
import os


counts = dict()
with open(sys.argv[1]) as response_file:
    reader = csv.reader(response_file)
    for row in reader:
        if row[0] == "Original":
            continue
        model = row[0]
        if model not in counts:
            counts[model] = [0,0,0,0]
        negate =''.join(row[2].strip().split())
        if negate == "Same":
            counts[model][0] += 1
        elif negate == "Unsure":
            counts[model][1] += 1
        
        rephrase =  ''.join(row[4].strip().split())
        print(model,negate,rephrase)
        if rephrase == "Different":
            counts[model][2] += 1
        elif rephrase == "Unsure":
            counts[model][3] += 1

for model,cnt in counts.items():
    print("{0} & {1} & {2} & {3} & {4} \\\\".format(model,cnt[0],cnt[1],cnt[2],cnt[3]))
