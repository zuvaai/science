#!/bin/bash

while read line
do
  name=$(basename "$line")
  python3 adjudicate-similarity.py "${line}"  "$1/${name}"
done < files
