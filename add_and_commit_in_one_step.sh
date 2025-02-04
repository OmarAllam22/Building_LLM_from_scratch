#!/bin/bash
num_args=$#   # the number of arguments only excluding the command itself (ex: script.sh arg1 arg2 commit_msg) num_args is 3 not 4
echo ${num_args}
if [[ $num_args -lt 2 ]] 
then echo "Usage: $0 file_1 file_2 file_3 .... commit_msg"
else
    # Create a list of files from the arguments (all except the last one)
    files=("${@:1:num_args-1}")  # Get all arguments except the last one
    echo "Files to add: ${files[@]}"

    # Add files to git
    git add "${files[@]}"

    # Commit with the last argument as the commit message
    git commit -m "${!num_args}"  # Use indirect expansion to get the last argument
fi