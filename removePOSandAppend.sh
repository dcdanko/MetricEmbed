#!/bin/bash
rm master
touch master

for file in $(ls c[a-z][0-9][0-9])
do 
	echo "processing: $file"
# remove POS tags
	sed "s/\/\S\+//g" "$file" | \
# remove funny ticks and '' with double quotes (can't key in or it fiddles with Bash)
	sed "s/\(\`\` \)\|\( ''\)/\"/g" | \
# remove space before certain punctuations
	sed "s/ \([,.]\)/\1/g" > proc_$file
	cat proc_$file >> master
done

tr '[:upper:]' '[:lower:]' < master > lowerMaster # convert uppercase to lowercase
sed "s/[,.\"?!]//g" lowerMaster | sed "s/--//g" > master_noPunctuation # remove punctuation
tr -d '\n' < master_noPunctuation > master_noP_noLine # remove newlines
