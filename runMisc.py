import embed_parse as mep

lines=mep.generateBrownFilesList()
with open('brownFilesToUse.txt','w') as f:
	for file in lines:
		f.write(file+'\n')