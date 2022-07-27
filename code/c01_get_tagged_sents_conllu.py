# Imports
import conllu



# Get tagged sentences from CoNLL-U format 
def get_tagged_sents_conllu(file_path:str):

	data = open(file_path, mode='r', encoding='utf8')
	annotations = data.read()
	sentences = conllu.parse(annotations)

	tagged_sents = []

	for sent in sentences:

		temp_sent = []

		for token in sent:

			temp_sent.append((token['form'], token['upos']))

		tagged_sents.append(temp_sent)

	# Remove words that have no tag, e.g. in the 3-tuple
	# word1 = 'participants', tag1 = 'NOUN', word2 = '’', tag2 = 'PART', 
	# word3 = 'participants’, tag3 = '_' 	
	return [[(w,t) for (w,t) in sent if t != '_'] for sent in tagged_sents]
