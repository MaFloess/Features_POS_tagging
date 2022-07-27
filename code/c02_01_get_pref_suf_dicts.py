# Imports 
import nltk



# Create a database to encode prefixes the words may contain ##########################################################

# Go through each word in the training data and create a dictionary for the most common tag for each prefix
# (provide a list of prefixes for which to check)
def create_pref_dict(prefixes_list, tagged_sents):
    
    tagged_words = sum(tagged_sents, [])
        
    prefix_tag = []
    
    for word, tag in tagged_words:
        for pref in prefixes_list:
            if len(word.removeprefix(pref)) != len(word):
                prefix_tag.append((pref, tag))
                break
      
    condf = nltk.ConditionalFreqDist((pref, tag) for (pref, tag) in prefix_tag)
     
    return {pref: condf[pref].most_common(1)[0][0] for pref in set([p for (p, _) in prefix_tag])}




# Create a database to encode suffixes the words may contain ##########################################################

# Go through each word in the training data and create a dictionary for the most common tag for each suffix
# (provide a stemmer (e.g. from the nltk package))
def create_suf_dict(stemmer, tagged_sents):
    
    tagged_words = sum(tagged_sents, [])
    
    suffix_tag = []
    
    for word, tag in tagged_words:
        if len(stemmer.stem(word)) != len(word):
            suffix_tag.append((word[len(stemmer.stem(word)):], tag))
            
    condf = nltk.ConditionalFreqDist((suffix, tag) for (suffix, tag) in suffix_tag)
     
    return {suf: condf[suf].most_common(1)[0][0] for suf in set([s for (s, _) in suffix_tag])}