# Imports
import numpy as np
import nltk
import gensim



# Functions for encoding a single feature #############################################################################
def is_alpha_encoded(word):
    return 1 if word.isalpha() else 0

def is_numeric_encoded(word):
    return 1 if word.isdigit() else 0

def is_alphanumeric_encoded(word):
    return 1 if word.isalnum() else 0

def is_capitalized_encoded(word):
    return 1 if word[0].isupper() else 0

def capital_inside_encoded(word):
    return 1 if any([char.isupper() for char in word[1:]]) else 0

def is_all_caps_encoded(word):
    return 1 if word.isupper() else 0

def is_all_lower_encoded(word):
    return 1 if word.islower() else 0

def digit_inside_encoded(word):
    return 1 if any([char.isdigit() for char in word]) else 0

def contains_hyphen_encoded(word):
    return 1 if "-" in word else 0



# Function that returns a one-hot-encoding for a single word with most common tag provided that a prefix of the 
# prefix-dictionary was found in it
def prefix_encoded(word, prefix_dict):
    
    word = word.lower()
    
    prefix_tags = np.array(list(set(prefix_dict.values())))
    prefix_tags.sort()
    
    prefix_encoding = np.array([0 for _ in range(len(prefix_tags))])
    
    word_prefix_tag = ""
    
    for pref in prefix_dict.keys():
        if len(word.removeprefix(pref)) != len(word):
            word_prefix_tag = prefix_dict[pref]
            
    np.putmask(prefix_encoding, word_prefix_tag == prefix_tags, 1)
    
    return prefix_encoding 



# Function that returns a one-hot-encoding for a single word with most common tag provided that a suffix of the 
# suffix-dictionary was found in it
def suffix_encoded(word, suffix_dict, stemmer):

    word = word.lower()
    
    suffix_tags = np.array(list(set(suffix_dict.values())))
    suffix_tags.sort()

    suffix_encoding = np.array([0 for _ in range(len(suffix_tags))])

    word_suffix_tag = ""
                
    if len(stemmer.stem(word)) != len(word):
        if word[len(stemmer.stem(word)):] in suffix_dict.keys():
            word_suffix_tag = suffix_dict[word[len(stemmer.stem(word)):]]
            
    np.putmask(suffix_encoding, word_suffix_tag == suffix_tags, 1)
    
    return suffix_encoding 


#######################################################################################################################

# Important features of the encoding of data for which combinations in the final analysis will be evaluated
# Feature groups:
# - char_related(alpha/numeric/aphanumeric/digit_inside/contains-hyphen)
# - case_related (capitalized/capitel_inside/all_caps/all_lower)
# - sent_positions (first/last)
# - affixes (prefix/suffix)

# Additional feature:
# - position of embedding


# Function to encode all features which are set True in the parameters
# Encoding function combining every single function relevant for inputs
def encode_features(sents,
    tagged=True,
    word_embedding=False,
    word_embedding_at_end=False,
    use_glove=False,
    golve_dimension=100,
    word_embedding_path=None,

    char_related=False,
    is_alpha=False, 
    is_numeric=False,
    is_alphanumeric=False,
    digit_inside=False,
    contains_hyphen=False,

    case_related=False,
    is_capitalized=False,
    capital_inside=False,
    is_all_caps=False,
    is_all_lower=False,
    
    sent_positions=False,
    is_first=False,
    is_last=False,

    affixes=False,
    prefix=False,
    suffix=False,
    pref_dict=None,
    suf_dict=None,
    stemmer=None): 

    result = []

    encoding_functions = []

    if word_embedding:

        if use_glove:
            embedding = dict()
            f = open(('data\\word_embeddings\\glove.6B.' + str(golve_dimension).strip() + 'd.txt'), encoding='utf-8')

            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embedding[word] = coefs

            f.close

        else:
            embedding = gensim.models.KeyedVectors.load(word_embedding_path)


    if is_alpha or char_related:
        encoding_functions.append(is_alpha_encoded)

    if is_numeric or char_related:
        encoding_functions.append(is_numeric_encoded)

    if is_alphanumeric or char_related:
        encoding_functions.append(is_alphanumeric_encoded)

    if digit_inside or char_related:
        encoding_functions.append(digit_inside_encoded)

    if contains_hyphen or char_related:
        encoding_functions.append(contains_hyphen_encoded)

    
    if is_capitalized or case_related:
        encoding_functions.append(is_capitalized_encoded)

    if capital_inside or case_related:
        encoding_functions.append(capital_inside_encoded)

    if is_all_caps or case_related:
        encoding_functions.append(is_all_caps_encoded)

    if is_all_lower or case_related:
        encoding_functions.append(is_all_lower_encoded)


    for j in range(len(sents)):

        encoded_sent = []

        for i in range(len(sents[j])):
            
            if tagged:
                word = sents[j][i][0]
            else:
                word = sents[j][i]

            encoded_word = np.array([])

            if word_embedding and not word_embedding_at_end:
                if use_glove:
                    if word.lower() in embedding.keys():
                        encoded_word = np.append(encoded_word, embedding[word.lower()])
                    else:
                        encoded_word = np.append(encoded_word, np.array([0 for _ in range(golve_dimension)]))

                else: 
                    if word.lower() in embedding.key_to_index.keys():
                        encoded_word = np.append(encoded_word, embedding[word.lower()])
                    else:
                        encoded_word = np.append(encoded_word, np.array([0 for _ in range(embedding.vector_size)]))
                    
                
            for fun in encoding_functions:
                encoded_word = np.append(encoded_word, fun(word))


            if is_first or sent_positions:
                if i == 0:
                    encoded_word = np.append(encoded_word, 1)
                else:
                    encoded_word = np.append(encoded_word, 0)

            if is_last or sent_positions:
                if i == (len(sents[j]) - 1):
                    encoded_word = np.append(encoded_word, 1)
                else:
                    encoded_word = np.append(encoded_word, 0)


            if prefix or affixes:
                encoded_word = np.concatenate((encoded_word, prefix_encoded(word, pref_dict)), axis=None)

            if suffix or affixes:
                encoded_word = np.concatenate((encoded_word, suffix_encoded(word, suf_dict, stemmer)), axis=None)

            if word_embedding and word_embedding_at_end:
                if use_glove:
                    if word.lower() in embedding.keys():
                        encoded_word = np.append(encoded_word, embedding[word.lower()])
                    else:
                        encoded_word = np.append(encoded_word, np.array([0 for _ in range(golve_dimension)]))

                else: 
                    if word.lower() in embedding.key_to_index.keys():
                        encoded_word = np.append(encoded_word, embedding[word.lower()])
                    else:
                        encoded_word = np.append(encoded_word, np.array([0 for _ in range(embedding.vector_size)]))


            encoded_sent.append(encoded_word)
            

        result.append(np.array(encoded_sent))

    return np.array(result, dtype=object)



# Encoding function for tags (one-hot-encoding) #######################################################################
def encode_tags(tagged_sents):
    
    tags = list(set([t for (w, t) in sum(tagged_sents, [])]))
    tags.sort()

    result = []
    
    for sent_count in range(len(tagged_sents)):
        temp_sent = tagged_sents[sent_count]
        sent_result = []
                            
        for word_count in range(len(temp_sent)):
            tag_encoding = (np.array(tags) == temp_sent[word_count][1]).astype(int)
            sent_result.append(tag_encoding)

                                        
        result.append(np.array(sent_result))
                            
    return np.array(result, dtype=object)
