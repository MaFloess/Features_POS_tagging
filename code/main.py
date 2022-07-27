# Setup for console input
# Which will be used for word embedding creations and regulate some combinations of parameters on which evaluations will be produced
# To get the full workflow which was covered in the thesis exchange the toggled comment parts in the section 'This main method will:'

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--we_min_count', 
    type=int,
    help="Minimal word count for Word2Vec/FastText model creation, (Covered in thesis: [1, 2, 3, 5])", 
    default=2)
parser.add_argument('--we_window_size', 
    type=int,
    help="Window size for Word2Vec/FastText model creation, (Covered in thesis: [2, 3, 5])", 
    default=3)
parser.add_argument('--we_vector_size', 
    type=int,
    help="Vector sizes (dimensions) for Word2Vec/FastText model creation, (Covered in thesis: [50, 100, 200])", 
    default=200)
parser.add_argument('--we_alpha', 
    type=int,
    help="Alpha (learning rate) for Word2Vec/FastText model creation, (Covered in thesis: [0.015, 0.03, 0.045])", 
    default=0.045)
parser.add_argument('--we_implementation', 
    type=str,
    help="Implementation (architecture) for Word2Vec/FastText model creation, (Covered in thesis: [\'cbow\', \'sg\'])", 
    default='sg')

parser.add_argument('--ft_char_range_min', 
    type=int,
    help="Character range (lower boundary) for FastText model creation, (Covered in thesis: [(2, 5), (3, 6)])", 
    default=2)
parser.add_argument('--ft_char_range_max', 
    type=int,
    help="Character range (upper boundary) for FastText model creation, (Covered in thesis: [(2, 5), (3, 6)])", 
    default=5)

parser.add_argument('--unit_count', 
    type=int,
    help="Unit count for the LSTM layer in the POS-tagger model, (Covered in thesis: [75, 125])", 
    default=125)

parser.add_argument('--glove_dimensions', 
    type=int,
    help="Dimensions (vector size) for GloVe model creation, (Covered in thesis: [50, 100, 200])", 
    default=200)


parser.add_argument('--eval_char_related', 
    type=int,
    help="0 or 1 for either not including or including character related features for final evaluation of POS-tagger models", 
    default=1)
parser.add_argument('--eval_case_related', 
    type=int,
    help="0 or 1 for either not including or including case related features for final evaluation of POS-tagger models", 
    default=1)
parser.add_argument('--eval_sent_position', 
    type=int,
    help="0 or 1 for either not including or including sentence positional features for final evaluation of POS-tagger models", 
    default=1)
parser.add_argument('--eval_affixes', 
    type=int,
    help="0 or 1 for either not including or including affix related features for final evaluation of POS-tagger models", 
    default=1)
parser.add_argument('--eval_we_end', 
    type=int,
    help="0 or 1 for either not putting or putting the word embeddings at the end of the encoded features for final evaluation of POS-tagger models", 
    default=1)

args = parser.parse_args()




# This main method will: ######################################################################################################################################


# 1) Automatically create (if they don't exist) Word2Vec embeddings: all specified combinations below 

# Covered in thesis:
# reasonable_min_counts = [1, 2, 3, 5]
# reasonable_window_sizes = [2, 3, 5]
# reasonable_vector_sizes = [50, 100, 200]
# reasonable_alphas = [0.015, 0.03, 0.045]
# implementation = ['cbow', 'sg']

# Chosen for your execution:
reasonable_min_counts = [args.we_min_count]
reasonable_window_sizes = [args.we_window_size]
reasonable_vector_sizes = [args.we_vector_size]
reasonable_alphas = [args.we_alpha]
implementation = [args.we_implementation]


# 2) Evaluate (if not done before) all of the Word2Vec embeddings (accuracy of a LSTM), 
# specified through combinations above in combination with unit counts below and add them to the 'results\w2v_lstm_evaluation.csv' file

# Covered in thesis:
# reasonable_unit_counts = [75, 125]

# Chosen for your execution:
reasonable_unit_counts = [args.unit_count]


# 3) Automatically create (if they don't exist) FastText embeddings: 
# Combinations of the hyperparameters of the (at maximum) 5 best performing Word2Vec models in 'results\w2v_lstm_evaluation.csv'
# with character ranges specified below

# Covered  in thesis:
# reasonable_character_ranges = [(2, 5), (3, 6)] 

# Chosen for your execution:
reasonable_character_ranges = [(args.ft_char_range_min, args.ft_char_range_max)]


# 4) Evaluate (if not done before) all the FastText embedding (accuracy of a LSTM)
# and add them to the 'results\fasttext_lstm_evaluation.csv' file


# 5) Evaluate LSTMs with combinations of dimensions and unit counts of LSTM layer for the pretrained Stanford GloVe embedding
# ('Wikipedia 2014 + Gigaword 5: glove6b.zip' must be unzipped in the 'data\word_embeddings' folder)
# Unit counts of reasonable_unit_counts are reused

# Covered in thesis:
# reasonable_dimension_glove = [50, 100, 200]

# Chosen for your execution:
reasonable_dimension_glove = [args.glove_dimensions]

# 6) Choose the best embeddings and their parameters of the self-trained Word2Vec, FastText and pretrained GloVe models


# 7) Evaluate (if not done before) model combinations of the best performing embeddings (with the used unit count) for Word2Vec, FastText, GloVe.
# The result will be saved in a dict in a json file with a name corresponding to the combination used for the data encoding.

# Combinations covered in thesis (all):
# char_related_choices = [0, 1]
# case_related_choices = [0, 1]
# sent_positions_choices = [0, 1]
# affixes_choices = [0, 1]
# word_embedding_at_end_choices = [0, 1]

# Chosen for your execution:
char_related_choices = [args.eval_char_related]
case_related_choices = [args.eval_case_related]
sent_positions_choices = [args.eval_sent_position]
affixes_choices = [args.eval_affixes]
word_embedding_at_end_choices = [args.eval_we_end]


# 8) Compare the results of 7) in a table



# Setup #######################################################################################################################################################

# Import of packages
import sys
import os
import csv
import operator
import json
import tabulate

# Import necessary self-created modules
sys.path.append('\\code')

from c01_get_tagged_sents_conllu import *
from c02_01_get_pref_suf_dicts import *
from c02_02_encode_data import *
from c02_03_pretraining_functions import *
from c03_01_generate_word2vec_model import *
from c03_02_generate_fasttext_model import *
from c04_01_build_lstm_predict import *
from c04_02_get_evaluation_dict import *
from c04_03_get_evaluation_metrics import *


# Get tagged data
tagged_sents_train = get_tagged_sents_conllu('data\\en_gum-ud-train.conllu')
tagged_sents_dev = get_tagged_sents_conllu('data\\en_gum-ud-dev.conllu')
tagged_sents_test = get_tagged_sents_conllu('data\\en_gum-ud-test.conllu')

# Get tags present in training data (should contain the whole tagset)
tags = list(set([t for (w,t) in sum(tagged_sents_train, [])]))
tags.sort()

# Encode the gold standard of the data for the data sets
Y_dev = encode_tags(tagged_sents_dev)
Y_test = encode_tags(tagged_sents_test)

# Get untagged sents for creating word embeddings
sents_train = [[w for (w,t) in sent] for sent in tagged_sents_train]




# 1)
# Create Word2Vec word embeddings that are not already present in the respective repository
for mc in reasonable_min_counts:
    for ws in reasonable_window_sizes:
        for vs in reasonable_vector_sizes:
            for a in reasonable_alphas:
                for i in implementation:

                    if not os.path.exists('data\\word_embeddings\\w2v_' + str(mc) + '_' + str(ws) + '_' + str(vs) + '_' + str(a)[2:] + '_' + i):

                        print('A Word2Vec model with minimal word frequency ' + str(mc) + ', window size ' + str(ws) + ', vector dimension ' + str(vs) +
                                ', learning rate ' + str(a) + ' and the ' + i + '-implementation will be created.\n', sep='')

                        generate_save_word2vec(sents_train, path='data\\word_embeddings\\', min_count=mc, window=ws, vector_size=vs, alpha=a, implementation=i)

                    else:
                        print('Exists already: A Word2Vec model with minimal word frequency ' + str(mc) + ', window size ' + str(ws) + ', vector dimension ' +
                            str(vs) + ', learning rate ' +  str(a) + ' and the ' + i + '-implementation.\n', sep='')




# 2)
# Check for the evaluation file of Word2Vec models and make a list of all previously tested models
if not os.path.exists('results\\w2v_lstm_evaluation.csv'):

    with open('results\\w2v_lstm_evaluation.csv', 'w') as f:

        print('modelname, min_count, window_size, vector_size, learning_rate, architecture, unit_count, accuracy', file=f)
        previously_tested = []

else:
    previously_tested = []

    with open('results\\w2v_lstm_evaluation.csv', 'r') as f:

        csvreader = csv.reader(f)
        header = next(csvreader)
        del header
        for row in csvreader:
            previously_tested.append(row)



# Evaluate all POS-tagging models with Word2Vec word embeddings as their sole input, if it was not done before (in previously_tested)
with open('results\\w2v_lstm_evaluation.csv', 'a') as f:

    for mc in reasonable_min_counts:
        for ws in reasonable_window_sizes:
            for vs in reasonable_vector_sizes:
                for al in reasonable_alphas:
                    for i in implementation:
                        for unit_count in reasonable_unit_counts:

                            model_name = 'w2v_'+ str(mc) + '_' + str(ws) + '_' + str(vs) + '_' + str(al)[2:] + '_' + i + '_' + str(unit_count)

                            if model_name in [row[0] for row in previously_tested]:

                                print(model_name + ' has already been evaluated.' + ' (Accuraccy: ' + 
                                    str([row[7] for row in previously_tested if row[0] == model_name][0]) + ')\n', sep='')

                            else:
                                pathname_we = 'data\\word_embeddings\\w2v_' + str(mc) + '_' + str(ws) + '_' + str(vs) + '_' + str(al)[2:] + '_' + i

                                X_train = encode_features(tagged_sents_train,
                                    word_embedding=True,
                                    word_embedding_path=pathname_we)

                                X_dev = encode_features(tagged_sents_dev,
                                    word_embedding=True,
                                    word_embedding_path=pathname_we)

                                Y_train = encode_tags(tagged_sents_train)

                                # Pad training data
                                X_train = pad_sequence_data(X_train)
                                Y_train = pad_sequence_data(Y_train)

                                # Shuffle data
                                X_train, Y_train = unison_shuffled_copies(X_train, Y_train)

                                predictions = build_lstm_predict(X_train, Y_train, X_dev, unit_count)

                                evaluation_dict = get_evaluation_dict(predictions, Y_dev, tags)

                                full_evaluation_dict = get_evaluation_metrics(evaluation_dict, tags)

                                print(model_name + ' has achieved an accuracy of', full_evaluation_dict['accuracy'])

                                print(model_name, mc, ws, vs, al, i, unit_count, full_evaluation_dict['accuracy'], sep = ', ', file=f)




# 3)
# Get the hyperparameter combinations of the (at maximum) 5 best performing Word2Vec embeddings in LSTMs with the used unit count
w2v_models = []

with open('results\\w2v_lstm_evaluation.csv', 'r') as f:
    csvreader = csv.reader(f)
    header = next(csvreader)
    del header
    for row in csvreader:
        w2v_models.append(row)


if len(w2v_models) > 5:
    w2v_models = sorted(w2v_models, key=operator.itemgetter(7))[-5:]


w2v_basis = []

for row in w2v_models:
    w2v_basis.append(dict(min_count=int(row[1]), window_size=int(row[2]), vector_size=int(row[3]), 
        alpha=float(row[4]), implementation=row[5].strip(), unit_count=int(row[6])))



# Create FastText word embeddings that are not already present in the respective repository
for model_dict in w2v_basis:
    for crange in reasonable_character_ranges:

        if not os.path.exists('data\\word_embeddings\\fasttext_' + 
            str(model_dict['min_count']) + '_' + str(model_dict['window_size']) + '_' + 
            str(model_dict['vector_size']) + '_' + str(model_dict['alpha'])[2:] + '_' + 
            model_dict['implementation'] + '_' + str(crange[0]) + '_' + str(crange[1])):

            print('A FastText model with minimal word frequency ' + str(model_dict['min_count']) + ', window size ' + str(model_dict['window_size']) +
                ', vector dimension ' + str(model_dict['vector_size']) + ', learning rate ' +  str(model_dict['alpha']) + ', ' + 
                model_dict['implementation'] + '-implementation, min_n ' + str(crange[0]) + ' and max_n ' + str(crange[1]) + ' will be created.\n', sep='')

            generate_save_fasttext(sents_train, path='data\\word_embeddings\\', min_count=model_dict['min_count'], window=model_dict['window_size'], 
                vector_size=model_dict['vector_size'], alpha=model_dict['alpha'], implementation=model_dict['implementation'], min_n=crange[0], max_n=crange[1])


        else:
            print('Exists already: A FastText model with minimal word frequency ' + str(model_dict['min_count']) + ', window size ' + str(model_dict['window_size']) +
                ', vector dimension ' + str(model_dict['vector_size']) + ', learning rate ' +  str(model_dict['alpha']) + ', ' + 
                model_dict['implementation'] + '-implementation, min_n ' + str(crange[0]) + ' and max_n ' + str(crange[1]) + '.\n', sep='')   




# 4)
# Check for the evaluation file of FastText models and make a list of all previously tested models
if not os.path.exists('results\\fasttext_lstm_evaluation.csv'):

    with open('results\\fasttext_lstm_evaluation.csv', 'w') as f:

        print('modelname, min_count, window_size, vector_size, learning_rate, architecture, min_n, max_n, unit_count, accuracy', file=f)
        previously_tested = []

else:
    previously_tested = []

    with open('results\\fasttext_lstm_evaluation.csv', 'r') as f:

        csvreader = csv.reader(f)
        header = next(csvreader)
        del header
        for row in csvreader:
            previously_tested.append(row)



# Evaluate all POS-tagging models with FastText word embeddings as their sole input, if it was not done before (in previously_tested)
with open('results\\fasttext_lstm_evaluation.csv', 'a') as f:

    for model_dict in w2v_basis:
        for crange in reasonable_character_ranges:

            model_name = ('fasttext_' + str(model_dict['min_count']) + '_' + str(model_dict['window_size']) + '_' + str(model_dict['vector_size']) + 
                '_' +  str(model_dict['alpha'])[2:] + '_' + model_dict['implementation'] + '_' + str(crange[0]) + '_' + str(crange[1]) + '_' + 
                str(model_dict['unit_count']))

            if model_name in [row[0] for row in previously_tested]:

                print(model_name + ' has already been evaluated.' + ' (Accuraccy: ' + 
                    str([row[9] for row in previously_tested if row[0] == model_name][0]) + ')\n', sep='')

            else:

                pathname_we = ('data\\word_embeddings\\fasttext_' + str(model_dict['min_count']) + '_' + str(model_dict['window_size']) + '_' + 
                    str(model_dict['vector_size']) + '_' +  str(model_dict['alpha'])[2:] + '_' + model_dict['implementation'] + '_' +
                     str(crange[0]) + '_' + str(crange[1]))

                X_train = encode_features(tagged_sents_train,
                    word_embedding=True,
                    word_embedding_path=pathname_we)

                X_dev = encode_features(tagged_sents_dev,
                    word_embedding=True,
                    word_embedding_path=pathname_we)

                Y_train = encode_tags(tagged_sents_train)

                # Pad training data
                X_train = pad_sequence_data(X_train)
                Y_train = pad_sequence_data(Y_train)

                # Shuffle data
                X_train, Y_train = unison_shuffled_copies(X_train, Y_train)

                predictions = build_lstm_predict(X_train, Y_train, X_dev, unit_count)

                evaluation_dict = get_evaluation_dict(predictions, Y_dev, tags)

                full_evaluation_dict = get_evaluation_metrics(evaluation_dict, tags)

                print(model_name + ' has achieved an accuracy of', full_evaluation_dict['accuracy'])

                print(model_name, model_dict['min_count'], model_dict['window_size'], model_dict['vector_size'], model_dict['alpha'],
                     model_dict['implementation'], crange[0], crange[1], model_dict['unit_count'], full_evaluation_dict['accuracy'], 
                     sep = ', ', file=f)




# 5)
# Check for the evaluation file of GloVe models and make a list of all previously tested models
if not os.path.exists('results\\glove_lstm_evaluation.csv'):

    with open('results\\glove_lstm_evaluation.csv', 'w') as f:

        print('Wikipedia 2014 + Gigaword 5 (glove.6B), dimensions, unit_count, accuracy', file=f)
        previously_tested = []

else:
    previously_tested = []

    with open('results\\glove_lstm_evaluation.csv', 'r') as f:

        csvreader = csv.reader(f)
        header = next(csvreader)
        del header
        for row in csvreader:
            previously_tested.append(row)



# Evaluate all POS-tagging models with GloVe word embeddings as their sole input, if it was not done before (in previously_tested)
with open('results\\glove_lstm_evaluation.csv', 'a') as f:

    for dim in reasonable_dimension_glove:
        for unit_count in reasonable_unit_counts:

            model_name = 'glove.6B.' + str(dim) + '_' + str(unit_count)

            if model_name in [row[0] for row in previously_tested]:

                print(model_name + ' has already been evaluated.' + ' (Accuraccy: ' + 
                    str([row[3] for row in previously_tested if row[0] == model_name][0]) + ')\n', sep='')

            else:

                X_train = encode_features(tagged_sents_train,
                    word_embedding=True,
                    use_glove=True,
                    golve_dimension=dim)

                X_dev = encode_features(tagged_sents_dev,
                    word_embedding=True,
                    use_glove=True,
                    golve_dimension=dim)

                Y_train = encode_tags(tagged_sents_train)

                # Pad training data
                X_train = pad_sequence_data(X_train)
                Y_train = pad_sequence_data(Y_train)

                # Shuffle data
                X_train, Y_train = unison_shuffled_copies(X_train, Y_train)

                predictions = build_lstm_predict(X_train, Y_train, X_dev, unit_count)

                evaluation_dict = get_evaluation_dict(predictions, Y_dev, tags)

                full_evaluation_dict = get_evaluation_metrics(evaluation_dict, tags)

                print(model_name + ' has achieved an accuracy of', full_evaluation_dict['accuracy'])

                print(model_name, str(dim), str(unit_count), full_evaluation_dict['accuracy'], sep = ', ', file=f)





# 6) 
# Choose the best embeddings and their parameters of the self-trained Word2Vec, FastText and pre-trained GloVe models

w2v_models = []
with open('results\\w2v_lstm_evaluation.csv', 'r') as f:
    csvreader = csv.reader(f)
    header = next(csvreader)
    del header
    for row in csvreader:
        w2v_models.append(row)


best_w2v_model = sorted(w2v_models, key=operator.itemgetter(7))[-1]


fasttext_models = []
with open('results\\fasttext_lstm_evaluation.csv', 'r') as f:
    csvreader = csv.reader(f)
    header = next(csvreader)
    del header
    for row in csvreader:
        fasttext_models.append(row)


best_fasttext_model = sorted(fasttext_models, key=operator.itemgetter(9))[-1]


glove_models = []
with open('results\\glove_lstm_evaluation.csv', 'r') as f:
    csvreader = csv.reader(f)
    header = next(csvreader)
    del header
    for row in csvreader:
        glove_models.append(row)


best_glove_model = sorted(glove_models, key=operator.itemgetter(3))[-1]



best_w2v_model_name = ('w2v_' + best_w2v_model[1].strip() + '_' + best_w2v_model[2].strip()  + '_' + best_w2v_model[3].strip()  + '_' + 
    best_w2v_model[4].strip()[2:]  + '_' + best_w2v_model[5].strip())

best_w2v_model_path = 'data\\word_embeddings\\' + best_w2v_model_name


best_fasttext_model_name = ('fasttext_' + best_fasttext_model[1].strip() + '_' + best_fasttext_model[2].strip()  + '_' + best_fasttext_model[3].strip()  + '_' + 
    best_fasttext_model[4].strip()[2:]  + '_' + best_fasttext_model[5].strip() + '_' + best_fasttext_model[6].strip() + '_' + best_fasttext_model[7].strip())

best_fasttext_model_path = 'data\\word_embeddings\\' + best_fasttext_model_name


best_glove_model_name = 'glove.6B.' + str(best_glove_model[1]).strip()


print('\nThe best previously evaluated word embeddings were:\n\n', 
    best_w2v_model_name + ' with a unit count of ' + str(best_w2v_model[6]) + ' (Accuraccy: ' + str(best_w2v_model[7]) +  ')\n\n',
    best_fasttext_model_name + ' with a unit count of ' + str(best_fasttext_model[8]) + ' (Accuraccy: ' + str(best_fasttext_model[9]) +  ')\n\n', 
    best_glove_model_name + ' with a unit count of ' + str(best_glove_model[2]) + ' (Accuraccy: ' + str(best_glove_model[3]) +  ')\n\n',
    'These will be fully evaluated with combinations of the feature groups contained in \'code\\c02_02_encode_data\'.\n', 
    sep='')




# 7) 
# Evaluate (if not done before) model combinations of the best performing embeddings (with the used unit count) for Word2Vec, FastText, GloVe.
# The result will be saved in a dict in a json file with a name corresponding to the combination used for data encoding.


# Build necessary dictionaries for affix encoding -------------------------------------------------------------------------------------------------------------

# According to the Cambridge Dictionary (https://dictionary.cambridge.org/grammar/british-grammar/prefixes)
most_common_english_prefixes = [
"anti",    # e.g. anti-goverment, anti-racist, anti-war
"auto",    # e.g. autobiography, automobile
"de",      # e.g. de-classify, decontaminate, demotivate
"dis",     # e.g. disagree, displeasure, disqualify
"down",    # e.g. downgrade, downhearted
"extra",   # e.g. extraordinary, extraterrestrial
"hyper",   # e.g. hyperactive, hypertension
"il",     # e.g. illegal
"im",     # e.g. impossible
"in",     # e.g. insecure
"ir",     # e.g. irregular
"inter",  # e.g. interactive, international
"mega",   # e.g. megabyte, mega-deal, megaton
"mid",    # e.g. midday, midnight, mid-October
"mis",    # e.g. misaligned, mislead, misspelt
"non",    # e.g. non-payment, non-smoking
"over",  # e.g. overcook, overcharge, overrate
"out",    # e.g. outdo, out-perform, outrun
"post",   # e.g. post-election, post-warn
"pre",    # e.g. prehistoric, pre-war
"pro",    # e.g. pro-communist, pro-democracy
"re",     # e.g. reconsider, redo, rewrite
"semi",   # e.g. semicircle, semi-retired
"sub",    # e.g. submarine, sub-Saharan
"super",   # e.g. super-hero, supermodel
"tele",    # e.g. television, telephathic
"trans",   # e.g. transatlantic, transfer
"ultra",   # e.g. ultra-compact, ultrasound
"un",      # e.g. under-cook, underestimate
"up",      # e.g. upgrade, uphill
]

pref_dict = create_pref_dict(most_common_english_prefixes, tagged_sents_train)

# Use an aggresive stemmer of the nltk package
snowball_stemmer = nltk.stem.SnowballStemmer('english')

suf_dict = create_suf_dict(snowball_stemmer, tagged_sents_train)



# Evaluate the combinations for the best performing Word2Vec model -------------------------------------------------------------------------------------------
for char_related in char_related_choices:
    for case_related in case_related_choices:
        for sent_positions in sent_positions_choices:
            for affixes in affixes_choices:
                for word_embedding_at_end in word_embedding_at_end_choices:

                    if (not char_related and not case_related and not sent_positions and not affixes):
                        word_embedding_at_end = 0

                    file_name = (best_w2v_model_name + '_' + str(best_w2v_model[6]).strip() + '_' + str(char_related) + '_' + str(case_related) + '_' + 
                        str(sent_positions) + '_' + str(affixes) + '_' + str(word_embedding_at_end) + '.json')

                    if not os.path.exists('results\\' + file_name):

                        X_train = encode_features(tagged_sents_train,
                            word_embedding=True,
                            word_embedding_at_end=word_embedding_at_end,
                            word_embedding_path=best_w2v_model_path,
                            char_related=char_related,
                            case_related=case_related,
                            sent_positions=sent_positions,
                            affixes=affixes,
                            pref_dict=pref_dict,
                            suf_dict=suf_dict,
                            stemmer=snowball_stemmer)

                        X_dev = encode_features(tagged_sents_dev,
                            word_embedding=True,
                            word_embedding_at_end=word_embedding_at_end,
                            word_embedding_path=best_w2v_model_path,
                            char_related=char_related,
                            case_related=case_related,
                            sent_positions=sent_positions,
                            affixes=affixes,
                            pref_dict=pref_dict,
                            suf_dict=suf_dict,
                            stemmer=snowball_stemmer)

                        Y_train = encode_tags(tagged_sents_train)

                        # Pad training data
                        X_train = pad_sequence_data(X_train)
                        Y_train = pad_sequence_data(Y_train)

                        # Shuffle data
                        X_train, Y_train = unison_shuffled_copies(X_train, Y_train)

                        predictions = build_lstm_predict(X_train, Y_train, X_dev, int(best_w2v_model[6]))

                        evaluation_dict = get_evaluation_dict(predictions, Y_dev, tags)

                        full_evaluation_dict = get_evaluation_metrics(evaluation_dict, tags)

                        json.dump(full_evaluation_dict, open(('results\\' + file_name), 'w'))

                        print(file_name + ' has been saved to the results folder. (Accuracy: ' + str(full_evaluation_dict['accuracy']) + ')\n')

                    else:

                        print(file_name[:-5] + ' has already been evaluated.\n')



# Evaluate the combinations for the best performing FastText model -------------------------------------------------------------------------------------------
for char_related in char_related_choices:
    for case_related in case_related_choices:
        for sent_positions in sent_positions_choices:
            for affixes in affixes_choices:
                for word_embedding_at_end in word_embedding_at_end_choices:

                    if (not char_related and not case_related and not sent_positions and not affixes):
                        word_embedding_at_end = 0

                    file_name = (best_fasttext_model_name + '_' + str(best_fasttext_model[8]).strip() + '_' + str(char_related) + '_' + str(case_related) + '_' + 
                        str(sent_positions) + '_' + str(affixes) + '_' + str(word_embedding_at_end) + '.json')

                    if not os.path.exists('results\\' + file_name):

                        X_train = encode_features(tagged_sents_train,
                            word_embedding=True,
                            word_embedding_at_end=word_embedding_at_end,
                            word_embedding_path=best_fasttext_model_path,
                            char_related=char_related,
                            case_related=case_related,
                            sent_positions=sent_positions,
                            affixes=affixes,
                            pref_dict=pref_dict,
                            suf_dict=suf_dict,
                            stemmer=snowball_stemmer)

                        X_dev = encode_features(tagged_sents_dev,
                            word_embedding=True,
                            word_embedding_at_end=word_embedding_at_end,
                            word_embedding_path=best_fasttext_model_path,
                            char_related=char_related,
                            case_related=case_related,
                            sent_positions=sent_positions,
                            affixes=affixes,
                            pref_dict=pref_dict,
                            suf_dict=suf_dict,
                            stemmer=snowball_stemmer)

                        Y_train = encode_tags(tagged_sents_train)

                        # Pad training data
                        X_train = pad_sequence_data(X_train)
                        Y_train = pad_sequence_data(Y_train)

                        # Shuffle data
                        X_train, Y_train = unison_shuffled_copies(X_train, Y_train)

                        predictions = build_lstm_predict(X_train, Y_train, X_dev, int(best_fasttext_model[8]))

                        evaluation_dict = get_evaluation_dict(predictions, Y_dev, tags)

                        full_evaluation_dict = get_evaluation_metrics(evaluation_dict, tags)

                        json.dump(full_evaluation_dict, open(('results\\' + file_name), 'w'))

                        print(file_name + ' has been saved to the results folder. (Accuracy: ' + str(full_evaluation_dict['accuracy']) + ')\n')

                    else:

                        print(file_name[:-5] + ' has already been evaluated.\n')




# Evaluate the combinations  for the best performing GloVe model -------------------------------------------------------------------------------------------
for char_related in char_related_choices:
    for case_related in case_related_choices:
        for sent_positions in sent_positions_choices:
            for affixes in affixes_choices:
                for word_embedding_at_end in word_embedding_at_end_choices:

                    if (not char_related and not case_related and not sent_positions and not affixes):
                        word_embedding_at_end = 0

                    file_name = (best_glove_model_name + '_' + str(best_glove_model[2]).strip() + '_' + str(char_related) + '_' + str(case_related) + '_' + 
                        str(sent_positions) + '_' + str(affixes) + '_' + str(word_embedding_at_end) + '.json')

                    if not os.path.exists('results\\' + file_name):

                        X_train = encode_features(tagged_sents_train,
                            word_embedding=True,
                            word_embedding_at_end=word_embedding_at_end,
                            use_glove=True,
                            golve_dimension=int(best_glove_model[1]),
                            char_related=char_related,
                            case_related=case_related,
                            sent_positions=sent_positions,
                            affixes=affixes,
                            pref_dict=pref_dict,
                            suf_dict=suf_dict,
                            stemmer=snowball_stemmer)

                        X_dev = encode_features(tagged_sents_dev,
                            word_embedding=True,
                            word_embedding_at_end=word_embedding_at_end,
                            use_glove=True,
                            golve_dimension=int(best_glove_model[1]),
                            char_related=char_related,
                            case_related=case_related,
                            sent_positions=sent_positions,
                            affixes=affixes,
                            pref_dict=pref_dict,
                            suf_dict=suf_dict,
                            stemmer=snowball_stemmer)

                        Y_train = encode_tags(tagged_sents_train)

                        # Pad training data
                        X_train = pad_sequence_data(X_train)
                        Y_train = pad_sequence_data(Y_train)

                        # Shuffle data
                        X_train, Y_train = unison_shuffled_copies(X_train, Y_train)

                        predictions = build_lstm_predict(X_train, Y_train, X_dev, int(best_glove_model[2]))

                        evaluation_dict = get_evaluation_dict(predictions, Y_dev, tags)

                        full_evaluation_dict = get_evaluation_metrics(evaluation_dict, tags)

                        json.dump(full_evaluation_dict, open(('results\\' + file_name), 'w'))

                        print(file_name + ' has been saved to the results folder. (Accuracy: ' + str(full_evaluation_dict['accuracy']) + ')\n')

                    else:

                        print(file_name[:-5] + ' has already been evaluated.\n')





# 8)
# Compare the results
files_in_results_folder = os.listdir('results')
json_files = [file for file in files_in_results_folder if file[-5:] == '.json']


evaluated_models = []

for file in json_files:
    file_dict = json.load(open('results\\' + file))

    evaluated_models.append([file, 
        round(file_dict['macro_precision'], 3),
        round(file_dict['macro_recall'], 3),
        round(file_dict['macro_f1'], 3),
        round(file_dict['weighted_precision'], 3),
        round(file_dict['weighted_recall'], 3),
        round(file_dict['weighted_f1'], 3),
        round(file_dict['accuracy'], 3)])


header = ['model_name', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_precision', 'weighted_recall', 'weighted_f1', 'accuracy']


print(tabulate.tabulate(evaluated_models, headers=header))




print('\nCode succesfully finished!!!\n')
