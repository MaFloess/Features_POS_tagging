# A Comparative Evaluation of the Utility of Linguistic Features for Part-of-Speech-Tagging 
## Bachelor's thesis at the Department of Statistics, LMU

This thesis explores the utility of features that can be extracted from raw text for the task of Part-of-Speech-tagging by comparing the performance of Long-Short-Term-Memory Neural Networks.
Three frameworks of word embeddings were used to encode the word identity - Word2Vec (self-trained), FastText (self-trained), GloVe (pre-trained).
The word identity serves as the baseline information the POS-tagging model (LSTM Neural Network), incorporates. 
Additionally four feature groups may be incorporated in the input space.
There is a group of features related with the character-classes which are contained in a word, a group of features dealing with case of characters that constitute a word, 
a group of features related to the position words have in their respective sentence and a group of features related to affixes that may be present in a word.


The approach was to find the best performing word embeddings on the task of Part-of-Speech-tagging first.
Afterwards, the feature groups have been added in all possible combinations to the input space of the POS-tagging models with the best performing word embedding respective to each framework.
All these models have been evaluated thouroghly by the creation of evaluation dictionaries that incorporate the metrics precision, recall and F1-score at the level of individual tags. 
Overarching the individual tags the accuracy, weighted precision, weighted recall, weighted F1-score, macro precision, macro recall, macro F1-score are computed and contained in the dictionary.
By comparing the results saved in these dictionaries the utility of the incorporated/omitted features groups was evaluated.


## Project's folder structure


```bash
    .
    ├── code
    │   ├── c01_get_tagged_sents_conllu.py      # Extract data from CoNLL-U format 
    │   ├── c02_01_get_pref_suf_dicts.py        # Get information on affixes in the training data
    │   ├── c02_02_encode_data.py               # Encode features and labels
    │   ├── c02_03_pretraining_functions.py     # Padding and shuffle function 
    │   ├── c03_01_generate_word2vec_model.py   # Create and save Word2Vec word embeddings
    │   ├── c03_02_generate_fasttext_model.py   # Create and save FastText word embeddings 
    │   ├── c04_01_build_lstm_predict.py        # Predict with a POS-tagging model and a certain input space
    │   ├── c04_02_get_evaluation_dict.py       # Get the true/false positives/negatives for the prediction
    │   ├── c04_03_get_evaluation_metrics.py    # Create the full evaluation dictionary for a model
    │   ├── main.py                             # Full workflow of thesis with verbose output
    │   └── requirements.txt                    # List of necessary python packages
    ├── data
    │   ├── en_gum-ud-train.conllu              # Training data
    │   ├── en_gum-ud-dev.conllu                # Development data
    │   ├── en_gum-ud-test.conllu               # Test data
    │   └── word_embeddings                     # Includes pre-trained and self-trained, once created, word e.
    ├── results
    │   ├── w2v_lstm_evaluation.csv             # Performance of 432 POS-taggers with 216 self-trained W2V
    │   ├── fasttext_lstm_evaluation.csv        # Performance of 10 POS-taggers with 10 self-trained FastText    
    │   ├── glove_lstm_evaluation.csv           # Performance of 6 POS-taggers with 3 pre-trained GloVe      
    │   └── ...                                 # 93 evaluation dictionaries in the JSON format          
    └── thesis
        ├── chapters                            # Folder of chapters for thesis
        ├── Pictures                            # Folder of pictures for thesis
        ├── thesis.tex                          # Bibliography of thesis
        ├── thesis.tex                          # Main LaTex code of thesis (requires the 3 items above)
        └── A_Comparative_Evaluation_of_the_Utility_of_Linguistic_Features_for_Part_of_Speech_Tagging.pdf

``` 

## Naming conventions

Created word embeddings are named by the convention of first naming the framework followed by all parameters that were regulated in this thesis separated by underscores. 
Word embeddings are saved in the 'data\word_embeddings' folder. 
The exception is the GloVe framework were a word embedding is referenced by 'glove.6B.' and its vector size to maintain the original naming convention of the pre-trained word embeddings.

If a POS-tagging model with word embedding as its sole input is referenced (as in the CSV files in the 'results' folder), the same naming convention as 
discussed above is expanded by an underscore and the used number of units in the model's hidden layer.

The thorough model evaluations (JSON files in the 'results' folder) are named by the same approach as mentioned above with addtional zeros or ones separated by underscores depending 
on whether the respective feature group was incorporated.

Examples:

```bash
    
    w2v_3_2_200_045_sg (word embedding)
    ├── w2v                             # Word2Vec framework 
    ├── 3                               # Minimal word count of 3
    ├── 2                               # Window size of 2
    ├── 200                             # Vector size of 200
    ├── 045                             # Alpha of 0.045                          
    └── sg                              # Skipgram architecture

    ft_3_2_200_045_sg_2_5_125 (POS-tagging model with word embedding as sole input)
    ├── ft                              # FastText framework 
    ├── 3                               # Minimal word count of 3
    ├── 2                               # Window size of 2
    ├── 200                             # Vector size of 200
    ├── 045                             # Alpha of 0.045                          
    ├── sg                              # Skipgram architecture
    ├── 2                               # Lower bound of character ngrams of 2
    ├── 5                               # Upper bound of character ngrams of 5
    └── 125                             # 125 units in the hidden layer of the model

    glove.6B.200_125_1_0_1_0.json (POS-tagging model evaluation dictionary)
    ├── glove.6B.                       # GloVe framework's pre-trained word embedding
    ├── 200                             # Vector size of 200
    ├── 125                             # 125 units in the hidden layer of the model                          
    ├── 1                               # Character related features incorporated
    ├── 0                               # Case related features omitted
    ├── 1                               # Sentence position related features incorporated
    └── 0                               # Affix related features omitted

``` 
