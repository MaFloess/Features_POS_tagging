# Imports
import gensim


# Generate Word2Vec model and save it
def generate_save_word2vec(sentences, path='data\\word_embeddings\\', 
	min_count=1, window=2, vector_size=50, alpha=0.015, epochs=5, implementation='cbow'):

	
	if implementation == 'cbow': 
		sg = 0
	else:
		sg = 1


	model = gensim.models.Word2Vec(
		sentences,
		min_count=min_count,
		window=window,
		vector_size=vector_size,
		alpha=alpha,
		min_alpha=alpha/epochs,
		epochs=epochs,
		sg=sg 
		)


	path = path + 'w2v_' + str(min_count) + '_' + str(window) + '_' + str(vector_size) + '_' + str(alpha)[2:] + '_' + implementation

	word_vectors = model.wv
	word_vectors.save(path)
