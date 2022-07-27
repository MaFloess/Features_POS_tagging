# Imports
import gensim


# Generate FastText model and save it
def generate_save_fasttext(sentences, path='data\\word_embeddings\\', 
	min_count=1, window=2, vector_size=50, alpha=0.015, epochs=5, implementation='cbow', min_n=2, max_n=5):

	
	if implementation == 'cbow': 
		sg = 0
	else:
		sg = 1


	model = gensim.models.FastText(
		sentences,
		min_count=min_count,
		window=window,
		vector_size=vector_size,
		alpha=alpha,
		min_alpha=alpha/epochs,
		epochs=epochs,
		sg=sg,
		min_n=min_n,
		max_n=max_n 
		)


	path = (path + 'fasttext_' + str(min_count) + '_' + str(window) + '_' + str(vector_size) + '_' + str(alpha)[2:] + '_' +
		implementation + '_' + str(min_n) + '_' + str(max_n))

	word_vectors = model.wv
	word_vectors.save(path)
