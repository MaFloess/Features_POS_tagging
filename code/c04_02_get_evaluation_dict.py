# Imports
import numpy as np



# Build a evaluation dictionary with tp, tn, fp, fn
# Based on predictions (in the return form of c04_01) and the respective gold standard
def get_evaluation_dict(predictions, Y_dev, tags):


	# Create dictionary layout
	evaluation_dict = dict()

	for tag in tags:
		evaluation_dict[tag] = dict(true_positive=0, false_positive=0, true_negative=0, false_negative=0)



	# Fill the dictionary according to predictions and gold standard
	for sent_count in range(len(Y_dev)):
		for word_count in range(len(Y_dev[sent_count])):
			if (Y_dev[sent_count][word_count] == predictions[sent_count][word_count]).all():
				evaluation_dict[tags[np.argmax(Y_dev[sent_count][word_count])]]['true_positive'] += 1

				temp_tags = tags.copy()
				temp_tags.pop(np.argmax(Y_dev[sent_count][word_count]))

				for tag in temp_tags:
					evaluation_dict[tag]['true_negative'] += 1

			else:
				evaluation_dict[tags[np.argmax(Y_dev[sent_count][word_count])]]['false_negative'] += 1
				evaluation_dict[tags[np.argmax(predictions[sent_count][word_count])]]['false_positive'] += 1

				temp_tags1 = tags.copy()
				temp_tags1.pop(np.argmax(Y_dev[sent_count][word_count]))

				temp_tags2 = tags.copy()
				temp_tags2.pop(np.argmax(predictions[sent_count][word_count]))

				temp_tags3 = [tag for tag in temp_tags1 if tag in temp_tags2]

				for tag in temp_tags3:
					evaluation_dict[tag]['true_negative'] += 1


	return evaluation_dict