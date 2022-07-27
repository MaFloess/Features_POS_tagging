# No imports



# Single class metrics ########################################################################################################################################
# Compute the precision for one class
def get_precision(eval_dict):

	return ((eval_dict['true_positive'] / (eval_dict['true_positive'] + eval_dict['false_positive'])) if 
		(eval_dict['true_positive'] + eval_dict['false_positive']) else 0)



# Compute the recall for one class
def get_recall(eval_dict):

	return ((eval_dict['true_positive'] / (eval_dict['true_positive'] + eval_dict['false_negative'])) if 
		(eval_dict['true_positive'] + eval_dict['false_negative']) else 0)



# Compute the F1 for one class
def get_f1(eval_dict):

	return ((eval_dict['true_positive'] / (eval_dict['true_positive'] + 0.5 * (eval_dict['false_positive'] + eval_dict['false_negative']))) if
		(eval_dict['true_positive'] + 0.5 * (eval_dict['false_positive'] + eval_dict['false_negative'])) else 0)



# Weighted overall metrics ####################################################################################################################################
# Compute macro precision
def get_macro_precision(eval_dict, tags):

	macro_precision = 0

	for key in tags:
		macro_precision += (eval_dict[key]['precision'] / len(tags))

	return macro_precision



# Compute macro recall
def get_macro_recall(eval_dict, tags):

	macro_recall = 0

	for key in tags:
		macro_recall += (eval_dict[key]['recall'] / len(tags))

	return macro_recall



# Compute macro F1
def get_macro_f1(eval_dict, tags):

	macro_f1 = 0

	for key in tags:
		macro_f1 += (eval_dict[key]['f1'] / len(tags))

	return macro_f1




# Weighted overall metrics ####################################################################################################################################
# Compute weighted precision
def get_weighted_precision(eval_dict, tags):

	weighted_precision = 0

	for key in tags:
		weighted_precision += ((eval_dict[key]['precision'] * (eval_dict[key]['true_positive'] + eval_dict[key]['false_negative'])) /
			(eval_dict[key]['true_positive'] + eval_dict[key]['false_negative'] + eval_dict[key]['true_negative'] + eval_dict[key]['false_positive']))

	return weighted_precision



# Compute weighted recall
def get_weighted_recall(eval_dict, tags):

	weighted_recall = 0

	for key in tags:
		weighted_recall += ((eval_dict[key]['recall'] * (eval_dict[key]['true_positive'] + eval_dict[key]['false_negative'])) /
			(eval_dict[key]['true_positive'] + eval_dict[key]['false_negative'] + eval_dict[key]['true_negative'] + eval_dict[key]['false_positive']))

	return weighted_recall



# Compute weighted F1
def get_weighted_f1(eval_dict, tags):

	weighted_f1 = 0

	for key in tags:
		weighted_f1 += ((eval_dict[key]['f1'] * (eval_dict[key]['true_positive'] + eval_dict[key]['false_negative'])) /
			(eval_dict[key]['true_positive'] + eval_dict[key]['false_negative'] + eval_dict[key]['true_negative'] + eval_dict[key]['false_positive']))

	return weighted_f1



# Overall accuracy ############################################################################################################################################
# Compute accuracy
def get_accuracy(eval_dict, tags):

	right_preds = 0
	overall_instances = 0

	for key in tags:
		right_preds += eval_dict[key]['true_positive']
		overall_instances += (eval_dict[key]['true_positive'] + eval_dict[key]['false_negative'])

	return (right_preds / overall_instances)



###############################################################################################################################################################
# Update a evaluation_dict in the form of the return value of the 'c04_02_get_evaluation_dict' function with metrics:
# Per class: precision, recall, F1
# Overall: macro-precision, macro-recall, macro-f1, weighted precision, weighted recall, weighted F1, accuracy
def get_evaluation_metrics(evaluation_dict, tags):

	for tag in tags:
		evaluation_dict[tag]['precision'] = get_precision(evaluation_dict[tag])
		evaluation_dict[tag]['recall'] = get_recall(evaluation_dict[tag])
		evaluation_dict[tag]['f1'] = get_f1(evaluation_dict[tag])

	evaluation_dict['macro_precision'] = get_macro_precision(evaluation_dict, tags)
	evaluation_dict['macro_recall'] = get_macro_recall(evaluation_dict, tags)
	evaluation_dict['macro_f1'] = get_macro_f1(evaluation_dict, tags)

	evaluation_dict['weighted_precision'] = get_weighted_precision(evaluation_dict, tags)
	evaluation_dict['weighted_recall'] = get_weighted_recall(evaluation_dict, tags)
	evaluation_dict['weighted_f1'] = get_weighted_f1(evaluation_dict, tags)

	evaluation_dict['accuracy'] = get_accuracy(evaluation_dict, tags)

	return evaluation_dict