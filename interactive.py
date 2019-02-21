from models import *
from preprocess import Preprocessor
import sys
import csv
import pandas as pd
import numpy as np

saved_model = './weights/adem_model.pkl'
if __name__ == '__main__':
	pp = Preprocessor()
	adem = ADEM(pp, None, saved_model)
	print('Model Loaded!')


	# contexts = ['</s> <first_speaker> hello . how are you today ? </s>',
	# 			'</s> <first_speaker> i love starbucks coffee </s>']
	# true = ['</s> <second_speaker> i am fine . thanks </s>',
	# 		'</s> <second_speaker> i like their latte </s>']
	# model = ['</s> <second_speaker> fantastic ! how are you ? </s>',
	# 		'</s> <second_speaker> me too ! better than timmies </s>']

	responses = pd.read_csv("baselines_test_set.csv")

	model_response_csv_column_titles = [
		"Model_1",
		"Model_2",
		"Model_3",
		"Model_4",
		"Model_5",
		"Model_6",
	]

	contexts = list(responses["Query"])
	true = list(responses["Model_1"])
	for model_name in model_response_csv_column_titles:
		model_replies = list(responses[model_name])
		
		scores = adem.get_scores(contexts, true, model_replies)
		
		score_col_name = model_name + "_score"
		
		updated_df = pd.DataFrame({score_col_name: scores})
		responses.update(updated_df)

	file_name = "baselines_test_set_with_ADEM_scores" + time.time() + ".csv"
		
	responses.to_csv(file_name)
	print("Wrote results to " + file_name)
	


