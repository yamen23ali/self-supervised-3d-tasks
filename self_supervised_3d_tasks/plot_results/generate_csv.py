import csv
import numpy as np

def write_result(base_path, row):
    with open(f'{base_path}/results.csv', "a") as csvfile:
        result_writer = csv.writer(csvfile, delimiter=",")
        result_writer.writerow(row)

def build_row(scores):
	row.append()


filename = '/Users/d070867/Desktop/Thesis/code/self-supervised-3d-tasks/results_full_baysien'
base_path='/Users/d070867/Desktop/Thesis/code/self-supervised-3d-tasks'

with open(filename) as f:
    lines = f.readlines()

splits = ['5%', '10%', '25%', '50%', '100%']
headers = ['Train Split', 'avg_dice', 'avg_dice_0', 'avg_dice_1', 'avg_dice_2']
repetetion_per_split = 3

write_result(base_path,headers)
line_index = 0

for split in splits:
	row = [split]
	repetions_scores = {
		'dice': [], 
		'jaccard': [],
		'dice_0': [],
		'dice_1': [],
		'dice_2': []
	}

	while True:
		line = lines[line_index]
		
		if line.startswith('[('):
			parts = line.split(',')
			repetions_scores['dice'].append(float(parts[1][:-3]))
			repetions_scores['jaccard'].append(float(parts[3][:-3]))
			repetions_scores['dice_0'].append(float(parts[5][:-3]))
			repetions_scores['dice_1'].append(float(parts[7][:-3]))
			repetions_scores['dice_2'].append(float(parts[9][:-3]))
		
		line_index+=1

		if len(repetions_scores['dice'])==3: break
	
	row.append(np.average(repetions_scores['dice']))
	row.append(np.average(repetions_scores['dice_0']))
	row.append(np.average(repetions_scores['dice_1']))
	row.append(np.average(repetions_scores['dice_2']))

	write_result(base_path, row)
	

	