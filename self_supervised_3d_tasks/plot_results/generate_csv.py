import csv
import numpy as np

def write_result(base_path, row):
    with open(f'{base_path}/results.csv', "a") as csvfile:
        result_writer = csv.writer(csvfile, delimiter=",")
        result_writer.writerow(row)

def build_row(scores):
	row.append()


filename = '/Users/d070867/Desktop/results/finetuned_brats_01/down_03_up_03/full/results_full_majority'
base_path='/Users/d070867/Desktop/results/finetuned_brats_01/down_03_up_03'

with open(filename) as f:
    lines = f.readlines()

splits = ['5%', '10%', '25%', '50%', '100%']
#headers = ['Train Split', 'avg_dice', 'avg_dice_0', 'avg_dice_1', 'avg_dice_2']
#entries = ['dice', 'jaccard', 'dice_0', 'dice_1', 'dice_2']

headers = ['Train Split', 'avg_dice', 'avg_brats_wt', 'avg_brats_tc', 'avg_brats_et']
entries = ['dice', 'jaccard', 'brats_wt', 'brats_tc', 'brats_et']

repetetion_per_split = 3

write_result(base_path,headers)
line_index = 0

for split in splits:
	row = [split]
	repetions_scores = {
		entries[0]: [],
		entries[1]: [],
		entries[2]: [],
		entries[3]: [],
		entries[4]: []
	}

	while True:
		line = lines[line_index]

		if line.startswith('[('):
			parts = line.split(',')
			repetions_scores[entries[0]].append(float(parts[1][:-3]))
			repetions_scores[entries[1]].append(float(parts[3][:-3]))
			repetions_scores[entries[2]].append(float(parts[5][:-3]))
			repetions_scores[entries[3]].append(float(parts[7][:-3]))
			repetions_scores[entries[4]].append(float(parts[9][:-3]))

		line_index+=1

		if len(repetions_scores[entries[0]])==3: break

	row.append(np.average(repetions_scores[entries[0]]))
	row.append(np.average(repetions_scores[entries[2]]))
	row.append(np.average(repetions_scores[entries[3]]))
	row.append(np.average(repetions_scores[entries[4]]))

	write_result(base_path, row)


