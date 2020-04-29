from os import path as osp
import json
import os
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='saved_models/test_epochs_17.json')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--dataroot', type=str, default='data/vqacp2/annotations/')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

	args = parse_args()

	anno_path = osp.join(args.dataroot, '%s_target_count.pth'%(args.name))
	annotations = torch.load(anno_path)

	annotations = sorted(annotations, key=lambda x: x['question_id'])
	predictions = sorted(json.load(open(args.input)), key=lambda x: x['question_id'])

	score = 0
	count = 0
	other_score = 0
	yes_no_score = 0
	num_score = 0
	yes_count = 0
	other_count = 0
	num_count = 0
	upper_bound = 0
	upper_bound_num = 0
	upper_bound_yes_no = 0
	upper_bound_other = 0

	for pred, anno in zip(predictions, annotations):
		if pred['question_id'] == anno['question_id']:
			G_T= max(anno['answer_count'].values())
			upper_bound += min(1, G_T / 3)
			if pred['answer'] in anno['answers_word']:
				proba = anno['answer_count'][pred['answer']]
				score += min(1, proba / 3)
				count +=1
				if anno['answer_type'] == 'yes/no':
					yes_no_score += min(1, proba / 3)
					upper_bound_yes_no += min(1, G_T / 3)
					yes_count +=1
				if anno['answer_type'] == 'other':
					other_score += min(1, proba / 3)
					upper_bound_other += min(1, G_T / 3)
					other_count +=1
				if anno['answer_type'] == 'number':
					num_score += min(1, proba / 3)
					upper_bound_num += min(1, G_T / 3)
					num_count +=1
			else:
				score += 0
				yes_no_score +=0
				other_score +=0
				num_score +=0
				if anno['answer_type'] == 'yes/no':
					upper_bound_yes_no += min(1, G_T / 3)
					yes_count +=1
				if anno['answer_type'] == 'other':
					upper_bound_other += min(1, G_T / 3)
					other_count +=1
				if anno['answer_type'] == 'number':
					upper_bound_num += min(1, G_T / 3)
					num_count +=1


	print('count:', count, ' score:', round(score*100/len(annotations),2))
	print('Yes/No:', round(100*yes_no_score/yes_count,2), 'Num:', round(100*num_score/num_count,2),
		  'other:', round(100*other_score/other_count,2))

	print('count:', len(annotations), ' upper_bound:', round(score*upper_bound/len(annotations)),2)
	print('upper_bound_Yes/No:', round(100*upper_bound_yes_no/yes_count,2), 'upper_bound_Num:',
		  round(100 * upper_bound_num/num_count,2), 'upper_bound_other:', round(100*upper_bound_other/other_count,2))

