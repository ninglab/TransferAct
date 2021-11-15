
def save_scores(scores, attn_path):

	with open(attn_path, 'a') as f:
		txt = ','.join(map(str, ['{:.2e}'.format(_) for _ in scores.flatten().tolist()]))
		print(txt, file=f)

