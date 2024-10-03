import json

with open('datasets_info.json', 'r') as f:
    dataset_info_dict = json.load(f)

duplicates = [
    'davanstrien__test_imdb_embedd2_default',
    'davanstrien__test_imdb_embedd_default',
    'davanstrien__test1_default',
    'shunk031__JGLUE_JSTS'
]

for ds in duplicates:
    del dataset_info_dict[ds]

dataset_info_dict['imdb_plain_text']['splits']['validation'] = 'test'
dataset_info_dict['tweet_eval_emotion']['splits']['validation'] = 'validation'
dataset_info_dict['tweet_eval_sentiment']['splits']['validation'] = 'validation'
dataset_info_dict['rotten_tomatoes_default']['splits']['validation'] = 'validation'
dataset_info_dict['llm-book__JGLUE_JSTS']['splits']['validation'] = 'validation'
dataset_info_dict['google_wellformed_query_default']['splits']['validation'] = 'validation'
dataset_info_dict['paws-x_en']['splits']['validation'] = 'validation'
dataset_info_dict['md_gender_bias_convai2_inferred']['splits']['validation'] = 'validation'
dataset_info_dict['google__civil_comments_default']['splits']['validation'] = 'validation'
