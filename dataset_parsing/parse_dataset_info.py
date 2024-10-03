from datasets import load_dataset, list_datasets#, VerificationMode
import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
import pickle
import os
# import pickle5 as pickle
from tqdm import tqdm
from datasets.features.features import Value as datasets_feautures_Value, Sequence as datasets_feautures_Sequence, ClassLabel as datasets_feautures_ClassLabel
import signal

DATASET_INFO_DIR = 'dataset_info_new'
config_filename = 'configs_20240208.pkl'

def filter_text_classification(dataset):
    return 'task_categories:text-classification' in dataset.tags# and 'language:en' in dataset.tags and 'multilinguality:monolingual' in dataset.tags


def query(API_URL):
    response = requests.get(API_URL)#, headers=headers)
    return response.json()


async def get_dataset_infos(dataset_ids):
    # return [
    #     calculate_single_emission(*row_values) for row_values in zip(df['account_id'], df['direct_distance'], df['Volume_Value'], df['Weight_Value'], df['destination_cdp'], df['BounceProbability'], df['Returned_Value'], df['dimensions_shipment_carrier_name'])
    # ]
    emissions = []
    async with aiohttp.ClientSession() as session:

        
        tasks = [
            get_dataset_info(session, ds_id) for ds_id in dataset_ids
        ]

        try:
            json_dicts = await asyncio.gather(*tasks)#, return_exceptions=True)
        except Exception as e:
            print(e)
            
        # return [json_dict['dataset_info'] for json_dict in json_dicts]
        return dict(zip(dataset_ids, json_dicts))


async def get_dataset_info(session, dataset_id):
    API_URL = f"https://datasets-server.huggingface.co/info?dataset={dataset_id}"#?dataset=glue"#&config=SelfRC"

    async with session.get(API_URL) as response:
        return await response.json()


def convert_features_to_feature_list(features):
    feature_list = []
    for feature_name, feature_obj in features.items():
        if isinstance(feature_obj, datasets_feautures_Sequence):
            continue

        if isinstance(feature_obj, datasets_feautures_ClassLabel):
            dtype = 'class_label'
        else:
            dtype = str(feature_obj.dtype)

        feature_list.append({'name': feature_name, 'dtype': dtype})

        # if isinstance(feauture_obj, datasets_feautures_Sequence):
        #     feauture_list.append({'name': feature_name, 'dypte': feature_obj.feature.dtype})
        # else:
        #     feauture_list.append({'name': feature_name, 'dypte': feature_obj.dtype})
    return feature_list


# def load_dataset_with_subset(dataset_name, subset_name, split=None, verification_mode=VerificationMode.NO_CHECKS,
#                              streaming=True):
#     if subset_name is None:
#         return load_dataset(dataset_name, verification_mode=verification_mode, split=split, streaming=streaming)
#
#     return load_dataset(dataset_name, subset_name, verification_mode=verification_mode, split=split,
#                         streaming=streaming)

def load_dataset_with_subset(dataset_name, subset_name, split=None, ignore_verifications=True, streaming=True):
    if subset_name is None:
        return load_dataset(dataset_name, ignore_verifications=ignore_verifications, split=split, streaming=streaming)

    return load_dataset(dataset_name, subset_name, ignore_verifications=ignore_verifications, split=split,
                        streaming=streaming)



def get_feature_names(features, dtype_filter=None):
    feature_names = []
    # print(features)
    if dtype_filter:
        for feature_name, feature in features.items():
            # print(feature)
            if 'dtype' in feature:
                if dtype_filter in feature['dtype']: feature_names.append(feature_name)
            elif '_type' in feature:
                if dtype_filter in feature['_type']: feature_names.append(feature_name)

        return feature_names
        # return [feature['name'] for feature in features if feature.get('dtype') == dtype_filter]

    return list(features.keys())


def count_dtypes(features, dtype):
    count = 0
    for feature in features.values():
        # try:
        # if feature.get('dtype') == dtype:
        # if 'dtype' in feature:
        #    ( if dtype in feature.get('dtype'):
        # printfeature)
        if 'dtype' in feature:
            if dtype in feature.get('dtype'):
                count += 1
    # except:
    #     print(feature)
    return count


def contains_col_name(features, col_name):
    for feature in features.items():
        # print(feature)
        try:
            # if feature['name'] == col_name:
            if col_name in feature:
                return True
        except:
            return -1
    return False


def contains_col_names(features, col_names):
    return all(contains_col_name(features, col_name) for col_name in col_names)

def get_train_split(splits):
    if len(splits) == 1:
        return next(iter(splits))
    if 'train' in splits:
        return 'train'
    max_num_rows_split_name = None
    max_num_rows = -1
    for split_name, split_info in splits.items():
        num_rows = split_info['num_examples']
        # if num_rows == -1:
        #     print(splits)
        if num_rows > max_num_rows:
            max_num_rows_split_name = split_name
            max_num_rows = num_rows

    return max_num_rows_split_name


# TODO: Handle cases of multiple features with same min lens
def get_min_unqiues_feature(dataset, feature_names, num_samples=1000, seed=42):
    samples = list(dataset.shuffle(seed=seed, buffer_size=num_samples).take(num_samples))
    min_uniques_feature = None
    min_uniques = np.Inf
    for feature_name in feature_names:
        # if not isinstance(feature, dict):
        # if feature.dtype == dtype:
        # print(feature_name, feature)
        num_uniques = len(set([sample[feature_name] for sample in samples]))
        # num_uniques = len(set(dataset[feature_name]))
        # print(feature_name, num_uniques)
        if 1 < num_uniques < min_uniques:
            min_uniques_feature = feature_name
            min_uniques = num_uniques

    return min_uniques_feature, min_uniques
    # print(feature_name, np.mean(ds[f'{feature_name}_len']))


def get_min_classes_class_label_feature(features):
    min_classes_feature = None
    min_classes = np.Inf
    for feature_name, feature in features.items():
        if '_type' in feature:
            if feature['_type'] == 'ClassLabel':
                num_classes = len(feature['names'])
                # print(feature_name, num_classes)
                if 1 < num_classes < min_classes:
                    min_classes_feature = feature_name
                    min_classes = num_classes

    return min_classes_feature


def get_label_col(df_row, max_classes=60):
    for typical_label_col_name in ['label', 'target']:
        if typical_label_col_name in df_row['feature_names']:
            return typical_label_col_name

    if len(df_row['class_label_features']) == 1:
        return df_row['class_label_features'][0]

    label = get_min_classes_class_label_feature(df_row['features'])
    if label is not None:
        return label

    streaming_ds = load_dataset_with_subset(df_row['dataset_name'], df_row['subset'], split=df_row['train_split'])
    label, num_uniques = get_min_unqiues_feature(streaming_ds, feature_names=df_row['str_features'])

    if num_uniques <= max_classes:
        return label

    if len(row['float_features']) > 0:
        return row['float_features'][0]

    return None


def get_num_words(input_text):
    # if example[column] is None:
    #     return 0

    return len(input_text.split())


def get_max_avg_len_str_feature(dataset, feature_names, num_samples=1000, seed=42):
    samples = list(dataset.shuffle(seed=seed, buffer_size=num_samples).take(num_samples))
    max_avg_len_str_feature = None
    max_avg_len = -1
    for feature_name in feature_names:
        feature_avg_len = np.mean([get_num_words(sample[feature_name]) for sample in samples])
        # print(feature_name, feature_avg_len)
        if feature_avg_len > max_avg_len:
            max_avg_len_str_feature = feature_name
            max_avg_len = feature_avg_len

    return max_avg_len_str_feature


def get_input_col(df_row):
    if len(df_row['str_features']) == 1:
        return row['str_features'][0]

    for relevant_input_col in ['sentence', 'input', 'inputs', 'text']:
        if relevant_input_col in row['feature_names']:
            return relevant_input_col

    for relevant_input_col_tuple in [('sentence1', 'sentence2'), ('premise', 'hypothesis'), ('question1', 'question2')]:
        if all(relevant_input_col in row['feature_names'] for relevant_input_col in relevant_input_col_tuple):
            return relevant_input_col_tuple

    streaming_ds = load_dataset_with_subset(row['dataset_name'], row['subset'], split=row['train_split'])

    return get_max_avg_len_str_feature(streaming_ds, feature_names=row['str_features'])


# ds_list = list_datasets(with_details=True)
# classification_datasets = list(filter(filter_text_classification, ds_list))
# dataset_api_infos = await get_dataset_infos(classification_datasets)

# error_counter = 0
# all_configs = {}
# for dataset_entry, info_dict in dataset_api_infos.items():
#     try:
#         dataset_info = info_dict['dataset_info']
#         configs = dataset_info.keys()
#         for config in configs:
#             # configs_list.append(config)
#             # print(info_dict['dataset_info'][config]['task_templates'])
#             splits = dataset_info[config]['splits']
#             # num_examples = [value['num_examples'] for value in splits.values()]
#             # print(len(num_examples))
#             features = dataset_info[config]['features']
#             # configs_results.append((config, splits))
#             all_configs['_'.join([dataset_entry.id, config])] = (dataset_entry.id, config, splits, features, dataset_entry.downloads)
#     except Exception as e:
#         print(e)
#         error_counter += 1
#         # print(dataset_id)

def handle(signum, frame):
    raise Exception('Timout occured.')

def get_train_split_size(row):
    train_split_name = row['train_split']
    # train_split_name = row['splits']
    return row['splits'][train_split_name]['num_examples']




os.makedirs(DATASET_INFO_DIR, exist_ok=True)

if not os.path.isfile(config_filename):
    ds_list = list_datasets(with_details=True)

    classification_datasets = list(filter(filter_text_classification, ds_list))

    dataset_ids = [ds.id for ds in classification_datasets]
    downloads_by_id = {ds.id: ds.downloads for ds in classification_datasets}
    dataset_api_infos = asyncio.run(get_dataset_infos(dataset_ids))

    with open("dataset_api_infos_20240208.pkl", "wb") as f:
        pickle.dump(dataset_api_infos, f)

    error_counter = 0
    all_configs = {}
    for ds_id, info_dict in tqdm(dataset_api_infos.items()):
        # for dataset_entry, info_dict in filtered_api_info.items():
        try:
            dataset_info = info_dict['dataset_info']
            configs = dataset_info.keys()
            for config in configs:
                # configs_list.append(config)
                # print(info_dict['dataset_info'][config]['task_templates'])
                splits = dataset_info[config]['splits']
                # num_examples = [value['num_examples'] for value in splits.values()]
                # print(len(num_examples))
                features = dataset_info[config]['features']
                # configs_results.append((config, splits))
                all_configs['_'.join([ds_id, config])] = (
                ds_id, config, splits, features, downloads_by_id[ds_id])
        except Exception as e:
            # print(e)
            error_counter += 1
            # print(dataset_id)

    print(error_counter)

    df = pd.DataFrame.from_dict(all_configs, orient='index',
                                columns=['dataset_name', 'subset', 'splits', 'features', 'downloads']).sort_values(
        'downloads', ascending=False)


    # configs_all = {key: all_configs[key] for key in all_configs if all_configs[key][-1]>=1000}



    with open(config_filename, 'wb') as f:
        pickle.dump(all_configs, f)

with open(config_filename, 'rb') as f:
    all_configs = pickle.load(f)

df = pd.DataFrame.from_dict(all_configs, orient='index', columns=['dataset_name', 'subset', 'splits', 'features', 'downloads']).sort_values('downloads', ascending=False)
# with open('df_2000.pkl', 'rb') as f:
#     df = pickle.load(f)
# df = pd.read_pickle('df_2000.pkl')

df = df.loc[~df['splits'].isna()]

dataset_names_set = set(df['dataset_name'])
for dataset_name in dataset_names_set:
        if '/' in dataset_name:
            if dataset_name.split('/')[-1] in dataset_names_set:
                df.drop(df.loc[df['dataset_name']==dataset_name].index, inplace=True)

df['num_columns'] = df['features'].apply(len)

for col_name in ['text', 'sentence', 'input', 'inputs']:
    df['contains_'+col_name] = df['features'].apply(lambda x: contains_col_name(x, col_name))
    
col_names_pairs = [('sentence1', 'sentence2')]
for col_names_pair in col_names_pairs:
    df['contains_'+'_'.join(col_names_pair)] = df['features'].apply(lambda x: contains_col_names(x, col_names_pair))

df['feature_names'] = df['features'].apply(get_feature_names)
df['class_label_features'] = df['features'].apply(lambda x: get_feature_names(x, 'ClassLabel'))
df['str_features'] = df['features'].apply(lambda x: get_feature_names(x, 'string'))
df['float_features'] = df['features'].apply(lambda x: get_feature_names(x, 'float'))
df['train_split'] = df['splits'].apply(get_train_split)

dataset_info_dict = {}
unclear_cases = []
# skip_dataset_idx = ['declare-lab/HarmfulQA_default',
#                     'NikiTricky/digital-bg_default',
#                     'lorenzoscottb/PLANE-ood_default',
#                     'mithmith/road_commenst_default']
skip_dataset_idx = []

for _, idx in enumerate(tqdm(df.index)):
    if not (os.path.isfile(os.path.join(DATASET_INFO_DIR, f'{idx.replace("/", "_")}.pkl')) or
    os.path.isfile(os.path.join(DATASET_INFO_DIR, f'failed_{idx.replace("/", "_")}.pkl'))):# and 'mlsum' not in idx: # and idx not in skip_dataset_idx:
        print(idx)
        # idx = 'NikiTricky/digital-bg_default'

        try:
            if idx in skip_dataset_idx :
                raise Exception('Unknown error')

            # signal.signal(signal.SIGALRM, handle)
            # signal.alarm(180)
            row = df.loc[idx]
            valid_dataset = True

            ds_name = row['dataset_name']
            ds_subset = row['subset']
            # print(idx)
            label = get_label_col(row)
            # print(label)

            regression = label not in row['class_label_features'] and label not in row['str_features']

            # print(regression)

            input_col_name = get_input_col(row)

            # print(input())

            df.loc[idx, 'label'] = label
            df.loc[idx, 'input'] = str(input_col_name)
            df.loc[idx, 'regression'] = regression
            with open(os.path.join(DATASET_INFO_DIR, f'{idx.replace("/", "_")}.pkl'), 'wb') as f:
                pickle.dump((idx, label, str(input_col_name), regression), f)
        except Exception as e:
            print(e)
            df.loc[idx, 'exception'] = e
            with open(os.path.join(DATASET_INFO_DIR, f'failed_{idx.replace("/", "_")}.pkl'), 'wb') as f:
                pickle.dump(str(e), f)

    if os.path.isfile(os.path.join(DATASET_INFO_DIR, f'{idx.replace("/", "_")}.pkl')):
        with open(os.path.join(DATASET_INFO_DIR, f'{idx.replace("/", "_")}.pkl'), "rb") as f:
            _, label, input_col_name, regression = pickle.load(f)
        df.loc[idx, 'label'] = label
        df.loc[idx, 'input'] = input_col_name
        df.loc[idx, 'regression'] = regression

    # if os.path.isfile(os.path.join(DATASET_INFO_DIR, f'failed_{idx.replace("/", "_")}.pkl')):
    #     with open(os.path.join(DATASET_INFO_DIR, f'failed_{idx.replace("/", "_")}.pkl'), "rb") as f:
    #         e = pickle.load(f)
    #     df.loc[idx, 'exception'] = e


df.to_csv(os.path.join(DATASET_INFO_DIR, 'dataset_info_df_multilang_all.csv'))
df.to_pickle(os.path.join(DATASET_INFO_DIR, 'dataset_info_df_multilang_all.pkl'))

output_df = df.loc[(~df['label'].isna()) & (df['input'] != df['label'])]
output_df['train_split_size'] = output_df.apply(get_train_split_size, axis=1)

output_df = output_df[['dataset_name', 'subset', 'input', 'label', 'train_split', 'regression', 'train_split_size']].rename(columns={'dataset_name': 'name',
                                                                                                                 'train_split': 'splits',
                                                                                                                 'regression': 'task_type'})
output_df.index = output_df.index.str.replace('/', '__')
output_df['splits'] = output_df['splits'].apply(lambda x: {'train': x})
output_df['task_type'] = output_df['task_type'].map({True: 'regression', False: 'classification'})
# output_df['input'] = output_df['input'].apply(lambda x: exec(x) if "'" in x else x)
output_df['input'] = output_df['input'].apply(lambda x: tuple(x.strip("('").strip(")'").split("', '")) if "', '" in x else x)
output_df.to_csv(os.path.join(DATASET_INFO_DIR, 'output_df.csv'))
output_df.to_pickle(os.path.join(DATASET_INFO_DIR, 'output_df.pkl'))


dataset_info_dict = output_df.to_dict(orient='index')


with open('dataset_info_dict_20240210.pkl', 'wb') as f:
    pickle.dump(dataset_info_dict, f)