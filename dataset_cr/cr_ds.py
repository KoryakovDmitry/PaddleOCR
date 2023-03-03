import json
import os
import os.path as osp
from glob import glob
from sklearn.model_selection import train_test_split
import json
import shutil

import shutil
from tqdm import tqdm
from glob import glob
import itertools

dir2imgs = "/Users/dmitry/PaddleOCR/dataset_cr/dataset_processed/images/"
# os.makedirs(dir2imgs, exist_ok=True)
#
# for i in tqdm(glob("dataset_cr/dataset/dataset_all/*/images/*")):
#     shutil.copy(i, dir2imgs)

with open(
        "dataset_cr/dataset/dataset_ksdr_ydx_v3_resized_bboxes_77_rec_synth_wiki_tc_v4_as_pretrain_gcloud_intrs_86_v4.json",
        "r", encoding="utf8") as f:
    dataset = json.load(f)

dataset = dict(itertools.islice(dataset.items(), 9))

train, test = train_test_split(list(dataset.keys()), test_size=0.5, random_state=42)
test, test_ = train_test_split(test, test_size=0.5, random_state=42)
len(train), len(test), len(test_)

with open('dataset_cr/dataset_processed/train.txt', 'w', encoding='utf8') as f:
    for t in train:
        # print(f"train/{t}\t{dataset[t]}\n")
        f.write(f"{t}\t{json.dumps(dataset[t], ensure_ascii=False)}\n")

with open('dataset_cr/dataset_processed/test.txt', 'w', encoding='utf8') as f:
    for t in test:
        # print(f"test/{t}\t{dataset[t]}\n")
        f.write(f"{t}\t{json.dumps(dataset[t], ensure_ascii=False)}\n")

with open('dataset_cr/dataset_processed/test_.txt', 'w', encoding='utf8') as f:
    for t in test_:
        # print(f"test/{t}\t{dataset[t]}\n")
        f.write(f"{t}\t{json.dumps(dataset[t], ensure_ascii=False)}\n")
