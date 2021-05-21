from transformers import XLMRobertaTokenizer
import numpy as np
import torch

#Load the XLMR_TOKER
XLMR_TOKER = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
with open('object_labels/img_label_objects.txt', "r") as f:
    IMG_LABEL_OBJECTS = f.readlines()
IMG_LABEL_OBJECTS = ['background'] + [x.strip() for x in IMG_LABEL_OBJECTS]

#initialize LABEL2TOKEN_MATRIX
VALID_XLMR_TOKEN_IDS = []
#LABEL2TOKEN = {}
LABEL2TOKEN_MATRIX = np.zeros((1601, XLMR_TOKER.vocab_size))
for i,label_word in enumerate(IMG_LABEL_OBJECTS):
    label_tokens = XLMR_TOKER.tokenize(label_word)
    label_tokens_ids = [XLMR_TOKER._convert_token_to_id(w) for w in label_tokens]
    #LABEL2TOKEN[label_word] = label_tokens_ids
    LABEL2TOKEN_MATRIX[i][label_tokens_ids] = 1
    VALID_XLMR_TOKEN_IDS.extend(label_tokens_ids)
#torch.tensor(LABEL2TOKEN_MATRIX)
VALID_XLMR_TOKEN_IDS = list(set(VALID_XLMR_TOKEN_IDS))
VALID_XLMR_TOKEN_IDS.sort()
