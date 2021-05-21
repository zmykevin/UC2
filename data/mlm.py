"""
MLM datasets
"""
import math
import random

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip

from .data import (DetectFeatTxtTokDataset, DetectFeatTxtTokDatasetCutDown, TxtTokLmdb,
                   get_ids_and_lens, pad_tensors, get_gather_index)
from .mrm import _mask_img_feat, _get_targets
from transformers import XLMRobertaTokenizer, BertTokenizer
#from pytorch_pretrained_bert import BertTokenizer
import numpy as np
from model.const_variable import LABEL2TOKEN_MATRIX, VALID_XLMR_TOKEN_IDS
#import h5py
#Load the XLMR_TOKER
XLMR_TOKER = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
UNITER_TOKER = BertTokenizer.from_pretrained('bert-base-cased')

#Load the IMG_LABEL_OBJECTS
with open('object_labels/img_label_objects.txt', "r") as f:
    IMG_LABEL_OBJECTS = f.readlines()
IMG_LABEL_OBJECTS = [x.strip() for x in IMG_LABEL_OBJECTS]
#initialize LABEL2TOKEN_MATRIX

def random_word(tokens, vocab_range, mask):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = mask

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(range(*vocab_range)))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        tokens[0] = mask

    return tokens, output_label

##############################Mingyang Zhou#########################################
def _get_targets(img_masks, img_soft_label):
    soft_label_dim = img_soft_label.size(-1)
    img_masks_ext_for_label = img_masks.unsqueeze(-1).expand_as(img_soft_label)
    label_targets = img_soft_label[img_masks_ext_for_label].contiguous().view(
        -1, soft_label_dim)
    return label_targets

def comasking_token_all(i_lang1, word2ids_lang1, word2ids_lang2, lang1_lang2_walign, len_tokens_lang1):
    #find the corresponding word 
    masked_i_lang1 = None
    masked_i_lang2 = None
    lang1_w_idx = 0
    for w_idx, w in enumerate(word2ids_lang1):
        if i_lang1 in w:
            #find the corresponding
            lang1_w_idx = w_idx
            masked_i_lang1 = w.copy() #Include all the tokens to be masked from English
            break
    
    assert masked_i_lang1 is not None, "masked_i_lang1 should not be empty"
    #find the corresponding german word
    lang2_w_idx = lang1_lang2_walign.get(lang1_w_idx, None)
    
    #find the corresponding german token
    if lang2_w_idx is None:
        lang2_w_idx = [0]
        
    masked_i_lang2 = word2ids_lang2[lang2_w_idx[0]].copy()
    
    return masked_i_lang1, masked_i_lang2

def random_word_dmasking_all(example, vocab_range, mask):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper. However, we only mask the English part of the concatenated sentence and then do comasking
        on German part
    :param example: datapoint from txtdb.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    tokens = example['input_ids']
    output_label = [-1]*len(tokens)
    
    len_tokens_lang1 = len(example['input_ids_lang1'])
    word2ids_lang1 = example['word2ids_lang1']
    #print("word2ids_lang1 is: {}".format(word2ids_lang1))
    word2ids_lang2 = example['word2ids_lang2']
    #print("word2ids_lang2 is: {}".format(word2ids_lang2))
    lang1_lang2_walign = example['lang1_lang2_walign']
    #print("lang1_lang2_walign is: {}".format(lang1_lang2_walign))
    for i, token in enumerate(tokens):
        if i >= len_tokens_lang1:
            break
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15 and i < len_tokens_lang1:
            #find the corresponding word, let's mask all corresponding words
            masked_lang1_ids, masked_lang2_ids = comasking_token_all(i, word2ids_lang1, word2ids_lang2, lang1_lang2_walign, len_tokens_lang1)
            #Get the masked_lang1_tokens
            masked_lang1_tokens = [tokens[j] for j in masked_lang1_ids]
            masked_lang2_tokens = [tokens[j+len_tokens_lang1+2] for j in masked_lang2_ids]
            # print(masked_lang1_ids)
            # print(masked_lang2_ids)
            
            
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                for j in masked_lang1_ids:
                    tokens[j] = mask
                for j in masked_lang2_ids:
                    tokens[j+len_tokens_lang1+2] = mask

            # 10% randomly change token to random token
            elif prob < 0.9:
                for j in masked_lang1_ids:
                    tokens[j] = random.choice(list(range(*vocab_range)))
                for j in masked_lang2_ids:
                    tokens[j+len_tokens_lang1+2] = random.choice(list(range(*vocab_range)))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            for j,j_token in zip(masked_lang1_ids,masked_lang1_tokens):
                output_label[j] = j_token
            for j,j_token in zip(masked_lang2_ids,masked_lang2_tokens):
                output_label[j+len_tokens_lang1+2] = j_token 
            
            # print(output_label)
            # return
    if all(o == -1 for o in output_label):
        # at least mask 1
        masked_lang1_ids, masked_lang2_ids = comasking_token_all(0, word2ids_lang1, word2ids_lang2, lang1_lang2_walign, len_tokens_lang1)
        masked_lang1_tokens = [tokens[j] for j in masked_lang1_ids]
        masked_lang2_tokens = [tokens[j+len_tokens_lang1+2] for j in masked_lang2_ids]
        
        #Adjust the token to the mask_id
        for j in masked_lang1_ids:
            tokens[j] = mask
        for j in masked_lang2_ids:
            tokens[j+len_tokens_lang1+2] = mask
        
        #update the output_label
        for j,j_token in zip(masked_lang1_ids,masked_lang1_tokens):
            output_label[j] = j_token
        for j,j_token in zip(masked_lang2_ids,masked_lang2_tokens):
            output_label[j+len_tokens_lang1+2] = j_token 
        

    return tokens, output_label

def comasking_token(i_lang1, word2ids_lang1, word2ids_lang2, lang1_lang2_walign, len_tokens_lang1):
    #find the corresponding word 
    lang1_w_idx = 0
    for w_idx, w in enumerate(word2ids_lang1):
        if i_lang1 in w:
            #find the corresponding
            lang1_w_idx = w_idx
            break
    #find the corresponding german word
    lang2_w_idx = lang1_lang2_walign.get(lang1_w_idx, None)
    #find the corresponding german token
    if lang2_w_idx is None:
        i_lang2 = 0
    else:
        #find the corresponding 
        random_index = random.randrange(0, len(word2ids_lang2[lang2_w_idx[0]]),1)
        i_lang2 = word2ids_lang2[lang2_w_idx[0]][random_index] #current strategy we will only mask the first sub segment of the first aligned word
    return i_lang2

def random_word_dmasking(example, vocab_range, mask):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper. However, we only mask the English part of the concatenated sentence and then do comasking
        on German part
    :param example: datapoint from txt_db.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    tokens = example['input_ids']
    output_label = [-1]*len(tokens)
    len_tokens_lang1 = len(example['input_ids_lang1'])
    word2ids_lang1 = example['word2ids_lang1']
    #print("word2ids_lang1 is: {}".format(word2ids_lang1))
    word2ids_lang2 = example['word2ids_lang2']
    #print("word2ids_lang2 is: {}".format(word2ids_lang2))
    lang1_lang2_walign = example['lang1_lang2_walign']
    #print("lang1_lang2_walign is: {}".format(lang1_lang2_walign))
    for i, token in enumerate(tokens):
        if i >= len_tokens_lang1:
            break
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15 and i < len_tokens_lang1:
            #find the corresponding word 
            i_lang2 = comasking_token(i, word2ids_lang1, word2ids_lang2, lang1_lang2_walign, len_tokens_lang1)
            token_lang2 = tokens[i_lang2+len_tokens_lang1+2]
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = mask
                tokens[i_lang2+len_tokens_lang1+2] = mask

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(range(*vocab_range)))
                tokens[i_lang2+len_tokens_lang1+2] = random.choice(list(range(*vocab_range)))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            output_label[i] = token
            output_label[i_lang2+len_tokens_lang1+2] = token_lang2
    
    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        tokens[0] = mask
        
        #mask the corresponding german word
        i_lang2 = comasking_token(0, word2ids_lang1, word2ids_lang2, lang1_lang2_walign, len_tokens_lang1)
        output_label[i_lang2+len_tokens_lang1+2] = tokens[i_lang2+len_tokens_lang1+2]
        tokens[i_lang2+len_tokens_lang1+2] = mask

    return tokens, output_label


# get the corresponding img_mask & img_label_tokens
def _get_img_mask_mmxlm(mask_prob, num_bb, img_soft_labels, language='en', background_index=956):
    img_label_objects_path = "object_labels/img_label_objects.txt" if language == "en" else "object_labels/img_label_objects_{}.txt".format(language)
    with open(img_label_objects_path, "r") as f:
        IMG_LABEL_OBJECTS = f.readlines()
    IMG_LABEL_OBJECTS = [x.strip() for x in IMG_LABEL_OBJECTS]
    img_mask = []
    img_txt_labels = []
    for i in range(num_bb):
        if random.random() < mask_prob:
            img_mask.append(True)
            #check the corresponding arg_max
            top1_label = np.argmax(img_soft_labels[i])
            if top1_label == 0:
                #top1_label_text = "background"
                top1_label_text = IMG_LABEL_OBJECTS[background_index]
            else:
                top1_label_text = IMG_LABEL_OBJECTS[top1_label-1]
            #tokenize the top1_label_text into id
            top1_label_tokens = XLMR_TOKER._tokenize(top1_label_text)
            top1_label_token_id = XLMR_TOKER._convert_token_to_id(random.choice(top1_label_tokens)) #random take the token
            #Try using Random Samplling
            img_txt_labels.append(top1_label_token_id)
        else:
            img_mask.append(False)
            img_txt_labels.append(-1)
        
    if not any(img_mask):
        # at least mask 1
        sampled_index = random.choice(range(num_bb))
        img_mask[sampled_index] = True
        
        #Get the corresponding id
        top1_label = np.argmax(img_soft_labels[sampled_index])
        if top1_label == 0:
            top1_label_text = IMG_LABEL_OBJECTS[background_index]
        else:
            top1_label_text = IMG_LABEL_OBJECTS[top1_label-1]
        #tokenize the top1_label_text into id
        top1_label_tokens = XLMR_TOKER._tokenize(top1_label_text)
        #top1_label_token_id = XLMR_TOKER._convert_token_to_id(top1_label_tokens[0])
        top1_label_token_id = XLMR_TOKER._convert_token_to_id(random.choice(top1_label_tokens)) #Minor Change
        #update img_txt_labels
        img_txt_labels[sampled_index] = top1_label_token_id
        
    img_mask = torch.tensor(img_mask)
    img_txt_labels = torch.tensor(img_txt_labels)
    return img_mask, img_txt_labels


def _get_img_mask(mask_prob, num_bb):
    img_mask = [random.random() < mask_prob for _ in range(num_bb)]
    if not any(img_mask):
        # at least mask 1
        img_mask[random.choice(range(num_bb))] = True
    img_mask = torch.tensor(img_mask)
    return img_mask

def _get_img_mask_mmxlm_softlabel(mask_prob, num_bb, img_soft_labels, topk=5):
    img_mask = []
    img_token_soft_labels = np.zeros((num_bb,XLMR_TOKER.vocab_size))
    #print(img_token_soft_labels.shape)
    for i in range(num_bb):
        if random.random() < mask_prob:
            img_mask.append(True)   
#             topk_labels = np.argsort(img_soft_labels[i])[-topk:] #[topk]
#             topk_labels_distribution = np.sort(img_soft_labels[i])[-topk:]#[topk]
#             for label,label_prob in zip(topk_labels,topk_labels_distribution):
#                 label_word = "background" if label == 0 else IMG_LABEL_OBJECTS[label-1]
#                 label_tokens = XLMR_TOKER.tokenize(label_word)
#                 label_tokens_ids = [XLMR_TOKER._convert_token_to_id(w) for w in label_tokens]
#                 #normalize
#                 img_token_soft_labels[i][label_tokens_ids] = label_prob
            img_token_soft_labels[i] = np.matmul(img_soft_labels[i][np.newaxis,:], LABEL2TOKEN_MATRIX)[0]
            img_token_soft_labels[i] = img_token_soft_labels[i]/np.sum(img_token_soft_labels[i])
            
        else:
            img_mask.append(False)
    
    #convert img_token_soft_labels to tensor
    img_mask = torch.tensor(img_mask)
    img_token_soft_labels = torch.tensor(img_token_soft_labels, dtype=torch.float32)
    #print(img_token_soft_labels.type())
            
    return img_mask, img_token_soft_labels
    
def _get_txt_mask_mmxlm_softlabel(caption_txt_labels):
    #get txt_mask
    txt_mask = [x != -1 for x in caption_txt_labels]
    #get the txt_token_soft_labels
    txt_token_soft_labels = np.zeros((len(caption_txt_labels), XLMR_TOKER.vocab_size))
    for i,x in enumerate(caption_txt_labels):
        if x != -1:
            txt_token_soft_labels[i][x] = 1
    txt_token_soft_labels = torch.tensor(txt_token_soft_labels,dtype=torch.float32)
    txt_mask = torch.tensor(txt_mask)
    return txt_mask, txt_token_soft_labels



class MlmDataset_Dmasking(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, comasking_mode="random", text_only=False):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db)
        self.position_padding = 1
        self.comasking_mode = comasking_mode
        self.text_only = text_only
    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(i)

        # text input
        #input_ids, txt_labels = self.create_mlm_io(example['input_ids'])
        input_ids, txt_labels, position_ids = self.create_mlm_io(example)
        #print(input_ids)
        #print(txt_labels)
        #return
        # img input
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])
        if not self.text_only:
            attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        else:
            attn_masks = torch.ones(len(input_ids), dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, txt_labels, position_ids

    def create_mlm_io(self, example):
        #use random_word_dmasking
        if self.comasking_mode == "random":
            input_ids, txt_labels = random_word_dmasking(example,
                                                self.txt_db.v_range,
                                                self.txt_db.mask)
        elif self.comasking_mode == "full":
            input_ids, txt_labels = random_word_dmasking_all(example,
                                                self.txt_db.v_range,
                                                self.txt_db.mask)
        elif self.comasking_mode == "mix":
            mix_prob = 0.5 #Hardly set.
            if random.random() < mix_prob:
                input_ids, txt_labels = random_word_dmasking(example,
                                                self.txt_db.v_range,
                                                self.txt_db.mask)
            else:
                input_ids, txt_labels = random_word(example['input_ids'],
                                            self.txt_db.v_range,
                                            self.txt_db.mask)
        else:
            raise ValueError('invalid comasking mode')
        input_ids = [self.txt_db.cls_] + input_ids + [self.txt_db.sep]
        
        #create position_ids, specific built for XLMR
        position_ids = []
        position_id = 2
        for input_id in input_ids:
            if input_id == 0:
                position_id = 2
            else:
                position_id += 1
            position_ids.append(position_id)
        
                
        #convert position_ids and input_ids to the tensor
        position_ids = torch.tensor(position_ids)
        input_ids = torch.tensor(input_ids)
        #position_ids = torch.tensor(position_ids)
        txt_labels = torch.tensor([-1] + txt_labels + [-1]) #-1 stands for the words that are not masked, and the remaining is the id for the masked token
        return input_ids, txt_labels, position_ids

#proposed new dataset for MmxlmDataset
class MmxlmDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, mask_prob):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db)
        self.mask_prob = mask_prob
    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - img_mask: (num_bb, ), [0,0,0,1, ..., 0,0]
        - txt_labels   : (L + num_bb, ), [-1, -1, wid, -1, -1, -1] 
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(i)

        # text input
        input_ids, caption_txt_labels = self.create_mlm_io(example['input_ids'])
   
        # img input
        img_feat, img_pos_feat,img_soft_labels, num_bb = self._get_img_feat(
            example['img_fname'])
        
        img_mask, img_txt_labels = _get_img_mask_mmxlm(self.mask_prob, num_bb, img_soft_labels)
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        
        txt_labels = torch.cat([caption_txt_labels, img_txt_labels])
        return input_ids, img_feat, img_pos_feat, attn_masks, img_mask, txt_labels

    def create_mlm_io(self, input_ids):
        input_ids, txt_labels = random_word(input_ids,
                                            self.txt_db.v_range,
                                            self.txt_db.mask)
        input_ids = torch.tensor([self.txt_db.cls_]
                                 + input_ids
                                 + [self.txt_db.sep])
        txt_labels = torch.tensor([-1] + txt_labels + [-1]) #-1 stands for the words that are not masked, and the remaining is the id for the masked token
        return input_ids, txt_labels
    
    def _get_img_feat(self, fname):
        img_dump = self.img_db.get_dump(fname)
        num_bb = self.img_db.name2nbb[fname]
        img_feat = torch.tensor(img_dump['features'])
        bb = torch.tensor(img_dump['norm_bb'])
        img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
        img_soft_label = img_dump['soft_labels'] #no need to convert to tensor, just keep it as numpy array
        #TODO: convert img_soft_label to 
        return img_feat, img_bb, img_soft_label, num_bb

class VmlmDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, mask_prob, language='en'):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db)
        self.mask_prob = mask_prob
        self.language=language
    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - img_mask: (num_bb, ), [0,0,0,1, ..., 0,0]
        - txt_labels   : (L + num_bb, ), [-1, -1, wid, -1, -1, -1] 
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(i)

        # text input
        #input_ids, caption_txt_labels = self.create_mlm_io(example['input_ids'])
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        caption_txt_labels = torch.ones(len(input_ids), dtype=torch.long)*(-1)
        # img input
        img_feat, img_pos_feat,img_soft_labels, num_bb = self._get_img_feat(
            example['img_fname'])
        
        img_mask, img_txt_labels = _get_img_mask_mmxlm(self.mask_prob, num_bb, img_soft_labels, language=self.language)
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        
        txt_labels = torch.cat([caption_txt_labels, img_txt_labels],dim=0)
        return input_ids, img_feat, img_pos_feat, attn_masks, img_mask, txt_labels
    def _get_img_feat(self, fname):
        img_dump = self.img_db.get_dump(fname)
        num_bb = self.img_db.name2nbb[fname]
        img_feat = torch.tensor(img_dump['features'])
        bb = torch.tensor(img_dump['norm_bb'])
        img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
        img_soft_label = img_dump['soft_labels'] #no need to convert to tensor, just keep it as numpy array
        #TODO: convert img_soft_label to 
        return img_feat, img_bb, img_soft_label, num_bb

class Vmlm_Softlabel_Dataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, mask_prob, img_soft_label_db):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db)
        self.mask_prob = mask_prob
        self.img_soft_label_db = img_soft_label_db
    def __getitem__(self, i):
        example = super().__getitem__(i)

        # text input
        #input_ids, caption_txt_labels = self.create_mlm_io(example['input_ids'])
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        caption_token_soft_labels = torch.zeros((len(input_ids), XLMR_TOKER.vocab_size))
        # img input
        img_feat, img_pos_feat,img_token_soft_labels, num_bb = self._get_img_feat(
            example['img_fname'])
        #img_mask, img_token_soft_labels = _get_img_mask_mmxlm_softlabel(self.mask_prob, num_bb, img_soft_labels)
        img_mask = _get_img_mask(self.mask_prob, num_bb)
        
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        
        tgt_mask = torch.cat([torch.zeros(len(input_ids), dtype=torch.uint8),img_mask])
        #tgt_token_soft_labels = torch.cat([caption_token_soft_labels, img_token_soft_labels],dim=0)
        return input_ids, img_feat, img_pos_feat, attn_masks, img_mask, tgt_mask, img_token_soft_labels
    def _get_img_feat(self, fname):
        img_dump = self.img_db.get_dump(fname)
        num_bb = self.img_db.name2nbb[fname]
        img_feat = torch.tensor(img_dump['features'])
        bb = torch.tensor(img_dump['norm_bb'])
        img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
        #img_soft_label = torch.tensor(img_dump['soft_labels'])
        #img_soft_label = img_dump['soft_labels']
        #print(img_soft_label.dtype)
        # with h5py.File(self.img_soft_label_db_path, 'r') as img_soft_label_db:
        #     img_token_soft_label = torch.tensor(np.array(img_soft_label_db.get(fname), dtype=np.float32))
        #print(fname)
        img_token_soft_label = torch.tensor(self.img_soft_label_db[fname]['img_soft_label'])
        #print(img_token_soft_label.size())
        #print(img_token_soft_label.dtype)
        if len(img_token_soft_label.size()) == 0:
            print(fname)

        assert len(img_token_soft_label.size()) != 0, "Datapoint with issue: {}".format(fname)
        return img_feat, img_bb, img_token_soft_label, num_bb


class Mmxlm_Softlabel_Dataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, mask_prob):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db)
        self.mask_prob = mask_prob
    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - img_mask: (num_bb, ), [0,0,0,1, ..., 0,0]
        - tgt_mask: (L+num_bb, ), [0,0,1,...,0,0]
        - txt_labels   : (L + num_bb, V), [[-1, -1, wid, -1, -1, -1]]
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(i)

        # text input
        input_ids, caption_txt_labels = self.create_mlm_io(example['input_ids'])
   
        # img input
        img_feat, img_pos_feat,img_soft_labels, num_bb = self._get_img_feat(
            example['img_fname'])
        
        img_mask, img_token_soft_labels= _get_img_mask_mmxlm_softlabel(self.mask_prob, num_bb, img_soft_labels)
        #get the masks
        txt_mask, txt_token_soft_labels = _get_txt_mask_mmxlm_softlabel(caption_txt_labels)
        
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        
        #txt_labels = torch.cat([caption_txt_labels, img_txt_labels])
        
        #concatenate the mask to make the total mask
        tgt_mask = torch.cat([txt_mask,img_mask])
        tgt_token_soft_labels = torch.cat([txt_token_soft_labels, img_token_soft_labels])
        return input_ids, img_feat, img_pos_feat, attn_masks, img_mask, tgt_mask, tgt_token_soft_labels 

    def create_mlm_io(self, input_ids):
        input_ids, txt_labels = random_word(input_ids,
                                            self.txt_db.v_range,
                                            self.txt_db.mask)
        input_ids = torch.tensor([self.txt_db.cls_]
                                 + input_ids
                                 + [self.txt_db.sep])
        #don't convert txt_labels to tensor for now
        txt_labels = [-1] + txt_labels + [-1]  #-1 stands for the words that are not masked, and the remaining is the id for the masked token
        return input_ids, txt_labels
    
    def _get_img_feat(self, fname):
        img_dump = self.img_db.get_dump(fname)
        num_bb = self.img_db.name2nbb[fname]
        img_feat = torch.tensor(img_dump['features'])
        bb = torch.tensor(img_dump['norm_bb'])
        img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
        img_soft_label = img_dump['soft_labels'] #no need to convert to tensor, just keep it as numpy array
        #TODO: convert img_soft_label to 
        return img_feat, img_bb, img_soft_label, num_bb

class MlmDataset_VLXLMR(DetectFeatTxtTokDatasetCutDown):
    def __init__(self, txt_db, img_db, training_cut=-1):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db, training_cut)

    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(i)

        # text input
        input_ids, txt_labels = self.create_mlm_io(example['input_ids'])

        # img input
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, txt_labels

    def create_mlm_io(self, input_ids):
        input_ids, txt_labels = random_word(input_ids,
                                            self.txt_db.v_range,
                                            self.txt_db.mask)
        input_ids = torch.tensor([self.txt_db.cls_]
                                 + input_ids
                                 + [self.txt_db.sep])
        txt_labels = torch.tensor([-1] + txt_labels + [-1]) #-1 stands for the words that are not masked, and the remaining is the id for the masked token
        return input_ids, txt_labels

###########################################

class MlmDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db)

    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(i)

        # text input
        input_ids, txt_labels = self.create_mlm_io(example['input_ids'])

        # img input
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, txt_labels

    def create_mlm_io(self, input_ids):
        input_ids, txt_labels = random_word(input_ids,
                                            self.txt_db.v_range,
                                            self.txt_db.mask)
        input_ids = torch.tensor([self.txt_db.cls_]
                                 + input_ids
                                 + [self.txt_db.sep])
        txt_labels = torch.tensor([-1] + txt_labels + [-1]) #-1 stands for the words that are not masked, and the remaining is the id for the masked token
        return input_ids, txt_labels


def mlm_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_pos_feat (n, max_num_bb, 7)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    """
    (input_ids, img_feats, img_pos_feats, attn_masks, txt_labels
     ) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_labels': txt_labels}
    return batch
############################Added by Mingyang Zhou###############################
def xlmr_mlm_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 1
    :position_ids (n, max_L) padded with 0 # Mingyang: doesn't matter for Roberta, as we will regenerate position ids. 
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_pos_feat (n, max_num_bb, 7)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    """
    (input_ids, img_feats, img_pos_feats, attn_masks, txt_labels
     ) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    #position_ids = 
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
    
    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_labels': txt_labels}
    return batch

def xlmr_tlm_ni_dmasking_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 1
    :position_ids (n, max_L) padded with 0 # Mingyang: doesn't matter for Roberta, as we will regenerate position ids. 
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_pos_feat (n, max_num_bb, 7)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    """
    (input_ids, img_feats, img_pos_feats, attn_masks, txt_labels, position_ids
     ) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    position_ids = pad_sequence(position_ids, batch_first=True, padding_value=1)
    #position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
    
    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    #out_size = attn_masks.size(1)
    gather_index = None

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_labels': txt_labels}
    return batch

def xlmr_mlm_dmasking_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 1
    :position_ids (n, max_L) padded with 0 # Mingyang: doesn't matter for Roberta, as we will regenerate position ids. 
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_pos_feat (n, max_num_bb, 7)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    """
    (input_ids, img_feats, img_pos_feats, attn_masks, txt_labels, position_ids
     ) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    position_ids = pad_sequence(position_ids, batch_first=True, padding_value=1)
    #position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0)
    
    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_labels': txt_labels}
    return batch

def xlmr_mmxlm_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 1
    :position_ids (n, max_L) padded with 0 # Mingyang: doesn't matter for Roberta, as we will regenerate position ids. 
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_pos_feat (n, max_num_bb, 7)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :img_masks    (n, num_bb)
    :txt_labels   (n, max_{L + num_bb}) padded with -1
    """
    (input_ids, img_feats, img_pos_feats, attn_masks, img_masks, txt_labels
     ) = map(list, unzip(inputs)) #added by Mingyang Zhou

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    
    #Added img_masks
    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    
    #mask img_feats
    img_feat = _mask_img_feat(img_feat, img_masks) #Added by Mingyang Zhou
    
    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'img_masks': img_masks,
             'txt_labels': txt_labels}
    return batch

def xlmr_mmxlm_softlabel_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 1
    :position_ids (n, max_L) padded with 0 # Mingyang: doesn't matter for Roberta, as we will regenerate position ids. 
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_pos_feat (n, max_num_bb, 7)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :img_masks    (n, num_bb)
    :tgt_masks    (n, max_{L+num_bb}) padded with 0
    :img_token_soft_labels   (n, num_bb, label_token_size) 
    """
    (input_ids, img_feats, img_pos_feats, attn_masks, img_masks, tgt_masks, img_token_soft_labels
     ) = map(list, unzip(inputs)) #added by Mingyang Zhou

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    #txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    l_num_bbs = [tl+il for tl,il in zip(txt_lens, num_bbs)] #demonstrate the l+num bbs
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    
#     for soft_label,l in zip(tgt_token_soft_labels,l):
#         print(soft_label.size())
#         print(l)
    #tgt_token_soft_labels = pad_tensors(tgt_token_soft_labels, l_num_bbs) #get the target token  soft labels  
    img_token_soft_label = pad_tensors(img_token_soft_labels, num_bbs)
    #Added img_masks
    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)
    tgt_masks = pad_sequence(tgt_masks, batch_first=True, padding_value=0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    
    img_feat = _mask_img_feat(img_feat, img_masks) #masked the corresponding img_feat
    label_targets = _get_targets(img_masks, img_token_soft_label) #Get the label_targets
    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'img_masks': img_masks,
             'tgt_masks': tgt_masks,
             'label_targets': label_targets}
    return batch

##########################################################################################
class BlindMlmDataset(Dataset):
    def __init__(self, txt_db):
        assert isinstance(txt_db, TxtTokLmdb)
        self.txt_db = txt_db
        self.lens, self.ids = get_ids_and_lens(txt_db)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        example = self.txt_db[id_]
        input_ids, txt_labels = self.create_mlm_io(example['input_ids'])
        attn_masks = torch.ones(len(input_ids), dtype=torch.long)

        return input_ids, attn_masks, txt_labels


def mlm_blind_collate(inputs):
    input_ids, attn_masks, txt_labels = map(list, unzip(inputs))

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'attn_masks': attn_masks,
             'txt_labels': txt_labels}
    return batch


def eval_mask(len_, num_samples=7):
    """ build the mask for evaluating MLM
    circularly mask 1 word out of every x words
    """
    # build the random masks
    if len_ <= num_samples:
        masks = torch.eye(len_).byte()
        num_samples = len_
    else:
        mask_inds = [list(range(i, len_, num_samples))
                     for i in range(num_samples)]
        masks = torch.zeros(num_samples, len_).byte()
        for i, indices in enumerate(mask_inds):
            for j in indices:
                masks.data[i, j] = 1
    assert (masks.sum(dim=0) != torch.ones(len_).long()).sum().item() == 0
    assert masks.sum().item() == len_
    return masks


def eval_gather_inds(len_, num_samples=7):
    """ get the gather indices """
    inds = torch.arange(0, num_samples, dtype=torch.long)
    mul = math.ceil(len_ / num_samples)
    output = inds.repeat(mul)[:len_]
    return output


def stack_pad_tensors(tensors, lens=None, ns=None, pad=0):
    """N x [B_i, T, ...]"""
    if ns is None:
        ns = [t.size(0) for t in tensors]
    if lens is None:
        lens = [t.size(1) for t in tensors]
    max_len = max(lens)
    bs = sum(ns)
    hid_dims = tensors[0].size()[2:]
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, *hid_dims, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    i = 0
    for t, l, n in zip(tensors, lens, ns):
        output.data[i:i+n, :l, ...] = t.data
        i += n
    return output


def expand_tensors(tensors, ns):
    return [t.unsqueeze(0).expand(n, *tuple([-1]*t.dim()))
            for t, n in zip(tensors, ns)]


class MlmEvalDataset(DetectFeatTxtTokDataset):
    """ For evaluating MLM training task """
    def __init__(self, txt_db, img_db):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db)

    def __getitem__(self, i):
        example = super().__getitem__(i)

        # text input
        (input_ids, txt_labels, gather_inds
         ) = self.create_mlm_eval_io(example['input_ids'])

        # img input
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        attn_masks = torch.ones(input_ids.size(1) + num_bb, dtype=torch.long)

        return (input_ids, img_feat, img_pos_feat, attn_masks,
                txt_labels, gather_inds)

    def create_mlm_eval_io(self, input_ids):
        txt_labels = torch.tensor(input_ids)
        masks = eval_mask(len(input_ids))
        n_mask = masks.size(0)
        masks = torch.cat([torch.zeros(n_mask, 1).byte(),
                           masks,
                           torch.zeros(n_mask, 1).byte()],
                          dim=1)
        input_ids = torch.tensor([[self.txt_db.cls_]
                                  + input_ids
                                  + [self.txt_db.sep]
                                  for _ in range(n_mask)])
        input_ids.data.masked_fill_(masks, self.txt_db.mask)
        gather_inds = eval_gather_inds(len(txt_labels))
        return input_ids, txt_labels, gather_inds


def _batch_gather_tgt(gather_inds, n_masks):
    gather_tgts = []
    offset = 0
    for g, n in zip(gather_inds, n_masks):
        gather_tgts.append(g + offset)
        offset += n
    gather_tgt = pad_sequence(gather_tgts, batch_first=True, padding_value=0)
    return gather_tgt


def mlm_eval_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, txt_labels, gather_inds
     ) = map(list, unzip(inputs))

    # sizes
    n_masks, txt_lens = map(list, unzip(i.size() for i in input_ids))

    # text batches
    input_ids = stack_pad_tensors(input_ids, txt_lens, n_masks)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    gather_tgt = _batch_gather_tgt(gather_inds, n_masks)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = stack_pad_tensors(expand_tensors(img_feats, n_masks),
                                 num_bbs, n_masks)
    img_pos_feat = stack_pad_tensors(expand_tensors(img_pos_feats, n_masks),
                                     num_bbs, n_masks)

    bs, max_tl = input_ids.size()
    attn_masks = stack_pad_tensors(expand_tensors(attn_masks, n_masks),
                                   None, n_masks)
    out_size = attn_masks.size(1)
    # repeat txt_lens, num_bbs
    txt_lens = [l for l, n in zip(txt_lens, n_masks) for _ in range(n)]
    num_bbs = [b for b, n in zip(num_bbs, n_masks) for _ in range(n)]
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'gather_tgt': gather_tgt,
             'txt_labels': txt_labels}
    return batch


class BlindMlmEvalDataset(Dataset):
    def __init__(self, txt_db):
        assert isinstance(txt_db, TxtTokLmdb)
        self.txt_db = txt_db
        self.lens, self.ids = get_ids_and_lens(txt_db)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        example = self.txt_db[id_]
        input_ids = example['input_ids']

        # text input
        input_ids = example['input_ids']
        (input_ids, txt_labels, gather_inds
         ) = self.txt_db.create_mlm_eval_io(input_ids)

        attn_masks = torch.ones(len(input_ids), dtype=torch.long)

        return input_ids, attn_masks, txt_labels, gather_inds


def mlm_blind_eval_collate(inputs):
    (input_ids, position_ids, attn_masks, txt_labels, gather_inds
     ) = map(list, unzip(inputs))

    # sizes
    n_masks, txt_lens = map(list, unzip(i.size() for i in input_ids))

    # text batches
    input_ids = stack_pad_tensors(input_ids, txt_lens, n_masks)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    attn_masks = stack_pad_tensors(expand_tensors(attn_masks, n_masks),
                                   None, n_masks)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    gather_tgt = _batch_gather_tgt(gather_inds, n_masks)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'attn_masks': attn_masks,
             'gather_tgt': gather_tgt,
             'txt_labels': txt_labels}
    return batch
