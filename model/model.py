"""
Pytorch modules
"""
from collections import defaultdict
import copy
import json
import logging
from io import open

import torch
from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .layer import GELU, BertLayer, BertPooler, BertOnlyMLMHead, RobertaLMHead, VisualRobertaLMHead
from .ot import optimal_transport_dist
from .const_variable import XLMR_TOKER, LABEL2TOKEN_MATRIX, VALID_XLMR_TOKEN_IDS
# with torch.no_grad():
#     LABEL2TOKEN_MATRIX = torch.cuda.HalfTensor(LABEL2TOKEN_MATRIX)

logger = logging.getLogger(__name__)

################vl-xlm-r##############################
def xlmr_sublayer_loading(state_dict, load_embedding_only=False, load_layer=0):
    old_keys = []
    new_keys = []
    if load_embedding_only:
        for key in state_dict.keys():
            if "roberta.embeddings" not in key:
                old_keys.append(key)
                new_keys.append("not_load."+key)
    
    elif load_layer:
        assert isinstance(load_layer,int) and load_layer > 0
        for key in state_dict.keys():
            if "roberta.encoder" in key and int(key.split('.')[3]) > load_layer:
                old_keys.append(key)
                new_keys.append("not_load."+key)

    for new_key, old_key in zip(new_keys, old_keys):
        state_dict[new_key] = state_dict.pop(old_key)



class VLXLMRConfig(object):
    """Configuration class to store the configuration of a `UniterModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=514,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 output_past=True,
                 layer_norm_eps= 1e-5,
                 pad_token_id=1,
                 ):
        """Constructs UniterConfig.
        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in
                `UniterModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer
                encoder.
            num_attention_heads: Number of attention heads for each attention
                layer in the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e.
                feed-forward) layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string)
                in the encoder and pooler. If string, "gelu", "relu" and
                "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully
                connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this
                model might ever be used with. Typically set this to something
                large just in case (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed
                into `UniterModel`.
            initializer_range: The sttdev of the truncated_normal_initializer
                for initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file,
                      "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps=layer_norm_eps
            self.pad_token_id=pad_token_id
        else:
            raise ValueError("First argument must be either a vocabulary size "
                             "(int) or the path to a pretrained model config "
                             "file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `UniterConfig` from a
           Python dictionary of parameters."""
        config = VLXLMRConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `UniterConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class VLXLMRPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, VLXLMRConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of "
                "class `VLXLMRConfig`. To create a model from a Google "
                "pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, config_file, state_dict, load_embedding_only=False, load_layer=None, *inputs, **kwargs):
        """
        Instantiate a VL_XLMRPreTrainedModel from a pre-trained model file or a
        pytorch state dict.
        Params:
            config_file: config json file
            state_dict: an state dictionnary
            *inputs, **kwargs: additional input for the specific Uniter class
        """
        # Load config
        config = VLXLMRConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []

        # for key in state_dict.keys():
        #     new_key = None
        #     if 'gamma' in key:
        #         new_key = key.replace('gamma', 'weight')
        #     if 'beta' in key:
        #         new_key = key.replace('beta', 'bias')
        #     if new_key:
        #         old_keys.append(key)
        #         new_keys.append(new_key)
        # for old_key, new_key in zip(old_keys, new_keys):
        #     state_dict[new_key] = state_dict.pop(old_key)
        if load_embedding_only:
            xlmr_sublayer_loading(state_dict, load_embedding_only=True)
        elif load_layer is not None:
            xlmr_sublayer_loading(state_dict, load_layer=load_layer)
        else:
            for key in state_dict.keys():
                new_key = None
                if 'gamma' in key:
                    new_key = key.replace('gamma', 'weight')
                if 'beta' in key:
                    new_key = key.replace('beta', 'bias')
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)


        
        #print the key of stat_dict]
        
        # print("Check the stat_dict after swapping the old key and new key")
        # for key in state_dict.keys():
        #     print(key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = ({} if metadata is None
                              else metadata.get(prefix[:-1], {}))
            #print("prefix is: {}".format(prefix))
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys,
                unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    #load(child, prefix + name + '.')

                    #only load text embedding layer
                    #if prefix+name == "roberta" or "roberta.embeddings" in prefix+name:
                        #print(prefix+name)
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'roberta.bert.') and any(s.startswith('roberta.bert.')
                                              for s in state_dict.keys()):
            start_prefix = 'roberta.bert.'
        # if not hasattr(model, 'roberta.') and any(s.startswith('roberta.')
        #                                       for s in state_dict.keys()):
        #     start_prefix = 'roberta.'
        # if not hasattr(model, 'bert.') and any(s.startswith('bert.')for s in state_dict.keys()):
        #     start_prefix = 'bert.'

        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from "
                        "pretrained model: {}".format(
                            model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in "
                        "{}: {}".format(
                            model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for '
                               '{}:\n\t{}'.format(
                                   model.__class__.__name__,
                                   "\n\t".join(error_msgs)))
        return model 

def create_position_ids_from_input_ids(input_ids, padding_idx):
    """ Replace non-padding symbols with their position numbers. Position numbers begin at
    padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
    `utils.make_positions`.
    :param torch.Tensor x:
    :return torch.Tensor:
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx

class VLXLMRTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=config.pad_token_id) 
        self.new_token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, position_ids=None,token_type_ids=None):
        """
        This forwarding is modified compared to Roberta.
        """
        #print(token_type_ids)
        if token_type_ids is None:
            #token_type_ids = torch.zeros_like(input_ids)
            token_type_ids = torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device)
        
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
                
        #print(position_ids[0])
        #print(position_ids.size())
        #print(input_ids[0])
        #print(input_ids.size())
        #return
        words_embeddings = self.word_embeddings(input_ids)
        mask = (input_ids != 1)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.new_token_type_embeddings(token_type_ids)
        
        embeddings = (words_embeddings
                      + position_embeddings
                      + token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings



class VLXLMRImageEmbeddings(nn.Module):
    def __init__(self, config, img_dim):
        super().__init__()
        self.img_linear = nn.Linear(img_dim, config.hidden_size)
        self.img_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pos_layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pos_linear = nn.Linear(7, config.hidden_size)
        self.mask_embedding = nn.Embedding(2, img_dim, padding_idx=0)

        # tf naming convention for layer norm
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, img_pos_feat, type_embeddings, img_masks=None):
        if img_masks is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask = self.mask_embedding(img_masks.long())
            img_feat = img_feat + mask
        #print(img_feat)
        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_pos = self.pos_layer_norm(self.pos_linear(img_pos_feat))
        embeddings = transformed_im + transformed_pos + type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class VLXLMREncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, input_, attention_mask,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        hidden_states = input_
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class VLXLMRModel(VLXLMRPreTrainedModel):
    """ Modification for Joint Vision-Language Encoding
    """
    def __init__(self, config, img_dim):
        super().__init__(config)
        self.embeddings = VLXLMRTextEmbeddings(config)
        self.img_embeddings = VLXLMRImageEmbeddings(config, img_dim)
        self.encoder = VLXLMREncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)

    def _compute_txt_embeddings(self, input_ids, position_ids,
                                txt_type_ids=None):
        output = self.embeddings(input_ids, position_ids, txt_type_ids)
        return output

    def _compute_img_embeddings(self, img_feat, img_pos_feat, img_masks=None,
                                img_type_ids=None):
        if img_type_ids is None:
            img_type_ids = torch.ones_like(img_feat[:, :, 0].long())
        img_type_embeddings = self.embeddings.new_token_type_embeddings(
            img_type_ids)
        #print(img_feat)
        output = self.img_embeddings(img_feat, img_pos_feat,
                                     img_type_embeddings, img_masks)
        return output

    def _compute_img_txt_embeddings(self, input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    gather_index, img_masks=None,
                                    txt_type_ids=None, img_type_ids=None):
        txt_emb = self._compute_txt_embeddings(
            input_ids, position_ids, txt_type_ids)
        img_emb = self._compute_img_embeddings(
            img_feat, img_pos_feat, img_masks, img_type_ids)
        # align back to most compact input
        gather_index = gather_index.unsqueeze(-1).expand(
            -1, -1, self.config.hidden_size)
        embedding_output = torch.gather(torch.cat([txt_emb, img_emb], dim=1),
                                        dim=1, index=gather_index)
        return embedding_output

    def forward(self, input_ids, position_ids,
                img_feat, img_pos_feat,
                attention_mask, gather_index=None, img_masks=None,
                output_all_encoded_layers=True,
                txt_type_ids=None, img_type_ids=None):
        # compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        #print(img_feat)
        # embedding layer
        if input_ids is None:
            # image only
            embedding_output = self._compute_img_embeddings(
                img_feat, img_pos_feat, img_masks, img_type_ids)
        elif img_feat is None:
            # text only
            embedding_output = self._compute_txt_embeddings(
                input_ids, position_ids, txt_type_ids)
        else:
            embedding_output = self._compute_img_txt_embeddings(
                input_ids, position_ids,
                img_feat, img_pos_feat,
                gather_index, img_masks, txt_type_ids, img_type_ids)

        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers

class VLXLMRForPretraining(VLXLMRPreTrainedModel):
    """ MLM + MRM """
    def __init__(self, config, img_dim, img_label_dim,
                 nce_temp=1, ot_pos_only=False):
        super().__init__(config)
        self.roberta = VLXLMRModel(config, img_dim)
        self.cls = RobertaLMHead(
            config, self.roberta.embeddings.word_embeddings.weight)
        self.vis_cls = VisualRobertaLMHead(config, self.roberta.embeddings.word_embeddings.weight, VALID_XLMR_TOKEN_IDS)
        self.feat_regress = RegionFeatureRegression(
            config.hidden_size, img_dim,
            self.roberta.img_embeddings.img_linear.weight)
        self.region_classifier = RegionClassification(
            config.hidden_size, img_label_dim)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        '''
        self.nce_output = BertPredictionHeadTransform(config)
        self.nce_output = nn.Sequential(BertPredictionHeadTransform(config),
                                        nn.Linear(config.hidden_size, img_dim))
        self.nce_norm = LayerNorm(config.hidden_size, eps=1e-12)
        self.nce_temp = nce_temp  # temperature
        '''
        self.ot_pos_only = ot_pos_only
        self.apply(self.init_weights)
        self.vocab_pad = 0 #What is this?

    def pad_vocab(self):
        # FIXME better padding after integrating huggingface
        emb_w = self.roberta.embeddings.word_embeddings.weight.data
        padded_emb_w, n_pad = pad_tensor_to_mul(emb_w)
        padded_emb_w = nn.Parameter(padded_emb_w)
        self.roberta.embeddings.word_embeddings.weight = padded_emb_w
        self.cls.decoder.weight = padded_emb_w
        self.vocab_pad = n_pad

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        if task == 'tlm':
            position_ids = batch['position_ids']
        else:
            position_ids = None
        #position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        #print("The task is: {}".format(task))
        if task in ['mlm','tlm']:
            txt_labels = batch['txt_labels']
            #print(txt_labels)
            return self.forward_mlm(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    txt_labels, compute_loss)
        elif task == 'tlm-ni':
            txt_labels = batch['txt_labels']
            img_feat = None
            img_pos_feat = None
            gather_index = None
            return self.forward_mlm(input_ids, position_ids, img_feat, img_pos_feat, attention_mask, gather_index, txt_labels, compute_loss)
        elif task in ['mmxlm','vmlm']:
            txt_labels = batch['txt_labels']
            img_masks = batch['img_masks']
            return self.forward_mmxlm(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,img_masks,
                                    txt_labels, compute_loss)
        elif task in ["mmxlm-soft", 'vmlm-soft']:
            img_masks = batch['img_masks']
            tgt_masks = batch['tgt_masks']
            mmxlm_soft_label_targets = batch['label_targets']
            #process the mmxlm_soft_label_targets
            # with torch.no_grad():
            #     mmxlm_soft_label_targets = torch.matmul(mmxlm_soft_label_targets, torch.cuda.HalfTensor(LABEL2TOKEN_MATRIX))
            #     mmxlm_soft_label_targets = mmxlm_soft_label_targets[:, VALID_XLMR_TOKEN_IDS]
            #     mmxlm_soft_label_targets = mmxlm_soft_label_targets / torch.sum(mmxlm_soft_label_targets, dim=1, keepdim=True)
            
            return self.forward_mmxlm_soft(input_ids, position_ids, 
                                           img_feat, img_pos_feat, 
                                           attention_mask, gather_index, img_masks,
                                           tgt_masks, mmxlm_soft_label_targets, compute_loss)
        elif task == 'mrfr':
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrfr_feat_target = batch['feat_targets']
            return self.forward_mrfr(input_ids, position_ids,
                                     img_feat, img_pos_feat,
                                     attention_mask, gather_index,
                                     img_masks, img_mask_tgt,
                                     mrfr_feat_target, compute_loss)
        elif task == 'itm':
            targets = batch['targets']
            ot_inputs = batch['ot_inputs']
            return self.forward_itm(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    targets, ot_inputs, compute_loss)
        elif task.startswith('mrc'):
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrc_label_target = batch['label_targets']
            return self.forward_mrc(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    img_masks, img_mask_tgt,
                                    mrc_label_target, task, compute_loss)
        else:
            raise ValueError('invalid task')

    # MLM
    def forward_mlm(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index,txt_labels, compute_loss=True):
        if gather_index is not None:
            sequence_output = self.roberta(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    output_all_encoded_layers=False)
        else:
            sequence_output = self.roberta(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, output_all_encoded_layers=False)
        # get only the text part
        sequence_output = sequence_output[:, :input_ids.size(1), :]
        # only compute masked tokens for better efficiency
        #print(txt_labels)
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    txt_labels != -1)
        prediction_scores = self._pad_layer_unpad(masked_output, self.cls)
        if self.vocab_pad:
            prediction_scores = prediction_scores[:, :-self.vocab_pad]

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='none')
            return masked_lm_loss
        else:
            return prediction_scores

    def forward_mmxlm(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, img_masks,
                    txt_labels, compute_loss=True):
        sequence_output = self.roberta(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    output_all_encoded_layers=False,
                                    img_masks=img_masks)
        # get only the text part
        # sequence_output = sequence_output[:, :input_ids.size(1), :]
        
        # only compute masked tokens for better efficiency
        # print
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    txt_labels != -1)
        prediction_scores = self._pad_layer_unpad(masked_output, self.cls)
        if self.vocab_pad:
            prediction_scores = prediction_scores[:, :-self.vocab_pad]

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='none')
            return masked_lm_loss
        else:
            return prediction_scores

    def forward_mmxlm_soft(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, img_masks, tgt_masks,
                    label_targets, compute_loss=True):
        sequence_output = self.roberta(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    output_all_encoded_layers=False,
                                    img_masks=img_masks)

        # only compute masked regions for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    tgt_masks)
        prediction_soft_label = self._pad_layer_unpad(masked_output,
                                                      self.cls)
        #get only the effective token for VLM
        prediction_soft_label = prediction_soft_label[:, VALID_XLMR_TOKEN_IDS]
        #print(compute_loss)
        if compute_loss:
            prediction_soft_label = F.log_softmax(
                prediction_soft_label, dim=-1)
            mmxlm_soft_loss = F.kl_div(
                prediction_soft_label, label_targets, reduction='none')
            return mmxlm_soft_loss
        else:
            return prediction_soft_label

    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def _pad_layer_unpad(self, input_, layer):
        input_, n_pad = pad_tensor_to_mul(input_) #Mingyang: What does this do? 
        output = layer(input_)
        if n_pad:
            output = output[:-n_pad, :]
        return output
    

    # MRFR
    def forward_mrfr(self, input_ids, position_ids, img_feat, img_pos_feat,
                     attention_mask, gather_index, img_masks, img_mask_tgt,
                     feat_targets, compute_loss=True):
        sequence_output = self.roberta(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    output_all_encoded_layers=False,
                                    img_masks=img_masks)

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_feat = self._pad_layer_unpad(masked_output,
                                                self.feat_regress)

        if compute_loss:
            mrfr_loss = F.mse_loss(prediction_feat, feat_targets,
                                   reduction='none')
            return mrfr_loss
        else:
            return prediction_feat

    def forward_itm(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, targets, ot_inputs,
                    compute_loss=True):
        sequence_output = self.roberta(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    output_all_encoded_layers=False)
        pooled_output = self.roberta.pooler(sequence_output)
        rank_scores = self.itm_output(pooled_output)

        # OT loss
        if ot_inputs is not None:
            ot_scatter = ot_inputs['ot_scatter']

            b = sequence_output.size(0)
            tl = input_ids.size(1)
            il = img_feat.size(1)
            max_l = max(ot_inputs['scatter_max'] + 1, tl+il)

            ot_scatter = ot_scatter.unsqueeze(-1).expand_as(sequence_output)
            ctx_emb = torch.zeros(b, max_l, self.config.hidden_size,
                                  dtype=sequence_output.dtype,
                                  device=sequence_output.device
                                  ).scatter_(dim=1, index=ot_scatter,
                                             src=sequence_output)
            txt_emb = ctx_emb[:, :tl, :]
            img_emb = ctx_emb[:, tl:tl+il, :]

            txt_pad = ot_inputs['txt_pad']
            img_pad = ot_inputs['img_pad']
            ot_dist = optimal_transport_dist(txt_emb, img_emb,
                                             txt_pad, img_pad)
            if self.ot_pos_only:
                ot_loss = ot_dist.masked_select(targets == 1)
            else:
                ot_pos_dist = ot_dist.masked_select(targets == 1)
                ot_neg_dist = ot_dist.masked_select(targets == 0)
                ot_loss = (ot_pos_dist, ot_neg_dist)
        else:
            ot_loss = None

        if compute_loss:
            itm_loss = F.cross_entropy(rank_scores, targets, reduction='none')
            return itm_loss, ot_loss
        else:
            return rank_scores, ot_loss

    # MRC
    def forward_mrc(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, img_masks, img_mask_tgt,
                    label_targets, task, compute_loss=True):
        sequence_output = self.roberta(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    output_all_encoded_layers=False,
                                    img_masks=img_masks)

        
        # print(sequence_output.size())
        # print("text length: ")
        # print(input_ids.size())
        # print("image input length: ")
        # print(img_feat.size())
        # print(img_mask_tgt.size())

        # only compute masked regions for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_soft_label = self._pad_layer_unpad(masked_output,
                                                      self.region_classifier)

        if compute_loss:
            if "kl" in task:
                prediction_soft_label = F.log_softmax(
                    prediction_soft_label, dim=-1)
                mrc_loss = F.kl_div(
                    prediction_soft_label, label_targets, reduction='none')
            else:
                # background class should not be the target
                label_targets = torch.max(label_targets[:, 1:], dim=-1)[1] + 1
                mrc_loss = F.cross_entropy(
                    prediction_soft_label, label_targets,
                    ignore_index=0, reduction='none')
            return mrc_loss
        else:
            return prediction_soft_label
    
######################################################
class UniterConfig(object):
    """Configuration class to store the configuration of a `UniterModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs UniterConfig.
        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in
                `UniterModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer
                encoder.
            num_attention_heads: Number of attention heads for each attention
                layer in the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e.
                feed-forward) layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string)
                in the encoder and pooler. If string, "gelu", "relu" and
                "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully
                connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this
                model might ever be used with. Typically set this to something
                large just in case (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed
                into `UniterModel`.
            initializer_range: The sttdev of the truncated_normal_initializer
                for initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file,
                      "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size "
                             "(int) or the path to a pretrained model config "
                             "file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `UniterConfig` from a
           Python dictionary of parameters."""
        config = UniterConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `UniterConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class UniterPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, UniterConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of "
                "class `UniterConfig`. To create a model from a Google "
                "pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, config_file, state_dict, *inputs, **kwargs):
        """
        Instantiate a UniterPreTrainedModel from a pre-trained model file or a
        pytorch state dict.
        Params:
            config_file: config json file
            state_dict: an state dictionnary
            *inputs, **kwargs: additional input for the specific Uniter class
        """
        # Load config
        config = UniterConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = ({} if metadata is None
                              else metadata.get(prefix[:-1], {}))
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys,
                unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.')
                                              for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from "
                        "pretrained model: {}".format(
                            model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in "
                        "{}: {}".format(
                            model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for '
                               '{}:\n\t{}'.format(
                                   model.__class__.__name__,
                                   "\n\t".join(error_msgs)))
        return model


class UniterTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)
        

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # token_type_embeddings = self.new_token_type_embeddings(token_type_ids)

        embeddings = (words_embeddings
                      + position_embeddings
                      + token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class UniterImageEmbeddings(nn.Module):
    def __init__(self, config, img_dim):
        super().__init__()
        self.img_linear = nn.Linear(img_dim, config.hidden_size)
        self.img_layer_norm = LayerNorm(config.hidden_size, eps=1e-12)
        self.pos_layer_norm = LayerNorm(config.hidden_size, eps=1e-12)
        self.pos_linear = nn.Linear(7, config.hidden_size)
        self.mask_embedding = nn.Embedding(2, img_dim, padding_idx=0)

        # tf naming convention for layer norm
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, img_pos_feat, type_embeddings, img_masks=None):
        if img_masks is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask = self.mask_embedding(img_masks.long())
            img_feat = img_feat + mask

        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_pos = self.pos_layer_norm(self.pos_linear(img_pos_feat))
        embeddings = transformed_im + transformed_pos + type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class UniterEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, input_, attention_mask,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        hidden_states = input_
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


def pad_tensor_to_mul(tensor, dim=0, mul=8):
    """ pad tensor to multiples (8 for tensor cores) """
    # TODO find out whether this helps speed
    return tensor, 0
    t_size = list(tensor.size())
    n_pad = mul - t_size[dim] % mul
    if n_pad == mul:
        n_pad = 0
        padded_tensor = tensor
    else:
        t_size[dim] = n_pad
        pad = torch.zeros(*t_size, dtype=tensor.dtype, device=tensor.device)
        padded_tensor = torch.cat([tensor, pad], dim=dim)
    return padded_tensor, n_pad


class UniterModel(UniterPreTrainedModel):
    """ Modification for Joint Vision-Language Encoding
    """
    def __init__(self, config, img_dim):
        super().__init__(config)
        self.embeddings = UniterTextEmbeddings(config)
        self.img_embeddings = UniterImageEmbeddings(config, img_dim)
        self.encoder = UniterEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)

    def _compute_txt_embeddings(self, input_ids, position_ids,
                                txt_type_ids=None):
        output = self.embeddings(input_ids, position_ids, txt_type_ids)
        return output

    def _compute_img_embeddings(self, img_feat, img_pos_feat, img_masks=None,
                                img_type_ids=None):
        if img_type_ids is None:
            img_type_ids = torch.ones_like(img_feat[:, :, 0].long())
        img_type_embeddings = self.embeddings.token_type_embeddings(
            img_type_ids)
        # img_type_embeddings = self.embeddings.new_token_type_embeddings(img_type_ids)
        output = self.img_embeddings(img_feat, img_pos_feat,
                                     img_type_embeddings, img_masks)
        return output

    def _compute_img_txt_embeddings(self, input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    gather_index, img_masks=None,
                                    txt_type_ids=None, img_type_ids=None):
        txt_emb = self._compute_txt_embeddings(
            input_ids, position_ids, txt_type_ids)
        img_emb = self._compute_img_embeddings(
            img_feat, img_pos_feat, img_masks, img_type_ids)
        # align back to most compact input
        gather_index = gather_index.unsqueeze(-1).expand(
            -1, -1, self.config.hidden_size)
        embedding_output = torch.gather(torch.cat([txt_emb, img_emb], dim=1),
                                        dim=1, index=gather_index)
        return embedding_output

    def forward(self, input_ids, position_ids,
                img_feat, img_pos_feat,
                attention_mask, gather_index=None, img_masks=None,
                output_all_encoded_layers=True,
                txt_type_ids=None, img_type_ids=None):
        # compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding layer
        if input_ids is None:
            # image only
            embedding_output = self._compute_img_embeddings(
                img_feat, img_pos_feat, img_masks, img_type_ids)
        elif img_feat is None:
            # text only
            embedding_output = self._compute_txt_embeddings(
                input_ids, position_ids, txt_type_ids)
        else:
            embedding_output = self._compute_img_txt_embeddings(
                input_ids, position_ids,
                img_feat, img_pos_feat,
                gather_index, img_masks, txt_type_ids, img_type_ids)

        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers


class RegionFeatureRegression(nn.Module):
    def __init__(self, hidden_size, feat_dim, img_linear_weight):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 GELU(),
                                 LayerNorm(hidden_size, eps=1e-12))

        self.weight = img_linear_weight
        self.bias = nn.Parameter(torch.zeros(feat_dim))

    def forward(self, input_):
        hidden = self.net(input_)
        output = F.linear(hidden, self.weight.t(), self.bias)
        return output


class RegionClassification(nn.Module):
    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 GELU(),
                                 LayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output


class UniterForPretraining(UniterPreTrainedModel):
    """ MLM + MRM """
    def __init__(self, config, img_dim, img_label_dim,
                 nce_temp=1, ot_pos_only=False):
        super().__init__(config)
        self.bert = UniterModel(config, img_dim)
        self.cls = BertOnlyMLMHead(
            config, self.bert.embeddings.word_embeddings.weight)
        self.feat_regress = RegionFeatureRegression(
            config.hidden_size, img_dim,
            self.bert.img_embeddings.img_linear.weight)
        self.region_classifier = RegionClassification(
            config.hidden_size, img_label_dim)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        '''
        self.nce_output = BertPredictionHeadTransform(config)
        self.nce_output = nn.Sequential(BertPredictionHeadTransform(config),
                                        nn.Linear(config.hidden_size, img_dim))
        self.nce_norm = LayerNorm(config.hidden_size, eps=1e-12)
        self.nce_temp = nce_temp  # temperature
        '''
        self.ot_pos_only = ot_pos_only
        self.apply(self.init_weights)
        self.vocab_pad = 0

    def pad_vocab(self):
        # FIXME better padding after integrating huggingface
        emb_w = self.bert.embeddings.word_embeddings.weight.data
        padded_emb_w, n_pad = pad_tensor_to_mul(emb_w)
        padded_emb_w = nn.Parameter(padded_emb_w)
        self.bert.embeddings.word_embeddings.weight = padded_emb_w
        self.cls.predictions.decoder.weight = padded_emb_w
        self.vocab_pad = n_pad

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        if task == 'mlm':
            txt_labels = batch['txt_labels']
            return self.forward_mlm(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    txt_labels, compute_loss)
        elif task == 'mmxlm':
            txt_labels = batch['txt_labels']
            img_masks = batch['img_masks']
            return self.forward_mmxlm(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,img_masks,
                                    txt_labels, compute_loss)
        elif task == 'mrfr':
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrfr_feat_target = batch['feat_targets']
            return self.forward_mrfr(input_ids, position_ids,
                                     img_feat, img_pos_feat,
                                     attention_mask, gather_index,
                                     img_masks, img_mask_tgt,
                                     mrfr_feat_target, compute_loss)
        elif task == 'mrm-nce':
            raise NotImplementedError('nce does not work')
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            img_masks_in = batch['img_masks_in']
            feat_target = batch['feat_targets']
            neg_feats = batch['neg_feats']
            return self.forward_mrm_nce(input_ids, position_ids,
                                        img_feat, img_pos_feat,
                                        attention_mask, gather_index,
                                        img_masks_in, img_masks, img_mask_tgt,
                                        feat_target, neg_feats, compute_loss)
        elif task == 'itm':
            targets = batch['targets']
            ot_inputs = batch['ot_inputs']
            return self.forward_itm(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    targets, ot_inputs, compute_loss)
        elif task.startswith('mrc'):
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrc_label_target = batch['label_targets']
            return self.forward_mrc(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    img_masks, img_mask_tgt,
                                    mrc_label_target, task, compute_loss)
        else:
            raise ValueError('invalid task')

    # MLM
    def forward_mlm(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index,
                    txt_labels, compute_loss=True):
        sequence_output = self.bert(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    output_all_encoded_layers=False)
        # get only the text part
        sequence_output = sequence_output[:, :input_ids.size(1), :]
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    txt_labels != -1)
        prediction_scores = self._pad_layer_unpad(masked_output, self.cls)
        if self.vocab_pad:
            prediction_scores = prediction_scores[:, :-self.vocab_pad]

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='none')
            return masked_lm_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def _pad_layer_unpad(self, input_, layer):
        input_, n_pad = pad_tensor_to_mul(input_)
        output = layer(input_)
        if n_pad:
            output = output[:-n_pad, :]
        return output

    def mlm_eval(self, input_ids, position_ids,
                 img_feat, img_pos_feat,
                 attention_mask, gather_index, gather_tgt):
        raise ValueError('Do not use this')
        sequence_output = self.bert(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    output_all_encoded_layers=False)
        # get only the text part (excluding [CLS], [SEP])
        sequence_output = sequence_output[:, 1:input_ids.size(1)-1, :]
        # only compute masked tokens for better efficiency
        index = gather_tgt.unsqueeze(-1).expand(
            -1, -1, self.config.hidden_size)
        masked_output = torch.gather(sequence_output, dim=0, index=index)
        prediction_scores = self.cls(masked_output)
        if self.vocab_pad:
            prediction_scores = prediction_scores[..., :-self.vocab_pad]
        return prediction_scores

    # MRFR
    def forward_mrfr(self, input_ids, position_ids, img_feat, img_pos_feat,
                     attention_mask, gather_index, img_masks, img_mask_tgt,
                     feat_targets, compute_loss=True):
        sequence_output = self.bert(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    output_all_encoded_layers=False,
                                    img_masks=img_masks)

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_feat = self._pad_layer_unpad(masked_output,
                                                self.feat_regress)

        if compute_loss:
            mrfr_loss = F.mse_loss(prediction_feat, feat_targets,
                                   reduction='none')
            return mrfr_loss
        else:
            return prediction_feat

    # MRM-NCE
    def forward_mrm_nce(self, input_ids, position_ids, img_feat, img_pos_feat,
                        attention_mask, gather_index,
                        img_masks_in, img_masks, img_mask_tgt,
                        feat_targets, neg_feats, compute_loss=True):
        sequence_output = self.bert(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    output_all_encoded_layers=False,
                                    img_masks=img_masks_in)

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)

        masked_output = self._pad_layer_unpad(masked_output, self.nce_output)
        # neg within batch
        batch_neg = self._compute_masked_hidden(img_feat, ~img_masks)
        neg_feats, _ = pad_tensor_to_mul(
            torch.cat([neg_feats, batch_neg], dim=0))

        # shared image linear transform
        neg_output = self.nce_norm(
            self.bert.img_embeddings.img_linear(neg_feats))
        pos_output = self._pad_layer_unpad(feat_targets,
                                           self.bert.img_embeddings.img_linear)
        pos_output = self.nce_norm(pos_output)

        if compute_loss:
            mrm_nce_loss = self.mrm_nce(masked_output, pos_output,
                                        neg_output, compute_loss=True)
            return mrm_nce_loss
        else:
            return masked_output, pos_output, neg_output

    def mrm_nce(self, masked_output, pos_output, neg_output,
                compute_loss=True):
        # dot product of ground truth feature
        masked_score = masked_output.matmul(pos_output.t())
        # dot product of neative samples
        neg_score = masked_output.matmul(neg_output.t())

        logits = torch.cat([masked_score, neg_score], dim=1).float()
        if compute_loss:
            targets = torch.arange(0, masked_output.size(0),
                                   dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits/self.nce_temp, targets,
                                   reduction='none')
            return loss
        else:
            return logits

    def forward_itm(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, targets, ot_inputs,
                    compute_loss=True):
        sequence_output = self.bert(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    output_all_encoded_layers=False)
        pooled_output = self.bert.pooler(sequence_output)
        rank_scores = self.itm_output(pooled_output)

        # OT loss
        if ot_inputs is not None:
            ot_scatter = ot_inputs['ot_scatter']
            b = sequence_output.size(0)
            tl = input_ids.size(1)
            il = img_feat.size(1)
            max_l = max(ot_inputs['scatter_max'] + 1, tl+il)

            ot_scatter = ot_scatter.unsqueeze(-1).expand_as(sequence_output)
            ctx_emb = torch.zeros(b, max_l, self.config.hidden_size,
                                  dtype=sequence_output.dtype,
                                  device=sequence_output.device
                                  ).scatter_(dim=1, index=ot_scatter,
                                             src=sequence_output)
            txt_emb = ctx_emb[:, :tl, :]
            img_emb = ctx_emb[:, tl:tl+il, :]

            txt_pad = ot_inputs['txt_pad']
            img_pad = ot_inputs['img_pad']
            ot_dist = optimal_transport_dist(txt_emb, img_emb,
                                             txt_pad, img_pad)
            if self.ot_pos_only:
                ot_loss = ot_dist.masked_select(targets == 1)
            else:
                ot_pos_dist = ot_dist.masked_select(targets == 1)
                ot_neg_dist = ot_dist.masked_select(targets == 0)
                ot_loss = (ot_pos_dist, ot_neg_dist)
        else:
            ot_loss = None

        if compute_loss:
            itm_loss = F.cross_entropy(rank_scores, targets, reduction='none')
            return itm_loss, ot_loss
        else:
            return rank_scores, ot_loss
    # MMXLM
    def forward_mmxlm(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, img_masks,
                    txt_labels, compute_loss=True):
        sequence_output = self.bert(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    output_all_encoded_layers=False,
                                    img_masks=img_masks)
        # get only the text part
        # sequence_output = sequence_output[:, :input_ids.size(1), :]
        
        # only compute masked tokens for better efficiency
        # print
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    txt_labels != -1)
        prediction_scores = self._pad_layer_unpad(masked_output, self.cls)
        if self.vocab_pad:
            prediction_scores = prediction_scores[:, :-self.vocab_pad]

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='none')
            return masked_lm_loss
        else:
            return prediction_scores
    # MRC
    def forward_mrc(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, img_masks, img_mask_tgt,
                    label_targets, task, compute_loss=True):
        sequence_output = self.bert(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    output_all_encoded_layers=False,
                                    img_masks=img_masks)

        # only compute masked regions for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_soft_label = self._pad_layer_unpad(masked_output,
                                                      self.region_classifier)
        
        if compute_loss:
            if "kl" in task:
                prediction_soft_label = F.log_softmax(
                    prediction_soft_label, dim=-1)
                mrc_loss = F.kl_div(
                    prediction_soft_label, label_targets, reduction='none')
            else:
                # background class should not be the target
                label_targets = torch.max(label_targets[:, 1:], dim=-1)[1] + 1
                mrc_loss = F.cross_entropy(
                    prediction_soft_label, label_targets,
                    ignore_index=0, reduction='none')
            return mrc_loss
        else:
            return prediction_soft_label
