"""
Bert for VQA model
"""
from collections import defaultdict

from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .layer import GELU
from .model import UniterPreTrainedModel, UniterModel, VLXLMRPreTrainedModel, VLXLMRModel


class VLXLMRForVisualQuestionAnswering(VLXLMRPreTrainedModel):
    """ Finetune multi-modal BERT for VQA
    """
    def __init__(self, config, img_dim, num_answer):
        super().__init__(config)
        self.roberta = VLXLMRModel(config, img_dim)
        self.vqa_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            GELU(),
            LayerNorm(config.hidden_size*2, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size*2, num_answer)
        )
        self.apply(self.init_weights)

    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        #position_ids = batch['position_ids']
        position_ids=None
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        sequence_output = self.roberta(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attn_masks, gather_index,
                                    output_all_encoded_layers=False)
        pooled_output = self.roberta.pooler(sequence_output)
        answer_scores = self.vqa_output(pooled_output)

        if compute_loss:
            targets = batch['targets']
            vqa_loss = F.binary_cross_entropy_with_logits(
                answer_scores, targets, reduction='none')
            return vqa_loss
        else:
            return answer_scores

class UniterForVisualQuestionAnswering(UniterPreTrainedModel):
    """ Finetune multi-modal BERT for VQA
    """
    def __init__(self, config, img_dim, num_answer):
        super().__init__(config)
        self.bert = UniterModel(config, img_dim)
        self.vqa_output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2),
            GELU(),
            LayerNorm(config.hidden_size*2, eps=1e-12),
            nn.Linear(config.hidden_size*2, num_answer)
        )
        self.apply(self.init_weights)

    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']
        sequence_output = self.bert(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attn_masks, gather_index,
                                    output_all_encoded_layers=False)
        pooled_output = self.bert.pooler(sequence_output)
        answer_scores = self.vqa_output(pooled_output)

        if compute_loss:
            targets = batch['targets']
            vqa_loss = F.binary_cross_entropy_with_logits(
                answer_scores, targets, reduction='none')
            return vqa_loss
        else:
            return answer_scores
