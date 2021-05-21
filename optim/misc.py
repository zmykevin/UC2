"""
Misc lr helper
"""
from torch.optim import Adam, Adamax

from .adamw import AdamW


def build_optimizer(model, opts):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    return optimizer

def xlmr_pretrained_encoder_layer(n, load_layer):
    """
    Verify whether n loads the  pretrained weights from xlmr
    """
    assert isinstance(load_layer, int)
    #print(n)
    if  'roberta.encoder' in n:
        if int(n.split('.')[3]) <= load_layer:
            return True
    elif 'roberta.embeddings' in n:
        return True
    
    return  False

def build_xlmr_optimizer(model, opts):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    #Divide the parameters to four groups, we will apply a relatively small lr to the word embedding
    #assert opts.load_embedding_only or isinstance(opts.load_layer, int)
    if opts.load_layer:
        assert isinstance(opts.load_layer,int) and opts.load_layer > 0
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay) and xlmr_pretrained_encoder_layer(n, opts.load_layer)],
             'weight_decay': opts.weight_decay, 'lr': opts.xlmr_lr},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay) and xlmr_pretrained_encoder_layer(n, opts.load_layer)],
             'weight_decay': 0.0, 'lr': opts.xlmr_lr},
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay) and not xlmr_pretrained_encoder_layer(n, opts.load_layer)],
             'weight_decay': opts.weight_decay, 'lr': opts.learning_rate},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay) and not xlmr_pretrained_encoder_layer(n, opts.load_layer)],
             'weight_decay': 0.0, 'lr': opts.learning_rate},
        ]
    else:
        #general case
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay) and 'roberta.embeddings' in n],
             'weight_decay': opts.weight_decay, 'lr': opts.xlmr_lr},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay) and 'roberta.embeddings' in n],
             'weight_decay': 0.0, 'lr': opts.xlmr_lr},
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay) and 'roberta.embeddings' not in n],
             'weight_decay': opts.weight_decay, 'lr': opts.learning_rate},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay) and 'roberta.embeddings' not in n],
             'weight_decay': 0.0, 'lr': opts.learning_rate},
        ]

    
    #print([n for n, p in param_optimizer if not any(nd in n for nd in no_decay) and 'roberta.embeddings' in n])
    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
        
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    return optimizer


