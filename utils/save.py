"""
saving utilities
"""
from collections import OrderedDict
import json
import os
from os.path import abspath, dirname, exists, join
import subprocess

import torch
from apex.amp._amp_state import _amp_state

from utils.logger import LOGGER
from horovod import torch as hvd


def save_training_meta(args):
    if args.rank > 0:
        return

    if not exists(args.output_dir):
        os.makedirs(join(args.output_dir, 'log'))
        os.makedirs(join(args.output_dir, 'ckpt'))

    with open(join(args.output_dir, 'log', 'hps.json'), 'w') as writer:
        json.dump(vars(args), writer, indent=4)
    model_config = json.load(open(args.model_config))
    with open(join(args.output_dir, 'log', 'model.json'), 'w') as writer:
        json.dump(model_config, writer, indent=4)
    # git info
    #Mingyang: Ignore the git info
    # try:
    #     LOGGER.info("Waiting on git info....")
    #     c = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
    #                        timeout=10, stdout=subprocess.PIPE)
    #     git_branch_name = c.stdout.decode().strip()
    #     LOGGER.info("Git branch: %s", git_branch_name)
    #     c = subprocess.run(["git", "rev-parse", "HEAD"],
    #                        timeout=10, stdout=subprocess.PIPE)
    #     git_sha = c.stdout.decode().strip()
    #     LOGGER.info("Git SHA: %s", git_sha)
    #     git_dir = abspath(dirname(__file__))
    #     git_status = subprocess.check_output(
    #         ['git', 'status', '--short'],
    #         cwd=git_dir, universal_newlines=True).strip()
    #     with open(join(args.output_dir, 'log', 'git_info.json'),
    #               'w') as writer:
    #         json.dump({'branch': git_branch_name,
    #                    'is_dirty': bool(git_status),
    #                    'status': git_status,
    #                    'sha': git_sha},
    #                   writer, indent=4)
    # except subprocess.TimeoutExpired as e:
    #     LOGGER.exception(e)
    #     LOGGER.warn("Git info not found. Moving right along...")


class ModelSaver(object):
    def __init__(self, output_dir, prefix='model_step', suffix='pt'):
        self.output_dir = output_dir
        self.prefix = prefix
        self.suffix = suffix

    def save(self, model, step, optimizer=None):
        output_model_file = join(self.output_dir,
                                 f"{self.prefix}_{step}.{self.suffix}")
        state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in model.state_dict().items()}
        if hasattr(model, 'vocab_pad') and model.vocab_pad:
            # store vocab embeddings before padding
            emb_w = state_dict['bert.embeddings.word_embeddings.weight']
            emb_w = emb_w[:-model.vocab_pad, :]
            state_dict['bert.embeddings.word_embeddings.weight'] = emb_w
            state_dict['cls.predictions.decoder.weight'] = emb_w
        torch.save(state_dict, output_model_file)
        if optimizer is not None:
            dump = {'step': step, 'optimizer': optimizer.state_dict()}
            if hasattr(optimizer, '_amp_stash'):
                pass  # TODO fp16 optimizer
            torch.save(dump, f'{self.output_dir}/train_state_{step}.pt')


def amp_state_dict(destination=None):
    if destination is None:
        destination = OrderedDict()

    for idx, loss_scaler in enumerate(_amp_state.loss_scalers):
        destination['loss_scaler%d' % idx] = {
            'loss_scale': loss_scaler.loss_scale(),
            'unskipped': loss_scaler._unskipped,
        }
    return destination


def amp_load_state_dict(state_dict):
    # Check if state_dict containes the same number of loss_scalers as current
    # setup
    if len(state_dict) != len(_amp_state.loss_scalers):
        print('Warning: state_dict contains {} entries, while {} loss_scalers '
              'are used'.format(len(state_dict), len(_amp_state.loss_scalers)))

    state_dict = state_dict.copy()

    nb_loss_scalers = len(_amp_state.loss_scalers)
    unexpected_keys = []
    # Initialize idx outside, since unexpected_keys will increase it if
    # enumerate is used
    idx = 0
    for key in state_dict:
        if 'loss_scaler' not in key:
            unexpected_keys.append(key)
        else:
            if idx > (nb_loss_scalers - 1):
                print('Skipping loss_scaler[{}], since num_losses was set to '
                      '{}'.format(idx, nb_loss_scalers))
                break
            _amp_state.loss_scalers[idx]._loss_scale = (
                state_dict[key]['loss_scale'])
            _amp_state.loss_scalers[idx]._unskipped = (
                state_dict[key]['unskipped'])
            idx += 1

    if len(unexpected_keys) > 0:
        raise RuntimeError(
            'Error(s) in loading state_dict. Unexpected key(s) in state_dict: '
            '{}. '.format(', '.join('"{}"'.format(k)
                                    for k in unexpected_keys)))

def _to_cuda(state):
    """ usually load from cpu checkpoint but need to load to cuda """
    if isinstance(state, torch.Tensor):
        ret = state.cuda()  # assume propoerly set py torch.cuda.set_device
        if 'Half' in state.type():
            ret = ret.float()  # apex O2 requires it
        return ret
    elif isinstance(state, list):
        new_state = [_to_cuda(t) for t in state]
    elif isinstance(state, tuple):
        new_state = tuple(_to_cuda(t) for t in state)
    elif isinstance(state, dict):
        new_state = {n: _to_cuda(t) for n, t in state.items()}
    else:
        return state
    return new_state


def _to_cpu(state):
    """ store in cpu to avoid GPU0 device, fp16 to save space """
    if isinstance(state, torch.Tensor):
        ret = state.cpu()
        if 'Float' in state.type():
            ret = ret.half()
        return ret
    elif isinstance(state, list):
        new_state = [_to_cpu(t) for t in state]
    elif isinstance(state, tuple):
        new_state = tuple(_to_cpu(t) for t in state)
    elif isinstance(state, dict):
        new_state = {n: _to_cpu(t) for n, t in state.items()}
    else:
        return state
    return new_state
    
class TrainingRestorer(object):
    def __init__(self, opts, model, optimizer):
        if exists(opts.output_dir) and hvd.rank() == 0:
            restore_opts = json.load(
                open(f'{opts.output_dir}/log/hps.json', 'r'))
            with open(join(
                    opts.output_dir, 'log',
                    'restore_hps.json'), 'w') as writer:
                json.dump(vars(opts), writer, indent=4)
            assert vars(opts) == restore_opts
        # keep 2 checkpoints in case of corrupted
        self.save_path = f'{opts.output_dir}/restore.pt'
        self.backup_path = f'{opts.output_dir}/restore_backup.pt'
        self.model = model
        self.optimizer = optimizer
        self.save_steps = opts.save_steps
        self.amp = opts.fp16
        if exists(self.save_path) or exists(self.backup_path):
            LOGGER.info('found previous checkpoint. try to resume...')
            self.restore(opts)
        else:
            self.global_step = 0

    def step(self):
        self.global_step += 1
        if self.global_step % self.save_steps == 0:
            self.save()

    def save(self):
        checkpoint = {'global_step': self.global_step,
                      'model_state_dict': _to_cpu(self.model.state_dict()),
                      'optim_state_dict': _to_cpu(self.optimizer.state_dict())}
        if self.amp:
            checkpoint['amp_state_dict'] = amp_state_dict()
        if exists(self.save_path):
            os.rename(self.save_path, self.backup_path)
        torch.save(checkpoint, self.save_path)

    def restore(self, opts):
        try:
            checkpoint = torch.load(self.save_path)
        except Exception:
            checkpoint = torch.load(self.backup_path)
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(_to_cuda(checkpoint['model_state_dict']))
        self.optimizer.load_state_dict(
            _to_cuda(checkpoint['optim_state_dict']))
        if self.amp:
            amp_load_state_dict(checkpoint['amp_state_dict'])
        LOGGER.info(f'resume training from step {self.global_step}')
