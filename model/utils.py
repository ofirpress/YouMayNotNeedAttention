import time
import os, shutil
import torch


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def batchify(data_src, data_trg, bsz, args):
    assert (data_src.size(0) == data_trg.size(0)) # src / trg data must be of equal lengths

    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data_src.size(0) // bsz

    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data_src = data_src.narrow(0, 0, nbatch * bsz)
    data_trg = data_trg.narrow(0, 0, nbatch * bsz)

 
    # Evenly divide the data across the bsz batches.
    data_src = data_src.view(bsz, -1).t().contiguous()
    data_trg = data_trg.view(bsz, -1).t().contiguous()
    #if args.cuda:
    #    data_src = data_src.cuda()
    #    data_trg = data_trg.cuda()
    return data_src, data_trg


def get_batch(source_src, source_trg, i, args, seq_len=None, evaluation=False):
    
    assert(source_src.size() == source_trg.size())
    seq_len = min(seq_len if seq_len else args.bptt, len(source_src) - 1 - i)
    data = source_src[i+1:i+1+seq_len]
    prev_target = source_trg[i:i+seq_len]
    target = source_trg[i+1:i+1+seq_len].view(-1)
    
    if args.cuda:
        data = data.cuda()
        prev_target = prev_target.cuda()
        target = target.cuda()

    return data, prev_target, target


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(model, optimizer, path, suffix=''):
    torch.save(model, os.path.join(path, 'model' + suffix + '.pt'))
    torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer' + suffix + '.pt'))

