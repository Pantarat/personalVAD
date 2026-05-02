"""@package evaluate_models

Author: Simon Sedlacek
Email: xsedla1h@stud.fit.vutbr.cz

This script takes all the models present in the data/models directory and 
runs an evaluation of each model. Intended to be used as an automated alternative
to the personal_vad_evaluate.ipynb evaluation jupyter notebook.

Average precision scores are computed for each class, mean average precision is
computed across all classses. Additionally, raw classification accuracy and
confusion matrix is computed.

The results are written to stdout.

Usage: python src/evaluate_models.py 2>/dev/null

"""

import numpy as np
from copy import copy
import torch

from personal_vad import PersonalVAD, pad_collate
from vad_et import VadETDataset
from vad_set import VadSETDataset
from vad_st import VadSTDataset
from vad_xvector import VadETDatasetX
from vad_ivector import VadETDatasetI
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import average_precision_score, confusion_matrix, \
        precision_score, accuracy_score
import kaldiio
import os
import sys
from glob import glob
from decimal import Decimal, ROUND_HALF_UP

#NOTE: edit if needed...
batch_size_test = 64
test_dir = 'vad_set_overlap_trained/'
data = 'data/84_ov_test_ov50pct_main84_aug_100' # change this to match your data directory
results_dir = 'model_evaluation_results/' # directory to save results

def quantize(number, prec):
    """Round to two decimal places"""
    prec = f'1e-{prec}'
    return str(Decimal(number).quantize(Decimal(prec), ROUND_HALF_UP))


def parse_model_name(model):
    """Parse the model name string and determine its specifications.

    Args:
        model (str): Name of the model.
    
    Returns:
        tuple: A tuple containing:

        arch (str): Personal VAD architecture type.
        embed (str): Target speaker embedding type.
        use_fc (bool): Indicates whether the model uses the last hidden layer.
        linear (bool): Indicates whether the last hidden layer activation function
            is linear. If false, the activation is tanh.
        score_type (int): Indicates the scoring method used by the model.
        input_dim (int): Model input layer dimension.
    """

    # determine the embedding type
    if 'ivec' in model:
        embed = 'ivec'
    elif 'xvec' in model:
        embed = 'xvec'
    else:
        embed = 'dvec'

    # determine the architecture and the input layer dimension
    if 'ae' in model:
        arch = 'set_ae'
        input_dim = 105
        
    elif 'set' in model:
        arch = 'set'
        input_dim = 297

    elif 'st' in model:
        arch = 'st'
        input_dim = 41

    elif 'et' in model:
        arch = 'et'

        if embed == 'dvec':
            input_dim = 296
        elif embed == 'xvec':
            input_dim = 552
        else: # ivec
            input_dim = 440
    else:
        # unknown architecture..
        print(model, "other architecture...")
        return None

    # determine score type
    score_type = 0
    if arch in ['set', 'st']:
        if 'score0' in model: score_type = 0
        elif 'score1' in model: score_type = 1
        elif 'score2' in model: score_type = 2
        else:
            # score type not specified, default to score1
            score_type = 1
            print(f"  ℹ️  Score type not specified, defaulting to score_type=1")

    # determine activation
    if embed == 'dvec':
        if 'tanh' in model:
            linear = False
            use_fc = True
        elif 'linear' in model:
            linear = True
            use_fc = True
        elif 'lrelu' in model:
            print(model, "leaky relu...")
            return None
        else:
            # not using the hidden layer..
            linear = True
            use_fc = False
    else:
        if 'tanh' in model:
            linear = False
            use_fc = True
        elif 'linear' in model:
            linear = True
            use_fc = True
        else:
            linear = False
            use_fc = True

    return arch, embed, use_fc, linear, score_type, input_dim

if __name__ == '__main__':
    # save original working directory
    orig_dir = os.getcwd()
    
    # convert all relative paths to absolute paths based on execution directory
    data_path = os.path.join(orig_dir, data)
    test_dir_path = os.path.join(orig_dir, test_dir)
    results_path = os.path.join(orig_dir, results_dir)
    embeddings_base = os.path.join(orig_dir, 'data')
    
    # create results directory in the original working directory
    os.makedirs(results_path, exist_ok=True)
    
    # determine output filename based on data directory
    data_name = os.path.basename(data.rstrip('/'))
    results_file = os.path.join(results_path, f'eval_{data_name}.txt')
    # results_file = os.path.join(results_path, f'test.txt')
   
    # open results file
    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results for: {data_path}\n")
        f.write(f"{'='*60}\n\n")
    
    print(f"Saving results to: {results_file}")
    
    # get the model list (relative to test_dir)
    # models = glob('models/*pt')
    models = [
        "../data/eval_dir/models/vad_set_tanh_score1_10ep.pt",
        "vad_set_main84_25pcttrain_tanh_score1_500.pt",
        "vad_set_main84_50pcttrain_tanh_score1_500.pt",
        "vad_set_main84_75pcttrain_tanh_score1_500.pt",
        "vad_set_main84_100pcttrain_tanh_score1_500.pt",
    ]
    
    # convert model paths to absolute paths
    models = [os.path.join(test_dir_path, model) for model in models]

    # evaluate the models one by one...
    for model in models:
        # check if model file exists
        if not os.path.exists(model):
            print(f"⚠️  Model not found: {model}")
            continue
            
        print(f"\n🔍 Evaluating: {model}")
        
        # get the model information
        ret = parse_model_name(model)
        if ret == None: 
            print(f"⚠️  Could not parse model name: {model}")
            continue
        (arch, embed, use_fc, linear, score_type, input_dim) = ret

        # load the model
        net = PersonalVAD(input_dim=input_dim, hidden_dim=64, num_layers=2,
                out_dim=3, use_fc=use_fc, linear=linear)
        net.load_state_dict(torch.load(model))

        # create the corresponding dataset objects
        if arch == 'et':
            if embed == 'dvec':
                test_data = VadETDataset(data_path, os.path.join(embeddings_base, 'embeddings'))
            elif embed == 'xvec':
                test_data = VadETDatasetX(data_path, os.path.join(embeddings_base, 'embeddings_xvec_l2'))

            else: # ivec
                if 'l2' in model:
                    test_data = VadETDatasetI(data_path, os.path.join(embeddings_base, 'embeddings_ivec_l2'))
                else:
                    test_data = VadETDatasetI(data_path, os.path.join(embeddings_base, 'embeddings_ivec'))

        elif arch == 'set':
            test_data = VadSETDataset(data_path, os.path.join(embeddings_base, 'embeddings'), score_type)

        elif arch == 'st':
            test_data = VadSTDataset(data_path, score_type)
            
        else:
            # we should not get here
            continue

        test_loader = DataLoader(dataset=test_data, batch_size=batch_size_test,
                num_workers=2, shuffle=False, collate_fn=pad_collate)

        # set the device to cuda and move the model to the gpu
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        net = net.to(device)

        # mAP evaluation
        softmax = torch.nn.Softmax(dim=1)
        targets = []
        outputs = []

        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for x_padded, y_padded, x_lens, y_lens in test_loader:
                y_padded = y_padded.to(device)

                # pass the data through the model
                out_padded, _ = net(x_padded.to(device), x_lens, None)

                # value, index
                for j in range(out_padded.size(0)):
                    p = softmax(out_padded[j][:y_lens[j]])
                    
                    outputs.append(p.cpu().numpy())
                    targets.append(y_padded[j][:y_lens[j]].cpu().numpy())
                        
        targets = np.concatenate(targets)
        outputs = np.concatenate(outputs)

        # convert the target array to one hot
        targets_oh = np.eye(3)[targets]

        # and run the AP
        out_AP = average_precision_score(targets_oh, outputs, average=None)
        mAP = average_precision_score(targets_oh, outputs, average='micro')

        # prepare output string
        result_str = f'\n{model}\n'
        result_str += f'{out_AP}\n'
        result_str += f"mAP: {mAP}\n"

        # compute the confusion matrix
        classes = np.argmax(outputs, axis=1)
        cm = confusion_matrix(classes, targets, normalize='pred')

        result_str += "confusion\n"
        result_str += f"{cm}\n"

        # compute the accuracy score
        acc = accuracy_score(classes, targets) * 100
        result_str += f"accuracy {quantize(acc, 2)}\n"
        result_str += f"\n{'='*60}\n"

        # print to console
        print(result_str)

        # append to results file
        with open(results_file, 'a') as f:
            f.write(result_str)
    
    print(f"\n✅ All results saved to: {results_file}")
