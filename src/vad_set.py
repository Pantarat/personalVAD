"""@package vad_set

Author: Simon Sedlacek
Email: xsedla1h@stud.fit.vutbr.cz

This module implements the SET personal VAD architecture training loop.

The input for this architecture is a 297-dimensional feature vector combining
the 40-dimensional log Mel-filterbank energies, the 256-dimensional target
speaker d-vector representation and the array of speaker verification scores
for each frame.

"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import kaldiio
import argparse as ap

from sklearn.metrics import average_precision_score

import numpy as np
import os

from personal_vad import PersonalVAD, WPL, pad_collate
from dataset_utils import sample_evenly_across_augmentations

# model hyper parameters
num_epochs = 200
batch_size = 64
batch_size_test = 32

input_dim = 297
hidden_dim = 64
out_dim = 3
num_layers = 2
lr = 1e-3
SCHEDULER = True

# Early stopping and checkpointing
EARLY_STOPPING_PATIENCE = 5  # Stop if no improvement for N epochs
SAVE_EVERY_N_EPOCHS = 1  # Save checkpoint every N epochs
METRIC_FOR_BEST = 'mAP'  # 'mAP' or 'accuracy'

DATA_TRAIN = 'data/train'
DATA_TEST = 'data/test'
EMBED_PATH = 'embeddings'
MODEL_PATH = 'vad_set_main84_50pcttrain_tanh_score1_500.pt'
SAVE_MODEL = True

USE_WPL = True
NUM_WORKERS = 2

# Selects which of the scoring methods should be used...
# legend: scores[0,:] -> baseline, 1 -> partially-constant, 2 -> linearly-interpolated
SCORE_TYPE = 0

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
WPL_WEIGHTS = torch.tensor([1.0, 0.1, 1.0]).to(device)

class VadSETDataset(Dataset):
    """VadSET dataset class. Uses kaldi scp and ark files."""

    def __init__(self, root_dir, embed_path, score_type, max_utterances=None):
        self.root_dir = root_dir
        self.embed_path = embed_path
        self.score_type = score_type

        # load the scp files...
        self.fbanks = kaldiio.load_scp(f'{self.root_dir}/fbanks.scp')
        self.scores = kaldiio.load_scp(f'{self.root_dir}/scores.scp')
        self.labels = kaldiio.load_scp(f'{self.root_dir}/labels.scp')
        self.keys = np.array(list(self.fbanks)) # get all the keys
        
        # Limit number of utterances if specified
        if max_utterances is not None and max_utterances > 0:
            self.keys = sample_evenly_across_augmentations(self.keys, max_utterances)
        
        self.embed = kaldiio.load_scp(f'{self.embed_path}/dvectors.scp')

        # load the target speaker ids
        self.targets = {}
        with open(f'{self.root_dir}/targets.scp') as targets:
            for line in targets:
                (utt_id, target) = line.split()
                self.targets[utt_id] = target

    def __len__(self):
        return self.keys.size

    def __getitem__(self, idx):
        key = self.keys[idx]
        target = self.targets[key]
        x = self.fbanks[key]
        scores = self.scores[key][self.score_type,:]
        embed = self.embed[target]
        y = self.labels[key]

        # add the speaker verification scores array to the feature vector
        x = np.hstack((x, np.expand_dims(scores, 1)))

        # add the dvector array to the feature vector
        x = np.hstack((x, np.full((x.shape[0], 256), embed)))

        x = torch.from_numpy(x.copy()).float()
        y = torch.from_numpy(y.copy()).long()
        return x, y


if __name__ == '__main__':
    """ Model training  """

    # program arguments
    parser = ap.ArgumentParser(description="Train the VAD SET model.")
    parser.add_argument('--train_dir', type=str, default=DATA_TRAIN)
    parser.add_argument('--test_dir', type=str, default=DATA_TEST)
    parser.add_argument('--embed_path', type=str, default=EMBED_PATH)
    parser.add_argument('--score_type', type=int, default=SCORE_TYPE)
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--max_train_utterances', type=int, default=None,
                        help='Maximum number of training utterances to use (default: use all)')
    parser.add_argument('--max_test_utterances', type=int, default=None,
                        help='Maximum number of test utterances to use (default: use all)')
    parser.add_argument('--use_kaldi', action='store_true')
    parser.add_argument('--use_wpl', action='store_true')
    parser.add_argument('--nuse_fc', action='store_false')
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--nsave_model', action='store_false')
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    DATA_TRAIN = args.train_dir
    DATA_TEST = args.test_dir
    EMBED_PATH = args.embed_path
    SCORE_TYPE = args.score_type
    linear = args.linear
    USE_WPL = args.use_wpl
    SAVE_MODEL = args.nsave_model

    if SCORE_TYPE not in [0, 1, 2]:
        print(f"Error: invalid scoring type: {SCORE_TYPE}. The values have to be in {0, 1, 2}.")
        sys.exit(1)

    # Load the data and create DataLoader instances
    train_data = VadSETDataset(DATA_TRAIN, EMBED_PATH, SCORE_TYPE, max_utterances=args.max_train_utterances)
    test_data = VadSETDataset(DATA_TEST, EMBED_PATH, SCORE_TYPE, max_utterances=args.max_test_utterances)
    
    print(f"Training utterances: {len(train_data)}")
    print(f"Test utterances: {len(test_data)}")

    train_loader = DataLoader(
            dataset=train_data, num_workers=NUM_WORKERS, pin_memory=True,
            batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    test_loader = DataLoader(
            dataset=test_data, num_workers=NUM_WORKERS, pin_memory=True,
            batch_size=batch_size_test, shuffle=False, collate_fn=pad_collate)

    model = PersonalVAD(input_dim, hidden_dim, num_layers, out_dim, use_fc=args.nuse_fc, linear=linear).to(device)

    if USE_WPL:
        criterion = WPL(WPL_WEIGHTS)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if SCHEDULER:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    softmax = nn.Softmax(dim=1)
    
    # Early stopping tracking
    best_metric = 0.0  # Best validation metric (mAP or accuracy)
    epochs_without_improvement = 0
    best_epoch = 0
    
    # Create checkpoint directory
    checkpoint_dir = MODEL_PATH.rpartition('.')[0] + '_checkpoints'
    if SAVE_MODEL and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Train!!! hype!!!
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)
    print(f"Early stopping: enabled (patience={EARLY_STOPPING_PATIENCE}, metric={METRIC_FOR_BEST})")
    print(f"Periodic saving: every {SAVE_EVERY_N_EPOCHS} epoch(s)")
    
    for epoch in range(num_epochs):
        print(f"====== Starting epoch {epoch} ======")
        for batch, (x_padded, y_padded, x_lens, y_lens) in enumerate(train_loader):
            y_padded = y_padded.to(device)

            # pass the data through the model
            out_padded, _ = model(x_padded.to(device), x_lens, None)

            # compute the loss
            loss = 0
            for j in range(out_padded.size(0)):
                loss += criterion(out_padded[j][:y_lens[j]], y_padded[j][:y_lens[j]])

            loss /= batch_size # normalize for the batch
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch % 10 == 0:
                print(f'Batch: {batch}, loss = {loss:.4f}')

        if SCHEDULER and epoch < 2:
            scheduler.step() # learning rate adjust
            if epoch == 1:
                optimizer.param_groups[0]['lr'] = 5e-5
        if SCHEDULER and epoch == 7:
            optimizer.param_groups[0]['lr'] = 1e-5

        # Test the model after each epoch
        with torch.no_grad():
            print("testing...")
            n_correct = 0
            n_samples = 0
            targets = []
            outputs = []
            for x_padded, y_padded, x_lens, y_lens in test_loader:
                y_padded = y_padded.to(device)

                # pass the data through the model
                out_padded, _ = model(x_padded.to(device), x_lens, None)

                # value, index
                for j in range(out_padded.size(0)):
                    classes = torch.argmax(out_padded[j][:y_lens[j]], dim=1)
                    n_samples += y_lens[j]
                    n_correct += torch.sum(classes == y_padded[j][:y_lens[j]]).item()

                    # average precision
                    p = softmax(out_padded[j][:y_lens[j]])
                    outputs.append(p.cpu().numpy())
                    targets.append(y_padded[j][:y_lens[j]].cpu().numpy())

            acc = 100.0 * n_correct / n_samples
            print(f"accuracy = {acc:.2f}")

            # and run the AP
            targets = np.concatenate(targets)
            outputs = np.concatenate(outputs)
            targets_oh = np.eye(3)[targets]
            out_AP = average_precision_score(targets_oh, outputs, average=None)
            mAP = average_precision_score(targets_oh, outputs, average='micro')

            print(out_AP)
            print(f"mAP: {mAP}")
            
            # Early stopping and checkpointing
            current_metric = mAP if METRIC_FOR_BEST == 'mAP' else acc / 100.0
            
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                epochs_without_improvement = 0
                
                # Save best model
                if SAVE_MODEL:
                    path_seg = MODEL_PATH.split('/')[:-1]
                    if path_seg != []:
                        if not os.path.exists(MODEL_PATH.rpartition('/')[0]):
                            os.makedirs('/'.join(path_seg))
                    torch.save(model.state_dict(), MODEL_PATH)
                    print(f"✓ New best {METRIC_FOR_BEST}: {best_metric:.4f} - Model saved")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epoch(s) (best {METRIC_FOR_BEST}: {best_metric:.4f} at epoch {best_epoch})")
                
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
                    print(f"Best {METRIC_FOR_BEST}: {best_metric:.4f} (epoch {best_epoch})")
                    break
            
            # Periodic checkpoint saving
            if SAVE_MODEL and (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_metric': best_metric,
                    'metric_type': METRIC_FOR_BEST,
                    'accuracy': acc,
                    'mAP': mAP
                }, checkpoint_path)
                print(f"📁 Checkpoint saved: {checkpoint_path}")
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n💾 Best model saved to: {MODEL_PATH}")
    print(f"📊 Best {METRIC_FOR_BEST}: {best_metric:.4f} (epoch {best_epoch})")
    if SAVE_MODEL:
        print(f"📁 Checkpoints saved to: {checkpoint_dir}")

