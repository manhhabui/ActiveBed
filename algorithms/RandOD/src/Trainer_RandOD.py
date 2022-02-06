import logging
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from algorithms.RandOD.src.dataloaders import dataloader_factory
from algorithms.RandOD.src.models import model_factory
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import random
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from algorithms.RandOD.src.utils import *

class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, device, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.device = device
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(self.device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(self.device)  # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(self.device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS
        loc_loss = torch.zeros((batch_size)).to(self.device)  # (N)
        # Localization loss is computed only over positive (non-background) priors
        for i in range(batch_size):
            loc_loss[i] = self.smooth_l1(predicted_locs[i][positive_priors[i]], true_locs[i][positive_priors[i]])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,conf_loss_neg
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive        
        conf_loss_pos = torch.zeros((batch_size)).to(self.device)  # (N)
        for i in range(batch_size):
            # print(conf_loss_all[i][positive_priors[i]].shape)
            conf_loss_pos[i] = conf_loss_all[i][positive_priors[i]].sum()

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(self.device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        
        conf_loss_hard_neg = torch.zeros((batch_size)).to(self.device)  # (N)
        for i in range(batch_size):
            conf_loss_hard_neg[i] = conf_loss_neg[i][hard_negatives[i]].sum()

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = torch.zeros((batch_size)).to(self.device)  # (N)
        for i in range(batch_size):
            conf_loss[i] = (conf_loss_hard_neg[i] + conf_loss_pos[i]) / n_positives[i].float()

        return conf_loss + self.alpha * loc_loss

class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)

def set_samples_labels(meta_filename):
    sample_paths, class_labels = [], []
    column_names = ["filename", "class_label"]
    data_frame = pd.read_csv(meta_filename, header=None, names=column_names, sep="\s+")
    sample_paths = data_frame["filename"]
    class_labels = data_frame["class_label"]

    return sample_paths, class_labels

class Trainer_RandOD:
    def __init__(self, args, device, exp_idx):
        self.args = args
        self.device = device
        self.writer = self.set_writer(
            log_dir="algorithms/"
            + self.args.algorithm
            + "/results/tensorboards/"
            + self.args.exp_name
            + "_"
            + exp_idx
            + "/"
        )
        self.checkpoint_name = (
            "algorithms/" + self.args.algorithm + "/results/checkpoints/" + self.args.exp_name + "_" + exp_idx
        )
        self.plot_dir = (
            "algorithms/" + self.args.algorithm + "/results/plots/" + self.args.exp_name + "_" + exp_idx + "/"
        ) 
            
        self.voc_train = dataloader_factory.get_train_dataloader(self.args.dataset)(
                        data_folder = self.args.data_path, split = 'train', keep_difficult = True)
        self.voc_unlabeled = dataloader_factory.get_test_dataloader(self.args.dataset)(
                        data_folder = self.args.data_path, split = 'train', keep_difficult = True)
        self.voc_test  = dataloader_factory.get_test_dataloader(self.args.dataset)(
                        data_folder = self.args.data_path, split = 'test', keep_difficult = True)     

        indices = list(range(len(self.voc_train)))
        random.shuffle(indices)
        self.labeled_set = indices[:self.args.number_start]
        self.unlabeled_set = indices[self.args.number_start:]
        self.train_loader = DataLoader(self.voc_train, batch_size = self.args.batch_size, 
                    sampler=SubsetRandomSampler(self.labeled_set), collate_fn=self.voc_train.collate_fn, num_workers = 4,
                    pin_memory=True)

        self.val_loader = DataLoader(self.voc_test, batch_size = self.args.batch_size, collate_fn=self.voc_test.collate_fn, num_workers = 4)    
        self.test_loader = DataLoader(self.voc_test, batch_size = self.args.batch_size, collate_fn=self.voc_test.collate_fn, num_workers = 4)

    def set_writer(self, log_dir):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        shutil.rmtree(log_dir)
        return SummaryWriter(log_dir)

    def train(self):
        for cycle in range(self.args.active_cycle):
            self.model = model_factory.get_model(self.args.model)(n_classes = 21).to(self.device)
            self.criterion = MultiBoxLoss(self.model.priors_cxcy, self.device).to(self.device)
            optim_backbone = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate,
                                momentum=0.9, weight_decay=5e-4)
            sched_backbone = torch.optim.lr_scheduler.MultiStepLR(optim_backbone, milestones=[240])
            self.val_loss_min = np.Inf

            for epoch in range(self.args.epochs):
                sched_backbone.step()
                self.model.train()

                total_loss = 0
                total_samples = 0
                for (samples, boxes, labels, _) in tqdm(self.train_loader, leave = False, total = len(self.train_loader)):
                    samples = samples.to(self.device)
                    boxes = [b.to(self.device) for b in boxes]
                    labels = [l.to(self.device) for l in labels]
                    predicted_locs, predicted_scores = self.model(samples)
                    target_loss = self.criterion(predicted_locs, predicted_scores, boxes, labels) 
                    m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
                    optim_backbone.zero_grad()
                    m_backbone_loss.backward()
                    optim_backbone.step()

                    total_loss += torch.sum(m_backbone_loss).item()
                    total_samples += len(samples)
                
                if epoch % self.args.step_eval == 0:
                    self.writer.add_scalar("Loss/train", total_loss / total_samples, epoch)
                    logging.info(
                        "Train set: Epoch: [{}/{}]\tLoss: {:.6f}".format(
                            epoch,
                            self.args.epochs,
                            total_loss / total_samples,
                        )
                    )
                    self.evaluate(epoch)

                total_loss = 0
                total_samples = 0

            self.test()
            random.shuffle(self.unlabeled_set)
            subset = self.unlabeled_set[:10000]
            
            self.labeled_set += list(torch.tensor(subset)[-self.args.number_query:].numpy())
            self.unlabeled_set = list(torch.tensor(subset)[:-self.args.number_query].numpy()) + self.unlabeled_set[10000:]
            self.train_loader = DataLoader(self.voc_train, batch_size = self.args.batch_size, 
                                sampler=SubsetRandomSampler(self.labeled_set), collate_fn=self.voc_train.collate_fn, num_workers = 4,
                                pin_memory=True)
            del self.model
            
    def evaluate(self, epoch):
        self.model.eval()

        total_loss = 0
        with torch.no_grad():
            for (samples, boxes, labels, _) in self.val_loader:
                samples = samples.to(self.device)
                boxes = [b.to(self.device) for b in boxes]
                labels = [l.to(self.device) for l in labels]
                predicted_locs, predicted_scores = self.model(samples)
                target_loss = self.criterion(predicted_locs, predicted_scores, boxes, labels) 
                m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
                total_loss += torch.sum(m_backbone_loss).item()

        self.writer.add_scalar("Loss/validate", total_loss / len(self.val_loader.dataset), epoch)
        logging.info(
            "Val set: Loss: {:.6f}".format(
                total_loss / len(self.val_loader.dataset),
            )
        )

        val_loss = total_loss / len(self.val_loader.dataset)

        self.model.train()
        if self.val_loss_min > val_loss:
            self.val_loss_min = val_loss
            torch.save(
                {
                    "model_state_dict": self.model.state_dict()
                },
                self.checkpoint_name + ".pt",
            )
    
    def test(self):
        checkpoint = torch.load(self.checkpoint_name + ".pt")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        det_boxes = list()
        det_labels = list()
        det_scores = list()
        true_boxes = list()
        true_labels = list()
        true_difficulties = list()

        with torch.no_grad():
            for i, (samples, boxes, labels, difficulties) in enumerate(tqdm(self.test_loader, desc='Evaluating')):
                samples = samples.to(self.device)
                boxes = [b.to(self.device) for b in boxes]
                labels = [l.to(self.device) for l in labels]
                difficulties = [d.to(self.device) for d in difficulties]
                predicted_locs, predicted_scores = self.model(samples)

                det_boxes_batch, det_labels_batch, det_scores_batch = self.model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)

                det_boxes.extend(det_boxes_batch)
                det_labels.extend(det_labels_batch)
                det_scores.extend(det_scores_batch)
                true_boxes.extend(boxes)
                true_labels.extend(labels)
                true_difficulties.extend(difficulties)
        
            APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
        
        logging.info(
            "Test set {}: Mean Average Precision (mAP): {:.3f}".format(
                len(self.labeled_set),
                mAP
            )
        )