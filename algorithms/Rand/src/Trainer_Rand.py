import logging
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from algorithms.Rand.src.dataloaders import dataloader_factory
from algorithms.Rand.src.models import model_factory
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import random
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

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

class Trainer_Rand:
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
            
        tr_sample_paths, tr_class_labels = set_samples_labels(self.args.train_meta)
        test_sample_paths, test_class_labels = set_samples_labels(self.args.test_meta)
        self.cifar10_train = dataloader_factory.get_train_dataloader(self.args.dataset)(
                        src_path=self.args.data_path, sample_paths=tr_sample_paths, class_labels=tr_class_labels)
        self.cifar10_unlabeled = dataloader_factory.get_test_dataloader(self.args.dataset)(
            src_path=self.args.data_path, sample_paths=tr_sample_paths, class_labels=tr_class_labels)
        self.cifar10_test  = dataloader_factory.get_test_dataloader(self.args.dataset)(
                        src_path=self.args.data_path, sample_paths=test_sample_paths, class_labels=test_class_labels)     
        
        indices = list(range(50000))
        random.shuffle(indices)
        self.labeled_set = indices[:self.args.number_start]
        self.unlabeled_set = indices[self.args.number_start:]
        self.train_loader = DataLoader(self.cifar10_train, batch_size = self.args.batch_size, 
                    sampler=SubsetRandomSampler(self.labeled_set), 
                    pin_memory=True)

        self.val_loader = DataLoader(self.cifar10_test, batch_size = self.args.batch_size)    
        self.test_loader = DataLoader(self.cifar10_test, batch_size = self.args.batch_size)

    def set_writer(self, log_dir):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        shutil.rmtree(log_dir)
        return SummaryWriter(log_dir)

    def train(self):
        for cycle in range(self.args.active_cycle):
            self.model = model_factory.get_model(self.args.model)(num_classes = 10).to(self.device)
            self.criterion = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate,
                                momentum=0.9, weight_decay=5e-4)
            sched_backbone = torch.optim.lr_scheduler.MultiStepLR(optim_backbone, milestones=[160])
            self.val_acc_max = 0

            for epoch in range(self.args.epochs):
                sched_backbone.step()
                self.model.train()

                n_class_corrected = 0
                total_classification_loss = 0
                total_samples = 0
                for (samples, labels) in tqdm(self.train_loader, leave = False, total = len(self.train_loader)):
                    samples, labels = samples.to(self.device), labels.to(self.device)
                    predicted_classes, features = self.model(samples)
                    target_loss = self.criterion(predicted_classes, labels)

                    m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
                    loss = m_backbone_loss
                    
                    optim_backbone.zero_grad()
                    loss.backward()
                    optim_backbone.step()

                    total_classification_loss += torch.sum(m_backbone_loss).item()
                    _, predicted_classes = torch.max(predicted_classes, 1)
                    n_class_corrected += (predicted_classes == labels).sum().item()
                    total_samples += len(samples)
                
                if epoch % self.args.step_eval == 0:
                    self.writer.add_scalar("Accuracy/train", 100.0 * n_class_corrected / total_samples, epoch)
                    self.writer.add_scalar("Loss/train", total_classification_loss / total_samples, epoch)
                    logging.info(
                        "Train set: Epoch: [{}/{}]\tAccuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}".format(
                            epoch,
                            self.args.epochs,
                            n_class_corrected,
                            total_samples,
                            100.0 * n_class_corrected / total_samples,
                            total_classification_loss / total_samples,
                        )
                    )
                    self.evaluate(epoch)

                n_class_corrected = 0
                total_classification_loss = 0
                total_samples = 0

            self.test()
            random.shuffle(self.unlabeled_set)
            subset = self.unlabeled_set[:10000]
            
            self.labeled_set += list(torch.tensor(subset)[-self.args.number_query:].numpy())
            self.unlabeled_set = list(torch.tensor(subset)[:-self.args.number_query].numpy()) + self.unlabeled_set[10000:]
            self.train_loader = DataLoader(self.cifar10_train, batch_size = self.args.batch_size, 
                                sampler=SubsetRandomSampler(self.labeled_set), 
                                pin_memory=True)
            del self.model
            
    def evaluate(self, epoch):
        self.model.eval()

        n_class_corrected = 0
        total_classification_loss = 0
        with torch.no_grad():
            for (samples, labels) in self.val_loader:
                samples, labels = samples.to(self.device), labels.to(self.device)
                predicted_classes, _ = self.model(samples)
                classification_loss = self.criterion(predicted_classes, labels)
                m_backbone_loss = torch.sum(classification_loss) / classification_loss.size(0)
                total_classification_loss += torch.sum(m_backbone_loss).item()

                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()

        self.writer.add_scalar("Accuracy/validate", 100.0 * n_class_corrected / len(self.val_loader.dataset), epoch)
        self.writer.add_scalar("Loss/validate", total_classification_loss / len(self.val_loader.dataset), epoch)
        logging.info(
            "Val set: Accuracy: {}/{} ({:.2f}%)\tLoss: {:.6f}".format(
                n_class_corrected,
                len(self.val_loader.dataset),
                100.0 * n_class_corrected / len(self.val_loader.dataset),
                total_classification_loss / len(self.val_loader.dataset),
            )
        )

        val_acc = n_class_corrected / len(self.val_loader.dataset)
        val_loss = total_classification_loss / len(self.val_loader.dataset)

        self.model.train()
        if self.val_acc_max < val_acc:
            self.val_acc_max = val_acc
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
        total = 0
        n_class_corrected = 0
        with torch.no_grad():
            for (samples, labels) in self.test_loader:
                samples, labels = samples.to(self.device), labels.to(self.device)
                predicted_classes, _ = self.model(samples)
                _, predicted_classes = torch.max(predicted_classes, 1)
                n_class_corrected += (predicted_classes == labels).sum().item()
        
        logging.info(
            "Test set {}: Accuracy: {}/{} ({:.2f}%)".format(
                len(self.labeled_set),
                n_class_corrected,
                len(self.test_loader.dataset),
                100.0 * n_class_corrected / len(self.test_loader.dataset),
            )
        )