"""CBIS-DDSM CC+MLO paired-view dataset."""

import os
import re
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import torchvision.transforms as T
from collections import defaultdict, Counter


class CBISDDSMDataset(Dataset):
    """CBIS-DDSM Dataset with binary risk mapping."""
    
    def __init__(self, root_dir, split='train', transform=None, img_size=224, 
                 val_ratio=0.15, seed=42):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.seed = seed
        self.img_dir = self.root_dir / 'patchs'
        
        self._load_data(val_ratio)
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_transforms(split == 'train')
    
    def _load_data(self, val_ratio):
        if self.split in ['train', 'val']:
            csv_path = self.root_dir / 'mass_case_description_train_set.csv'
        else:
            csv_path = self.root_dir / 'mass_case_description_test_set.csv'
        
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip().str.replace(' ', '_')
        
        all_images = list(self.img_dir.glob('*.png'))
        
        self.image_info = {}
        for img_path in all_images:
            info = self._parse_filename(img_path.name)
            if info:
                key = (info['patient_id'], info['laterality'], info['view'], info['abnormality_id'])
                self.image_info[key] = {'path': img_path, **info}
        
        self._create_pairs()
        
        if self.split in ['train', 'val']:
            self._split_train_val(val_ratio)
    
    def _parse_filename(self, filename):
        pattern = r'Mass-(Training|Test)_P_(\d+)_(LEFT|RIGHT)_(CC|MLO)_(\d+)_'
        match = re.match(pattern, filename)
        if match:
            return {
                'patient_id': f"P_{match.group(2)}",
                'laterality': match.group(3),
                'view': match.group(4),
                'abnormality_id': int(match.group(5)),
            }
        return None
    
    def _get_label(self, patient_id, laterality, view, abnorm_id):
        mask = (
            (self.df['patient_id'] == patient_id) &
            (self.df['left_or_right_breast'] == laterality) &
            (self.df['image_view'] == view) &
            (self.df['abnormality_id'] == abnorm_id)
        )
        rows = self.df[mask]
        if len(rows) == 0:
            return -1
        
        # Use PATHOLOGY label (actual diagnosis) instead of shape-based mapping
        pathology = str(rows.iloc[0]['pathology']).upper()
        
        if 'MALIGNANT' in pathology:
            return 1
        elif 'BENIGN' in pathology:
            return 0
        return -1  # Skip uncertain cases
    
    def _create_pairs(self):
        self.pairs = []
        groups = defaultdict(dict)
        
        for key, info in self.image_info.items():
            patient_id, laterality, view, abnorm_id = key
            group_key = (patient_id, laterality, abnorm_id)
            groups[group_key][view] = info
        
        for group_key, views in groups.items():
            patient_id, laterality, abnorm_id = group_key
            
            if 'CC' in views and 'MLO' in views:
                label = self._get_label(patient_id, laterality, 'CC', abnorm_id)
                if label >= 0:
                    self.pairs.append({
                        'patient_id': patient_id,
                        'cc_path': views['CC']['path'],
                        'mlo_path': views['MLO']['path'],
                        'label': label
                    })
    
    def _split_train_val(self, val_ratio):
        patients = list(set(p['patient_id'] for p in self.pairs))
        np.random.seed(self.seed)
        np.random.shuffle(patients)
        
        n_val = int(len(patients) * val_ratio)
        val_patients = set(patients[:n_val])
        
        if self.split == 'train':
            self.pairs = [p for p in self.pairs if p['patient_id'] not in val_patients]
        else:
            self.pairs = [p for p in self.pairs if p['patient_id'] in val_patients]
    
    def _get_transforms(self, is_training):
        if is_training:
            return T.Compose([
                T.Resize((self.img_size + 20, self.img_size + 20)),
                T.RandomCrop((self.img_size, self.img_size)),
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.2),
                T.RandomRotation(20),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        return T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        img_cc = Image.open(pair['cc_path']).convert('RGB')
        img_mlo = Image.open(pair['mlo_path']).convert('RGB')
        
        if self.transform:
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            np.random.seed(seed)
            img_cc = self.transform(img_cc)
            torch.manual_seed(seed)
            np.random.seed(seed)
            img_mlo = self.transform(img_mlo)
        
        return {
            'img_cc': img_cc,
            'img_mlo': img_mlo,
            'label': torch.tensor(pair['label'], dtype=torch.long),
            'patient_id': pair['patient_id'],
        }


def get_dataloaders(data_root, batch_size=16, num_workers=4, oversample=True):
    """Get dataloaders with optional oversampling."""
    
    train_dataset = CBISDDSMDataset(data_root, split='train')
    val_dataset = CBISDDSMDataset(data_root, split='val')
    test_dataset = CBISDDSMDataset(data_root, split='test')
    
    # Oversampling for training
    if oversample:
        labels = [p['label'] for p in train_dataset.pairs]
        class_counts = Counter(labels)
        weights = [1.0 / class_counts[l] for l in labels]
        sampler = WeightedRandomSampler(weights, len(weights) * 2, replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    
    # Class weights
    train_labels = [p['label'] for p in train_dataset.pairs]
    counts = np.bincount(train_labels, minlength=2)
    weights = len(train_labels) / (2 * counts + 1e-6)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'class_weights': class_weights,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }
