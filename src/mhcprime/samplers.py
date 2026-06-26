import math
import numpy as np
from torch.utils.data import DataLoader, Sampler

class SimpleNegativeSampler(Sampler):
    """
    A simplified sampler for Pooled Simple Sampling (spss). For every epoch, it randomly selects a fixed number of positives and negatives. No cross-epoch tracking or state persistence.
    """
    def __init__(self, labels, batch_size, num_pos_per_epoch, 
                 neg_pos_ratio=1.0, seed=42):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.num_pos_per_epoch = num_pos_per_epoch
        self.neg_pos_ratio = neg_pos_ratio
        self.rng = np.random.default_rng(seed)
        
        # pre-calculate indices for fast access
        self.pos_indices = np.flatnonzero(self.labels == 1)
        self.neg_indices = np.flatnonzero(self.labels == 0)
        
        print(f"Simple Sampler initialized: {len(self.pos_indices)} Pos, {len(self.neg_indices)} Neg available.")

    def __iter__(self):
        """
        Randomly select fresh samples for this epoch and yield batches.
        """
        num_pos = min(self.num_pos_per_epoch, len(self.pos_indices))
        selected_pos = self.rng.choice(self.pos_indices, size=num_pos, replace=False)
        
        num_neg_needed = int(num_pos * self.neg_pos_ratio)
        num_neg = min(num_neg_needed, len(self.neg_indices))
        selected_neg = self.rng.choice(self.neg_indices, size=num_neg, replace=False)
        
        all_indices = np.concatenate([selected_pos, selected_neg])
        self.rng.shuffle(all_indices)
        
        for i in range(0, len(all_indices), self.batch_size):
            yield all_indices[i : i + self.batch_size].tolist()

    def __len__(self):
        """
        Return the number of batches in one epoch.
        """
        num_neg = int(self.num_pos_per_epoch * self.neg_pos_ratio)
        total_samples = self.num_pos_per_epoch + num_neg
        return math.ceil(total_samples / self.batch_size)

def make_simple_negative_loader(dataset, labels, batch_size, num_pos_per_epoch, 
                               neg_pos_ratio=1.0, num_workers=0, seed=42,
                               collate_fn=None, persistent_workers=True):
    """
    Create a loader with the simple negative sampler.
    """
    
    print("\n Creating SimpleNegativeSampler")
    sampler = SimpleNegativeSampler(
        labels=labels, 
        batch_size=batch_size,
        num_pos_per_epoch=num_pos_per_epoch,
        neg_pos_ratio=neg_pos_ratio,
        seed=seed
    )
    
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=persistent_workers
    )

    return loader, sampler