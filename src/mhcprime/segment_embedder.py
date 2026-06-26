import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentEmbedding(nn.Module):
    """
    Simple segment ID embedding layer that adds distinct segment vectors to 
    different parts of the model input (peptide vs MHC).
    """
    def __init__(self, embedding_dim=400, random_seed=None):
        super().__init__()
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        # Create embedding table with just 2 entries:
        # 0 for peptide, 1 for MHC
        self.segment_embedding = nn.Embedding(2, embedding_dim)

    def forward(self, x, segment_id):
        """
        Add segment embeddings to input.
        
        Args:
            x: Input tensor
            segment_id: Integer (0 for peptide, 1 for MHC)
            
        Returns:
            x with segment embeddings added
        """
        # Handle different input shapes
        if len(x.shape) == 3:
            # Standard case: [batch_size, seq_len, embed_dim]
            batch_size, seq_len, _ = x.shape
            segment_ids = torch.full((batch_size, seq_len), segment_id, 
                                    dtype=torch.long, device=x.device)
            
            # Get segment embeddings
            segment_emb = self.segment_embedding(segment_ids)
            
            # Add to input
            return x + segment_emb
        
        elif len(x.shape) == 4:
            # Handle multi-allele case: [batch_size, num_alleles, seq_len, embed_dim]
            batch_size, num_alleles, seq_len, embed_dim = x.shape
            
            # Reshape to [batch_size*num_alleles, seq_len, embed_dim]
            x_reshaped = x.reshape(-1, seq_len, embed_dim)
            
            # Create segment IDs tensor for all alleles
            segment_ids = torch.full((batch_size*num_alleles, seq_len), segment_id, 
                                    dtype=torch.long, device=x.device)
            
            # Get segment embeddings
            segment_emb = self.segment_embedding(segment_ids)
            
            # Add to input
            result = x_reshaped + segment_emb
            
            # Reshape back to original shape
            return result.reshape(batch_size, num_alleles, seq_len, embed_dim)
        
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}. Expected 3D or 4D tensor.")