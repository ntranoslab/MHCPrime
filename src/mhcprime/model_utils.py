from dataclasses import dataclass
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import _mhc_column_for_allele

"""
BOS token is at idx 0 and padding token is $ at idx 1.
"""
@dataclass
class AminoAcidTokenizer:
    def __init__(self):
        self.aa_tokens = ['<BOS>', '$', '*'] + list('ACDEFGHIKLMNPQRSTVWYUX') # Added mask token '*'
        self.token2idx = {token: idx for idx, token in enumerate(self.aa_tokens)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.pad_idx = self.token2idx['$']
        self.bos_idx = self.token2idx['<BOS>']
        self.mask_idx = self.token2idx['*']  # New mask token index
        
    def encode(self, sequence: str) -> torch.Tensor:
        """Encodes a sequence into a PyTorch tensor of token indices."""
        token_ids = [self.token2idx[aa] for aa in sequence]
        return torch.tensor(token_ids, dtype=torch.long)  
    
    def decode(self, indices: torch.Tensor) -> str:
        """Decodes a tensor of token indices back into an amino acid sequence."""
        return ''.join(self.idx2token[idx.item()] for idx in indices) 
    
    @property
    def vocab_size(self) -> int:
        return len(self.aa_tokens)

class PeptideMHCDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AminoAcidTokenizer, feature_tokenizers=None, feature_names=None, include_hardness_score=False, include_anchor_flag=False, include_domain_id=False, include_prior_score=False):
        self.df = df
        self.tokenizer = tokenizer
        self.feature_tokenizers = feature_tokenizers or {}
        self.feature_names = feature_names or []
        self.include_hardness_score = include_hardness_score
        self.include_anchor_flag=include_anchor_flag
        self.include_domain_id=include_domain_id
        self.include_prior_score=include_prior_score
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        n_flank = self.tokenizer.encode(row['n_flank']) 
        peptide = self.tokenizer.encode(row['seq'])     
        c_flank = self.tokenizer.encode(row['c_flank']) 

        peptide_tokens = torch.cat([n_flank, peptide, c_flank]) 

        # get feature values if requested and available
        feature_values = {}
        for feature_name in self.feature_names:
            if feature_name in row:
                feature_values[f"peptide_{feature_name}"] = torch.tensor(row[feature_name], dtype=torch.float)

        sa_ma = row['sa_ma']

        if sa_ma == "SA":
            # allele_type = row['allele'][0]
            # mhc_col = f"mhc_{allele_type.lower()}_1"
            mhc_col = _mhc_column_for_allele(row["allele"])
            mhc_seq = row[mhc_col]
            mhc_tokens = self.tokenizer.encode(mhc_seq)
            mhc_list = torch.stack([mhc_tokens])  
            
            # Create item with standard fields
            item = {
                'peptide': peptide_tokens, 
                'mhc_list': mhc_list, 
                'label': torch.tensor(row['label'], dtype=torch.float),
                'allele': row['allele'], # handle alleles for MA as well
                'original_idx': idx
            }

            # Add hardness score if needed
            if self.include_hardness_score and 'hardness_score' in row:
                hardness_score = float(row['hardness_score'])
                item['hardness_score'] = torch.tensor(hardness_score, dtype=torch.float)

            # Add anchor flag if needed
            if self.include_anchor_flag and 'contains_canonical_anchors' in row:
                anchor_flag = int(row['contains_canonical_anchors'])
                item['anchor_flag'] = torch.tensor(anchor_flag, dtype=torch.int)

            # if using domain id for conditional transformer for MS vs nonMS (11.10.25)
            if self.include_domain_id and 'domain_id' in row:
                domain_id = float(row['domain_id'])
                item['domain_id'] = torch.tensor(domain_id, dtype=torch.float)

            # for hard pos/neg mining
            if "example_id" in self.df.columns.tolist():
                example_id = int(row['example_id'])
                item['example_id'] = torch.tensor(example_id, dtype=torch.long)

            # for log odds prior scores
            if self.include_prior_score and 'prior_score' in row:
                # ensure it is float32
                item['prior_score'] = torch.tensor(row['prior_score'], dtype=torch.float)


            # add feature values if any
            if feature_values:
                item.update(feature_values)
            
            # add feature tokenization if feature tokenizers are provided
            if self.feature_tokenizers:
                feature_data = {}
                
                for feature_name, feature_tokenizer in self.feature_tokenizers.items():
                    feature_data[feature_name] = {}
                    
                    # tokenize peptide with feature tokenizer
                    feature_n_flank = feature_tokenizer.encode(row['n_flank'])
                    feature_peptide = feature_tokenizer.encode(row['seq'])
                    feature_c_flank = feature_tokenizer.encode(row['c_flank'])
                    feature_peptide_tokens = torch.cat([feature_n_flank, feature_peptide, feature_c_flank])
                    feature_data[feature_name]['peptide'] = feature_peptide_tokens
                    
                    # tokenize MHC with feature tokenizer
                    feature_mhc_tokens = feature_tokenizer.encode(mhc_seq)
                    feature_data[feature_name]['mhc'] = torch.stack([feature_mhc_tokens])
                
                item['feature_data'] = feature_data
            
            return item
        
        # multi-allele case
        alleles = row['allele'].split("_")
        mhc_seqs = set()
        for allele in alleles:
            allele_type = allele[0]
            for i in [1, 2]:
                mhc_col = f"mhc_{allele_type.lower()}_{i}"
                if pd.notna(row[mhc_col]):
                    mhc_seqs.add(row[mhc_col])
        
        mhc_tokens_list = [self.tokenizer.encode(mhc_seq) for mhc_seq in mhc_seqs]
        mhc_list = torch.stack(mhc_tokens_list)  

        # create item with standard fields
        item = {
            'peptide': peptide_tokens, 
            'mhc_list': mhc_list, 
            'label': torch.tensor(row['label'], dtype=torch.float),
            'original_idx': idx
        }
        
        # add feature values if any
        if feature_values:
            item.update(feature_values)
        
        # handle feature tokenization for multi-allele case
        if self.feature_tokenizers:
            feature_data = {}
            
            for feature_name, feature_tokenizer in self.feature_tokenizers.items():
                feature_data[feature_name] = {}
                
                # tokenize peptide with feature tokenizer
                feature_n_flank = feature_tokenizer.encode(row['n_flank'])
                feature_peptide = feature_tokenizer.encode(row['seq'])
                feature_c_flank = feature_tokenizer.encode(row['c_flank'])
                feature_peptide_tokens = torch.cat([feature_n_flank, feature_peptide, feature_c_flank])
                feature_data[feature_name]['peptide'] = feature_peptide_tokens
                
                # tokenize each MHC with feature tokenizer
                feature_mhc_tokens_list = [feature_tokenizer.encode(mhc_seq) for mhc_seq in mhc_seqs]
                feature_mhc_list = torch.stack(feature_mhc_tokens_list)
                feature_data[feature_name]['mhc'] = feature_mhc_list
            
            item['feature_data'] = feature_data
        
        return item

def collate_fn(batch):
    max_peptide_len = 34
    max_mhc_len = 34
    max_num_alleles = max(len(x['mhc_list']) for x in batch)  

    peptides = torch.stack([
        torch.cat([x['peptide'], torch.full((max_peptide_len - len(x['peptide']),), 1, dtype=torch.long)])
        for x in batch
    ])

    peptide_masks = (peptides != 1)

    # pad MHC sequences to have the same number of alleles per batch
    mhc_list = torch.stack([
        torch.cat([
            torch.cat([mhc, torch.full((max_mhc_len - len(mhc),), 1, dtype=torch.long)]).unsqueeze(0)
            for mhc in x['mhc_list']
        ] + [
            torch.full((1, max_mhc_len), 1, dtype=torch.long) 
        ] * (max_num_alleles - len(x['mhc_list'])), dim=0)  
        for x in batch
    ])  # Shape: [batch_size, max_num_alleles, max_mhc_len]

    mhc_mask_list = (mhc_list != 1)

    labels = torch.tensor([x['label'] for x in batch], dtype=torch.float)
    
    # collect original indices
    original_indices = torch.tensor([x['original_idx'] for x in batch], dtype=torch.long)
    alleles = [x['allele'] for x in batch] 

    result = {
        'peptide': peptides,
        'peptide_mask': peptide_masks,  
        'mhc_list': mhc_list,  
        'mhc_mask_list': mhc_mask_list,  
        'label': labels,
        'original_idx': original_indices,
        'allele': alleles
    }

    # add hardness scores if available
    if 'hardness_score' in batch[0]:
        hardness_scores = torch.tensor([float(x['hardness_score']) for x in batch], dtype=torch.float)
        # Ensure it's a 1D tensor
        hardness_scores = hardness_scores.view(-1)
        result['hardness_score'] = hardness_scores

    # add anchor flags if available
    if 'anchor_flag' in batch[0]:
        anchor_flags = torch.tensor([int(x['anchor_flag']) for x in batch], dtype=torch.int)
        result['anchor_flag'] = anchor_flags

    # add domain id if available (11.10.25)
    if 'domain_id' in batch[0]:
        domain_ids = torch.tensor([int(x['domain_id']) for x in batch], dtype=torch.int)
        result['domain_id'] = domain_ids

    if 'example_id' in batch[0]:
        example_ids = torch.as_tensor([b['example_id'] for b in batch], dtype=torch.long)
        result['example_id'] = example_ids

    if 'prior_score' in batch[0]:
        prior_scores = torch.tensor([float(x['prior_score']) for x in batch], dtype=torch.float)
        result['prior_score'] = prior_scores
    
    # collect feature values if present
    for feature_key in batch[0].keys():
        if feature_key.startswith('peptide_') and feature_key not in ['peptide', 'peptide_mask']:
            # extract feature values for all samples
            feature_values = torch.tensor([x[feature_key] for x in batch if feature_key in x], dtype=torch.float)
            if len(feature_values) == len(batch):  # only include if all samples have this feature
                result[feature_key] = feature_values
    
    # process feature data if any entry has it
    has_features = any('feature_data' in x for x in batch)
    if has_features:
        # find all unique feature names across batch
        feature_names = set()
        for x in batch:
            if 'feature_data' in x:
                feature_names.update(x['feature_data'].keys())
        
        feature_data = {}
        for feature_name in feature_names:
            feature_data[feature_name] = {}
            
            # process peptide features
            feature_peptides = torch.stack([
                torch.cat([
                    x.get('feature_data', {}).get(feature_name, {}).get('peptide', 
                          torch.full((1,), 1, dtype=torch.long)),  # default to padding if missing
                    torch.full((max_peptide_len - len(x.get('feature_data', {}).get(feature_name, {}).get('peptide', 
                                torch.full((1,), 1, dtype=torch.long))),), 1, dtype=torch.long)
                ])
                for x in batch
            ])
            feature_data[feature_name]['peptide'] = feature_peptides
            
            # process MHC features
            feature_mhc_list = torch.stack([
                torch.cat([
                    torch.cat([
                        feature_mhc, 
                        torch.full((max_mhc_len - len(feature_mhc),), 1, dtype=torch.long)
                    ]).unsqueeze(0)
                    for feature_mhc in x.get('feature_data', {}).get(feature_name, {}).get('mhc', 
                                torch.full((1, 1), 1, dtype=torch.long))
                ] + [
                    torch.full((1, max_mhc_len), 1, dtype=torch.long)
                ] * (max_num_alleles - len(x.get('feature_data', {}).get(feature_name, {}).get('mhc', 
                            torch.full((1, 1), 1, dtype=torch.long)))), dim=0)
                for x in batch
            ])
            feature_data[feature_name]['mhc'] = feature_mhc_list
        
        result['feature_data'] = feature_data
    
    return result
