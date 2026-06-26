import numpy as np
import pandas as pd
import peptides
from sklearn.decomposition import PCA
from pathlib import Path
from importlib.resources import files

def extract_descriptor_columns(descriptor_dict, selected_keys):
    """
    Given a descriptor dictionary and a list of descriptor group keys,
    return a flat list of all associated column names.

    Args:
        descriptor_dict (dict): Dictionary mapping descriptor group names to column names.
        selected_keys (list): List of descriptor group keys to extract.

    Returns:
        list: Aggregated list of column names across selected groups.
    """
    columns = []
    for key in selected_keys:
        if key in descriptor_dict:
            columns.extend(descriptor_dict[key])
        else:
            raise ValueError(f"Descriptor group '{key}' not found in descriptor dictionary.")
    return columns


def load_feature_data():
    amino_acids=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    # peptides library descriptor groups
    # https://github.com/althonos/peptides.py

    # aggregated aa features: AF, F, PP, E, Z, VHSE, KF	
    # sequence specific features: ST, SV, SVGER, VSTPV, MSWHIM	
    # mixed: BLOSUM, PRIN, PD, ProtFP	

    descriptor_groups = {
        "Atchley": ['AF1', 'AF2', 'AF3', 'AF4', 'AF5'],
        "BLOSUM": ['BLOSUM1', 'BLOSUM2', 'BLOSUM3', 'BLOSUM4', 'BLOSUM5',
                'BLOSUM6', 'BLOSUM7', 'BLOSUM8', 'BLOSUM9', 'BLOSUM10'],
        "Polarity/Physicochemical (PP)": ['PP1', 'PP2', 'PP3'],
        "F-descriptors": ['F1', 'F2', 'F3', 'F4', 'F5', 'F6'],
        "Kidera Factors": ['KF1', 'KF2', 'KF3', 'KF4', 'KF5', 'KF6', 'KF7', 'KF8', 'KF9', 'KF10'],
        "MS-WHIM": ['MSWHIM1', 'MSWHIM2', 'MSWHIM3'],
        "E-descriptors": ['E1', 'E2', 'E3', 'E4', 'E5'],
        "PD": ['PD1', 'PD2'], # (Physicochemical Descriptors)
        "PRIN": ['PRIN1', 'PRIN2', 'PRIN3'],
        "ProtFP": ['ProtFP1', 'ProtFP2', 'ProtFP3', 'ProtFP4',
                'ProtFP5', 'ProtFP6', 'ProtFP7', 'ProtFP8'],
        "ST (Statistical)": ['ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'ST6', 'ST7', 'ST8'],
        "SV (Sequence Vector)": ['SV1', 'SV2', 'SV3', 'SV4'],
        "SVGER": ['SVGER1', 'SVGER2', 'SVGER3', 'SVGER4', 'SVGER5', 'SVGER6',
                'SVGER7', 'SVGER8', 'SVGER9', 'SVGER10', 'SVGER11'],
        "T-descriptors": ['T1', 'T2', 'T3', 'T4', 'T5'],
        "VHSE": ['VHSE1', 'VHSE2', 'VHSE3', 'VHSE4', 'VHSE5', 'VHSE6', 'VHSE7', 'VHSE8'],
        "VSTPV": ['VSTPV1', 'VSTPV2', 'VSTPV3', 'VSTPV4', 'VSTPV5', 'VSTPV6'],
        "Z-scale": ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']
    }

    aggregated_keys = [
        "Atchley",
        "BLOSUM",
        "Polarity/Physicochemical (PP)",
        "F-descriptors",
        "Kidera Factors",
        "E-descriptors",
        "PD",
        "PRIN",
        "ProtFP",
        "VHSE",
        "Z-scale"
    ]

    sequence_specific_keys = [
        "MS-WHIM",
        "ST (Statistical)",
        "SV (Sequence Vector)",
        "SVGER",
        "T-descriptors",
        "VSTPV"
    ]


    aa_agg_features=extract_descriptor_columns(descriptor_groups, aggregated_keys)

    # peptide amino acid descriptors
    amino_acid_descriptors = pd.DataFrame([peptides.Peptide(s).descriptors() for s in amino_acids], index=amino_acids) # index=data_split_train_feature.index
    amino_acid_descriptors=amino_acid_descriptors[aa_agg_features]

    amino_acid_descriptors.loc["X"]=0
    amino_acid_descriptors.loc["U"]=0

    # atchley factors
    # https://github.com/vadimnazarov/kidera-atchley

    atchley_str = "A,-0.591,-1.302,-0.733,1.570,-0.146;C,-1.343,0.465,-0.862,-1.020,-0.255;D,1.050,0.302,-3.656,-0.259,-3.242;E,1.357,-1.453,1.477,0.113,-0.837;F,-1.006,-0.590,1.891,-0.397,0.412;G,-0.384,1.652,1.330,1.045,2.064;H,0.336,-0.417,-1.673,-1.474,-0.078;I,-1.239,-0.547,2.131,0.393,0.816;K,1.831,-0.561,0.533,-0.277,1.648;L,-1.019,-0.987,-1.505,1.266,-0.912;M,-0.663,-1.524,2.219,-1.005,1.212;N,0.945,0.828,1.299,-0.169,0.933;P,0.189,2.081,-1.628,0.421,-1.392;Q,0.931,-0.179,-3.005,-0.503,-1.853;R,1.538,-0.055,1.502,0.440,2.897;S,-0.228,1.399,-4.760,0.670,-2.647;T,-0.032,0.326,2.213,0.908,1.313;V,-1.337,-0.279,-0.544,1.242,-1.262;W,-0.595,0.009,0.672,-2.128,-0.184;Y,0.260,0.830,3.097,-0.838,1.512"

    rows = atchley_str.split(";")
    data = [r.split(",") for r in rows]
    atchley_df = pd.DataFrame(data, columns=["amino_acid", "f1", "f2", "f3", "f4", "f5"])
    for col in ["f1", "f2", "f3", "f4", "f5"]:
        atchley_df[col] = atchley_df[col].astype(float)
    atchley_df.set_index("amino_acid", inplace=True)

    # rename columns
    atchley_df.columns=["a1", "a2", "a3", "a4", "a5"]

    # add U and X to atchley factors table, fix later.
    atchley_df.loc["X"]=0
    atchley_df.loc["U"]=0

    # feature information
    # https://github.com/vadimnazarov/kidera-atchley
    aa_prop = pd.DataFrame(map(lambda x: x.split(","), "A,1.29,0.9,0,0.049,1.8,0,0,0.047,0.065,0.78,67,1,0,0,1,1;C,1.11,0.74,0,0.02,2.5,-2,0,0.015,0.015,0.8,86,1,1,-1,0,1;D,1.04,0.72,-1,0.051,-3.5,-2,1,0.071,0.074,1.41,91,1,0,1,1;E,1.44,0.75,-1,0.051,-3.5,-2,1,0.094,0.089,1,109,1,0,1,0,1;F,1.07,1.32,0,0.051,2.8,0,0,0.021,0.029,0.58,135,1,1,-1,0,1;G,0.56,0.92,0,0.06,-0.4,0,0,0.071,0.07,1.64,48,1,0,1,1,1;H,1.22,1.08,0,0.034,-3.2,1,1,0.022,0.025,0.69,118,1,0,-1,0,1;I,0.97,1.45,0,0.047,4.5,0,0,0.032,0.035,0.51,124,1,1,-1,0,1;K,1.23,0.77,1,0.05,-3.9,2,1,0.105,0.08,0.96,135,1,0,1,0,1;L,1.3,1.02,0,0.078,3.8,0,0,0.052,0.063,0.59,124,1,1,-1,1,1;M,1.47,0.97,0,0.027,1.9,0,0,0.017,0.016,0.39,124,1,1,1,0,1;N,0.9,0.76,0,0.058,-3.5,0,1,0.062,0.053,1.28,96,1,0,1,1,1;P,0.52,0.64,0,0.051,-1.6,0,0,0.052,0.054,1.91,90,1,0,1,0,1;Q,1.27,0.8,0,0.051,-3.5,1,1,0.053,0.051,0.97,114,1,0,1,0,1;R,0.96,0.99,1,0.066,-4.5,2,1,0.068,0.059,0.88,148,1,0,1,1,1;S,0.82,0.95,0,0.057,-0.8,-1,1,0.072,0.071,1.33,73,1,0,1,1,1;T,0.82,1.21,0,0.064,-0.7,-1,0,0.064,0.065,1.03,93,1,0,0,1,1;V,0.91,1.49,0,0.049,4.2,0,0,0.048,0.048,0.47,105,1,1,-1,0,1;W,0.99,1.14,0,0.022,-0.9,1,1,0.007,0.012,0.75,163,1,1,-1,0,1;Y,0.72,1.25,0,0.07,-1.3,-1,1,0.032,0.033,1.05,141,1,1,-1,1,1".split(";")), columns=['aminoacid', 'alpha', 'beta', 'charge', 'core', 'hydropathy', 'pH', 'polarity', 'rim', 'surface', 'turn', 'volume', 'count', 'strength', 'disorder', 'high_contact', 'count'], index=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])

    kidera = pd.DataFrame.from_records(list(map(lambda x: list(map(float, x.split(','))), "-1.56,-1.67,-0.97,-0.27,-0.93,-0.78,-0.2,-0.08,0.21,-0.48;0.22,1.27,1.37,1.87,-1.7,0.46,0.92,-0.39,0.23,0.93;1.14,-0.07,-0.12,0.81,0.18,0.37,-0.09,1.23,1.1,-1.73;0.58,-0.22,-1.58,0.81,-0.92,0.15,-1.52,0.47,0.76,0.7;0.12,-0.89,0.45,-1.05,-0.71,2.41,1.52,-0.69,1.13,1.1;-0.47,0.24,0.07,1.1,1.1,0.59,0.84,-0.71,-0.03,-2.33;-1.45,0.19,-1.61,1.17,-1.31,0.4,0.04,0.38,-0.35,-0.12;1.46,-1.96,-0.23,-0.16,0.1,-0.11,1.32,2.36,-1.66,0.46;-0.41,0.52,-0.28,0.28,1.61,1.01,-1.85,0.47,1.13,1.63;-0.73,-0.16,1.79,-0.77,-0.54,0.03,-0.83,0.51,0.66,-1.78;-1.04,0,-0.24,-1.1,-0.55,-2.05,0.96,-0.76,0.45,0.93;-0.34,0.82,-0.23,1.7,1.54,-1.62,1.15,-0.08,-0.48,0.6;-1.4,0.18,-0.42,-0.73,2,1.52,0.26,0.11,-1.27,0.27;-0.21,0.98,-0.36,-1.43,0.22,-0.81,0.67,1.1,1.71,-0.44;2.06,-0.33,-1.15,-0.75,0.88,-0.45,0.3,-2.3,0.74,-0.28;0.81,-1.08,0.16,0.42,-0.21,-0.43,-1.89,-1.15,-0.97,-0.23;0.26,-0.7,1.21,0.63,-0.1,0.21,0.24,-1.15,-0.56,0.19;0.3,2.1,-0.72,-1.57,-1.16,0.57,-0.48,-0.4,-2.3,-0.6;1.38,1.48,0.8,-0.56,0,-0.68,-0.31,1.03,-0.05,0.53;-0.74,-0.71,2.04,-0.4,0.5,-0.81,-1.07,0.06,-0.46,0.65".split(";"))), index=["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"], columns=list(map(lambda x: "f"+str(x), range(1,11))))


    aa_property_path = files("mhcprime.data").joinpath("aa_property_table.txt")

    # from: https://github.com/mikessh/vdjtools/blob/master/src/main/resources/profile/aa_property_table.txt
    aa_property_table = pd.read_csv(aa_property_path, sep="\t", header=1)
    # aa_property_table = pd.read_csv(aa_property_path, sep="\t", header=1)
    # aa_property_table=pd.read_csv("/Data/aa_property_table.txt", sep="\t", header=1)
    # aa_property_table.set_index("amino_acid", inplace=True)

    # add U and X to kidera factors table, fix later.
    kidera.loc["X"]=0
    kidera.loc["U"]=0

    # Volume (Å³)
    volume = {
        'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
        'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
        'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
        'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6, 'X':0, 'U': 168.05
        # missing X - we'll handle this below
    }
        
    # isoelectric point (pI)
    isoelectric_point = {
        'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48,
        'G': 5.97, 'H': 7.59, 'I': 6.02, 'K': 9.74, 'L': 5.98,
        'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65, 'R': 10.76,
        'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.66, 'X':0, 'U':0.5
        # missing U and X
    }

    # polarity (Grantham scale)
    polarity = {
        'A': 8.1, 'C': 5.5, 'D': 13.0, 'E': 12.3, 'F': 5.2,
        'G': 9.0, 'H': 10.4, 'I': 5.2, 'K': 11.3, 'L': 4.9,
        'M': 5.7, 'N': 11.6, 'P': 8.0, 'Q': 10.5, 'R': 10.5,
        'S': 9.2, 'T': 8.6, 'V': 5.9, 'W': 5.4, 'Y': 6.2, 'X':0, 'U': 10 
        # fix value for U
        # missing U and X
    }

    # Kyte-Doolittle hydrophobicity (hydropathy)
    hydrophobicity = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
            'X': 0.0, 'U': -0.5  # Estimated values for unknown (X) and selenocysteine (U)
        }

    alpha=dict(zip(aa_prop.aminoacid, aa_prop.alpha.astype(float)))
    alpha['X']=0
    alpha['U']=float(alpha['C'])

    beta=dict(zip(aa_prop.aminoacid, aa_prop.beta.astype(float)))
    beta['X']=0
    beta['U']=float(beta['C'])

    surface=dict(zip(aa_prop.aminoacid, aa_prop.surface.astype(float)))
    surface['X']=0
    surface['U']=float(surface['C'])

    turn=dict(zip(aa_prop.aminoacid, aa_prop.turn.astype(float)))
    turn['X']=0
    turn['U']=float(turn['C'])

    mjenergy=dict(zip(aa_property_table.amino_acid, aa_property_table.mjenergy.astype(float)))
    mjenergy['X']=0
    mjenergy['U']=float(mjenergy['C'])

    aa_features_table={"volume":volume, "isoelectric_point":isoelectric_point, "polarity":polarity, "hydrophobicity":hydrophobicity, "alpha":alpha, "beta":beta, "surface":surface, "turn":turn, "mjenergy":mjenergy} # remove mjenergy

    aa_features_table=pd.DataFrame.from_dict(aa_features_table)
    aa_features_table.index_name="aminoacid"

    # add peptide aa descriptors
    aa_features_table_ext=pd.concat([aa_features_table, amino_acid_descriptors], axis=1)

    # add mj energy
    feature_boundaries = {
        'volume': [0, 70, 110, 150, 170, 200, 250],
        'isoelectric_point': [0, 3, 5, 6, 7, 9, 11],
        'polarity': [0, 5.3, 5.6, 7, 8.7, 9.9, 11, 12, 14],
        'hydrophobicity': [-5, -3.5, -2, 0, 2, 4.5],
        'alpha': [0.0, 0.52, 0.82, 0.97, 1.11, 1.47], 
        'beta': [0.0, 0.64, 0.76, 0.92, 1.02, 1.49], 
        'surface': [0.0, 0.015, 0.035, 0.053, 0.065, 0.089], 
        'turn': [0.0, 0.51, 0.75, 0.97, 1.03, 1.91]
    }

    return aa_features_table_ext, descriptor_groups

primitive_feats = [
    "volume",
    "isoelectric_point",
    "polarity",
    "hydrophobicity",
    "alpha",
    "beta",
    "surface",
    "turn",
    "mjenergy",
]

pca_groups = {
    "AF":     ["AF1", "AF2", "AF3", "AF4", "AF5"],
    "BLOSUM": [f"BLOSUM{i}" for i in range(1, 11)],
    "PP":     ["PP1", "PP2", "PP3"],
    "F":      [f"F{i}" for i in range(1, 7)],
    "KF":     [f"KF{i}" for i in range(1, 11)],
    "E":      [f"E{i}" for i in range(1, 6)],
    "PD":     ["PD1", "PD2"],
    "PRIN":   ["PRIN1", "PRIN2", "PRIN3"],
    "ProtFP": [f"ProtFP{i}" for i in range(1, 9)],
    "VHSE":   [f"VHSE{i}" for i in range(1, 9)],
    "Z":      ["Z1", "Z2", "Z3", "Z4", "Z5"],
}

def process_aa_feature_table(feature_df: pd.DataFrame):
    """
    Process AA feature table for use in FeatureProjectionEncoder.

    Args
    ----
    feature_df : pd.DataFrame
        Index: amino acid tokens (e.g. 20 AAs + X/U)
        Columns: raw features (volume, AF1, ..., Z5)

    Returns
    -------
    processed_df : pd.DataFrame
        Index: same as input
        Columns: primitive features (z-scored) + PCA components (z-scored).
    corr_summary : dict
        Simple correlation stats per group (for inspection only).
    """

    df = feature_df.copy().astype(float)

    # define primitive and group features
    primitive_feats = [
        "volume",
        "isoelectric_point",
        "polarity",
        "hydrophobicity",
        "alpha",
        "beta",
        "surface",
        "turn",
        "mjenergy",
    ]

    pca_groups = {
        "AF":     ["AF1", "AF2", "AF3", "AF4", "AF5"],
        "BLOSUM": [f"BLOSUM{i}" for i in range(1, 11)],
        "PP":     ["PP1", "PP2", "PP3"],
        "F":      [f"F{i}" for i in range(1, 7)],
        "KF":     [f"KF{i}" for i in range(1, 11)],
        "E":      [f"E{i}" for i in range(1, 6)],
        "PD":     ["PD1", "PD2"],
        "PRIN":   ["PRIN1", "PRIN2", "PRIN3"],
        "ProtFP": [f"ProtFP{i}" for i in range(1, 9)],
        "VHSE":   [f"VHSE{i}" for i in range(1, 9)],
        "Z":      ["Z1", "Z2", "Z3", "Z4", "Z5"],
    }

    # check columns
    all_expected = set(primitive_feats)
    for cols in pca_groups.values():
        all_expected.update(cols)

    missing = all_expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected feature columns: {sorted(missing)}")

    # z-score all raw features across AAs
    df_z = df.copy()
    for col in all_expected:
        col_vals = df_z[col].values.astype(float)
        mean = col_vals.mean()
        std = col_vals.std()
        if std < 1e-8:
            std = 1.0 
        df_z[col] = (col_vals - mean) / std

    # corr summaries
    corr_summary = {}

    # global correlation between all standardized features
    corr_mat = df_z[list(all_expected)].corr()
    abs_corr = corr_mat.abs()
    np.fill_diagonal(abs_corr.values, 0.0)
    corr_summary["global_max_abs_corr"] = abs_corr.to_numpy().max()

    # grouped correlation summary
    group_corr_stats = {}
    for group_name, cols in pca_groups.items():
        sub_corr = df_z[cols].corr()
        sub_abs = sub_corr.abs()
        np.fill_diagonal(sub_abs.values, 0.0)
        group_corr_stats[group_name] = {
            "max_abs_corr": float(sub_abs.to_numpy().max()),
            "mean_abs_corr": float(
                sub_abs.to_numpy()[np.triu_indices_from(sub_abs, k=1)].mean()
                if len(cols) > 1 else 0.0
            ),
        }

    corr_summary["group_corr"] = group_corr_stats

    # processed feature table
    processed = pd.DataFrame(index=df.index)

    # add primitives
    for col in primitive_feats:
        processed[col] = df_z[col]

    # pca per group
    for group_name, cols in pca_groups.items():
        X = df_z[cols].values  # (n_AA, n_cols)
        n_cols = X.shape[1]
        n_comp = min(5, n_cols)  # cannot exceed number of columns

        pca = PCA(n_components=n_comp)
        pcs = pca.fit_transform(X)  # (n_AA, n_comp)

        # add new pcs
        for i in range(n_comp):
            processed[f"{group_name}_PC{i+1}"] = pcs[:, i]

    # final z-score over all resulting features
    processed_z = processed.copy()
    for col in processed_z.columns:
        vals = processed_z[col].values.astype(float)
        mean = vals.mean()
        std = vals.std()
        if std < 1e-8:
            std = 1.0
        processed_z[col] = (vals - mean) / std

    return processed_z, corr_summary



def load_processed_feature_table():
    path = files("mhcprime.data").joinpath("processed_feature_table.csv")
    return pd.read_csv(path, index_col=0)


def get_feature_names(feature_table=None):
    if feature_table is None:
        feature_table = load_processed_feature_table()
    return list(feature_table.columns)