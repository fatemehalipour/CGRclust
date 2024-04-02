import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from utils import CGR_utils


def plot_random_fcgr(df, k=6, random_seed=42):
    """
    Plots a 3x3 grid of random FCGR images from sequences in a DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing sequences and their labels.
    - k (int): The k-mer size for FCGR calculation. Defaults to 6.
    - random_seed (int): Seed for the random number generator. Defaults to 42.
    """
    np.random.seed(random_seed)  # Using NumPy's random seed function for consistency
    fcgr = CGR_utils.FCGR(k=k)

    fig = plt.figure(figsize=(9, 9))
    rows, cols = 3, 3

    for i in range(1, rows * cols + 1):
        random_idx = np.random.randint(0, len(df))  # Ensuring index is within bounds
        seq, label = df.iloc[random_idx]["sequence"], df.iloc[random_idx]["label"]

        # Ensure sequence length is not zero to avoid division by zero
        if len(seq) == 0:
            continue

        # Generate and normalize chaos
        chaos = fcgr(seq)
        chaos_normalized = chaos / np.max(chaos)  # Normalizing by the maximum value

        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(1 - chaos_normalized, cmap="gray")  # Inverting colors for better visibility
        ax.set_title(label)
        ax.axis('off')  # Hide axis for better visualization

    plt.tight_layout()
    plt.show()


def mutation(seq: str,
             transition: bool,
             transversion: bool,
             transition_prob: float = 0.0,
             transversion_prob: float = 0.0) -> str:
    """
    Applies transition and transversion mutations to a DNA sequence.

    Args:
      seq (str): DNA sequence to mutate.
      transition (bool): Enables transition mutations if True.
      transversion (bool): Enables transversion mutations if True.
      transition_prob (float): Probability of each nucleotide undergoing a transition.
      transversion_prob (float): Probability of each nucleotide undergoing a transversion.

    Returns:
      str: The mutated DNA sequence.
    """
    # create a list of random numbers with length of the input sequence
    transition_indexes = []
    transversion_indexes = []

    # mutation dict based on transitions
    transition_mutations = {"A": "G",
                            "G": "A",
                            "C": "T",
                            "T": "C"}

    # mutation dict based on transversions
    transversion_mutations = {"A": ["T", "C"],
                              "G": ["T", "C"],
                              "C": ["A", "G"],
                              "T": ["A", "G"]}

    # transistion mutations
    if transition:
        random_list = np.random.random(len(seq))
        transition_indexes = np.where(random_list <= transition_prob)[0]

    # transversion mutations
    if transversion:
        random_list = np.random.random(len(seq))
        transversion_indexes = np.where(random_list <= transversion_prob)[0]

    # enumerating the input sequence and perform the mutations
    mutated_seq = []
    for i, nucleotide in enumerate(seq):
        if i in transition_indexes:
            try:
                mutated_seq.append(transition_mutations[nucleotide])
            except KeyError:
                pass
        elif i in transversion_indexes:
            try:
                mutated_seq.append(transversion_mutations[nucleotide][round(np.random.uniform())])
            except KeyError:
                pass
        else:
            mutated_seq.append(nucleotide)

    return "".join(mutated_seq)


def mutation_optimized(seq: str, transition: bool, transversion: bool, transition_prob: float = 0.0,
                       transversion_prob: float = 0.0) -> str:
    if not transition and not transversion:
        # If no mutations are enabled, return the original sequence
        return seq

    seq_array = np.array(list(seq))

    # Generate separate probability lists for transitions and transversions
    if transition:
        transition_probs = np.random.random(len(seq))
    if transversion:
        transversion_probs = np.random.random(len(seq))
    # Apply transition mutations
    if transition:
        for original, mutated in [("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")]:
            mask = (seq_array == original) & (transition_probs <= transition_prob)
            seq_array[mask] = mutated

    # Apply transversion mutations
    if transversion:
        for original, mutations in [("A", ["T", "C"]), ("G", ["T", "C"]), ("C", ["A", "G"]), ("T", ["A", "G"])]:
            transversion_mask = (seq_array == original) & (transversion_probs <= transversion_prob)
            for mutated in mutations:
                # Apply mutation to a subset of the masked indices
                mutation_application_mask = np.random.choice([True, False], size=transversion_mask.sum())
                applicable_indices = np.where(transversion_mask)[0][mutation_application_mask]
                seq_array[applicable_indices] = np.random.choice(mutations, size=applicable_indices.size)

    return "".join(seq_array)


def fragmentation(seq: str,
                  frag_len: int) -> str:
    """
    Extracts a random fragment of specified length from the given sequence.

    Parameters:
    - seq (str): The DNA sequence from which a fragment will be extracted.
    - frag_len (int): The length of the fragment to extract. If "frag_len" exceeds
      the length of "seq", the entire sequence is returned.

    Returns:
    - str: A substring of "seq" representing a randomly selected fragment.
    """
    # Ensure fragment length does not exceed sequence length
    if frag_len > len(seq):
        return seq

    start_index = np.random.randint(0, len(seq) - frag_len + 1)
    return seq[start_index: start_index + frag_len]


def augment_seq(seq, fcgr_instance, augmentation_type="None", mutation_rate=None, frag_perc=None):
    if augmentation_type == "mutation":
        mutated_seq = mutation_optimized(seq,
                                         transition=True,
                                         transversion=True,
                                         transition_prob=mutation_rate,
                                         transversion_prob=(0.5 * mutation_rate))
        fcgr_repr = fcgr_instance(mutated_seq)
    elif augmentation_type == "fragmentation":
        frag_len = int(frag_perc * len(seq))  # Adjusted to use `len(seq)` directly
        frag_seq = fragmentation(seq, frag_len)
        fcgr_repr = fcgr_instance(frag_seq)
    else:
        raise ValueError("Unsupported augmentation type specified.")

    return fcgr_repr


def generate_pairs(data: pd.DataFrame,
                   class_to_idx,
                   k: int = 6,
                   number_of_pairs: int = 1,
                   mutation_rate_weak=None,
                   mutation_rate_strong=None,
                   frag_perc_weak=None,
                   frag_perc_strong=None):
    fcgr = CGR_utils.FCGR(k=k)

    # Determine the total number of pairs to be generated
    total_pairs = data.shape[0] * number_of_pairs

    # Initialize NumPy arrays with the appropriate shape and dtype from the start
    X_train = np.zeros((total_pairs, 2),
                       dtype="object")
    X_test = np.zeros((data.shape[0],), dtype="object")
    y_test = np.zeros((data.shape[0],), dtype=int)

    pair_idx = 0
    for idx, record in tqdm(data.iterrows(), total=data.shape[0]):
        label = class_to_idx[record["label"]]
        original_seq = record["sequence"]
        original_seq_fcgr = fcgr(original_seq)

        for _ in range(number_of_pairs):
            if mutation_rate_weak is not None:
                weak_fcgr = augment_seq(seq=original_seq,
                                        fcgr_instance=fcgr,
                                        augmentation_type="mutation",
                                        mutation_rate=mutation_rate_weak)
                strong_fcgr = augment_seq(seq=original_seq,
                                          fcgr_instance=fcgr,
                                          augmentation_type="mutation",
                                          mutation_rate=mutation_rate_strong)
            else:
                weak_fcgr = augment_seq(seq=original_seq,
                                        fcgr_instance=fcgr,
                                        augmentation_type="fragmentation",
                                        frag_perc=frag_perc_weak)
                strong_fcgr = augment_seq(seq=original_seq,
                                          fcgr_instance=fcgr,
                                          augmentation_type="fragmentation",
                                          frag_perc=frag_perc_strong)
            X_train[pair_idx] = (weak_fcgr, strong_fcgr)
            pair_idx += 1

        X_test[idx] = original_seq_fcgr
        y_test[idx] = label

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape} | Number of labels in y_test: {len(y_test)}")

    return X_train, X_test, y_test
