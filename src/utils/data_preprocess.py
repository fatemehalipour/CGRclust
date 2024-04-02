import os

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def walk_through_dir(dir_path):
    """walks though dir_path returning its content"""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} files in {dirpath}")



def preprocess_seq(seq: str, add_reverse_complement: bool = False) -> str:
    """
    Preprocess a DNA sequence by replacing non-canonical nucleotides with "N".
    Optionally, append the reverse complement of the processed sequence.

    Parameters:
    - seq (str): The DNA sequence to preprocess.
    - add_reverse_complement (bool): If True, append the reverse complement of the
                                         preprocessed sequence. Default is False.

    Returns:
    - str: The preprocessed (and possibly extended) DNA sequence.
    """
    # Replace non-canonical nucleotides with "N" using list comprehension
    processed_seq = [nuc if nuc in ["A", "C", "G", "T"] else "N" for nuc in seq]

    # Convert the list back to a string
    processed_seq_str = "".join(processed_seq)

    # Generate reverse complement if required
    if add_reverse_complement:
        reverse_complement = str(Seq(processed_seq_str).reverse_complement())
        processed_seq_str += reverse_complement

    return processed_seq_str


def make_big_fasta(df: pd.DataFrame,
                   filename: str):
    """
    Writes sequences from a DataFrame to a FASTA file.

    Parameters:
    - df (pd.DataFrame): DataFrame containing sequence data
    - filename (str): Path to the output fasta file.

    The function creates SeqRecord objects for each sequence in the DataFrame and writes them
    to the specified fasta file.
    """
    # Create SeqRecord objects using DataFrame iteration for efficiency
    seq_records = [SeqRecord(Seq(row["sequence"]), id=row["id"], description=row["label"]) for _, row in df.iterrows()]

    # Write to a FASTA file
    with open(filename, "w") as output_handle:
        SeqIO.write(seq_records, output_handle, "fasta")


def read_fasta(fasta_file: str):
    # Initialize the records dictionary
    records = {"id": [], "sequence": [], "label": []}
    with open(fasta_file, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            # Convert U to T (RNA -> DNA) and preprocess the sequence
            sequence = preprocess_seq(record.seq.replace("U", "T"))
            records["id"].append(record.id)
            records["sequence"].append(sequence)
            records["label"].append(record.description.split(" ")[1])

    # Convert the dictionary into a DataFrame
    records_df = pd.DataFrame(records)
    return records_df
