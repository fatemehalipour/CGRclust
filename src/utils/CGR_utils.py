
"""Class CGR and FCGR are from complexCGR repo"""

from itertools import product
from typing import Dict, Optional, List
from collections import defaultdict, namedtuple
from PIL import Image
import numpy as np
import random

# coordinates for x+iy
Coord = namedtuple("Coord", ["x", "y"])

# coordinates for a CGR encoding
CGRCoords = namedtuple("CGRCoords", ["N", "x", "y"])

# coordinates for each nucleotide in the 2d-plane
DEFAULT_COORDS = dict(G=Coord(1, 1), C=Coord(-1, 1), A=Coord(-1, -1), T=Coord(1, -1))


class CGR:
    "Chaos Game Representation for DNA"

    def __init__(self, coords: Optional[Dict[chr, tuple]] = None):
        self.nucleotide_coords = DEFAULT_COORDS if coords is None else coords
        self.cgr_coords = CGRCoords(0, 0, 0)

    def nucleotide_by_coords(self, x, y):
        "Get nucleotide by coordinates (x,y)"
        # filter nucleotide by coordinates
        filtered = dict(filter(lambda item: item[1] == Coord(x, y), self.nucleotide_coords.items()))

        return list(filtered.keys())[0]

    def forward(self, nucleotide: str):
        "Compute next CGR coordinates"
        x = (self.cgr_coords.x + self.nucleotide_coords.get(nucleotide).x) / 2
        y = (self.cgr_coords.y + self.nucleotide_coords.get(nucleotide).y) / 2

        # update cgr_coords
        self.cgr_coords = CGRCoords(self.cgr_coords.N + 1, x, y)

    def backward(self, ):
        "Compute last CGR coordinates. Current nucleotide can be inferred from (x,y)"
        # get current nucleotide based on coordinates
        n_x, n_y = self.coords_current_nucleotide()
        nucleotide = self.nucleotide_by_coords(n_x, n_y)

        # update coordinates to the previous one
        x = 2 * self.cgr_coords.x - n_x
        y = 2 * self.cgr_coords.y - n_y

        # update cgr_coords
        self.cgr_coords = CGRCoords(self.cgr_coords.N - 1, x, y)

        return nucleotide

    def coords_current_nucleotide(self, ):
        x = 1 if self.cgr_coords.x > 0 else -1
        y = 1 if self.cgr_coords.y > 0 else -1
        return x, y

    def encode(self, sequence: str):
        "From DNA sequence to CGR"
        # reset starting position to (0,0,0)
        self.reset_coords()
        for nucleotide in sequence:
            self.forward(nucleotide)
        return self.cgr_coords

    def reset_coords(self, ):
        self.cgr_coords = CGRCoords(0, 0, 0)

    def decode(self, N: int, x: int, y: int) -> str:
        "From CGR to DNA sequence"
        self.cgr_coords = CGRCoords(N, x, y)

        # decoded sequence
        sequence = []

        # Recover the entire genome
        while self.cgr_coords.N > 0:
            nucleotide = self.backward()
            sequence.append(nucleotide)
        return "".join(sequence[::-1])


class FCGR(CGR):
    """Frequency matrix CGR
    an (2**k x 2**k) 2D representation will be created for a
    n-long sequence.
    - k represents the k-mer.
    - 2**k x 2**k = 4**k the total number of k-mers (sequences of length k)
    """

    def __init__(self, k: int, bits: int = 8):
        super().__init__()
        self.k = k  # k-mer representation
        self.kmers = list("".join(kmer) for kmer in product("ACGT", repeat=self.k))
        self.kmer2pixel = self.kmer2pixel_position()
        self.freq_kmer = defaultdict(int)
        self.bits = bits
        self.max_color = 8 ** bits - 1

    def __call__(self, sequence: str):
        "Given a DNA sequence, returns an array with his FCGR"
        self.count_kmers(sequence)

        # Create an empty array to save the FCGR values
        array_size = int(2 ** self.k)
        fcgr = np.zeros((array_size, array_size))

        # Assign frequency to each box in the matrix
        for kmer, freq in self.freq_kmer.items():
            pos_x, pos_y = self.kmer2pixel[kmer]
            fcgr[int(pos_x) - 1, int(pos_y) - 1] = freq
        return fcgr

    def count_kmer(self, kmer):
        if "N" not in kmer:
            self.freq_kmer[kmer] += 1

    def count_kmers(self, sequence: str):
        self.freq_kmer = defaultdict(int)
        # representativity of kmers
        last_j = len(sequence) - self.k + 1
        kmers = (sequence[i:(i + self.k)] for i in range(last_j))
        # count kmers in a dictionary
        list(self.count_kmer(kmer) for kmer in kmers)

    def pixel_position(self, kmer: str):
        "Get pixel position in the FCGR matrix for a k-mer"

        coords = self.encode(kmer)
        N, x, y = coords.N, coords.x, coords.y

        # Coordinates from [-1,1]² to [1,2**k]²
        np_coords = np.array([(x + 1) / 2, (y + 1) / 2])  # move coordinates from [-1,1]² to [0,1]²
        np_coords *= 2 ** self.k  # rescale coordinates from [0,1]² to [0,2**k]²
        x, y = np.ceil(np_coords)  # round to upper integer

        # Turn coordinates (cx,cy) into pixel (px,py) position
        # px = 2**k-cy+1, py = cx
        return 2 ** self.k - int(y) + 1, int(x)

    def kmer2pixel_position(self, ):
        kmer2pixel = dict()
        for kmer in self.kmers:
            kmer2pixel[kmer] = self.pixel_position(kmer)
        return kmer2pixel

    def plot(self, fcgr):
        "Given a FCGR, plot it in grayscale"
        img_pil = self.array2img(fcgr)
        return img_pil

    def save_img(self, fcgr, path: str):
        "Save image in grayscale for the FCGR provided as input"
        img_pil = self.array2img(fcgr)
        img_pil.save(path)

    def array2img(self, array):
        "Array to PIL image"
        m, M = array.min(), array.max()
        # rescale to [0,1]
        img_rescaled = (array - m) / (M - m)

        # invert colors black->white
        img_array = np.ceil(self.max_color - img_rescaled * self.max_color)
        dtype = eval(f"np.int{self.bits}")
        img_array = np.array(img_array, dtype=dtype)

        # convert to Image
        img_pil = Image.fromarray(img_array, 'L')
        return img_pil


def generate_sequence(avoided_patterns: List[str],
                      seq_len: int = 200_000,
                      avoidance_probability: float = 0.9) -> str:
    """
    Generate a DNA sequence with a specified length while avoiding given patterns.

    Parameters:
    avoided_patterns (list(str)): The patterns to avoid in the generated sequence.
    seq_len (int): The length of the DNA sequence to generate. Default is 200,000.
    avoidance_probability (float): The probability of avoiding the specified pattern. Default is 0.9.

    Returns:
    str: The generated DNA sequence not containing the specified patterns.
    """
    # Initial random DNA sequence generation
    seq = "".join(random.choices("ACGT", k=seq_len))
    output_seq = seq

    # Iterate over each pattern to avoid in the sequence
    for avoided_pattern in avoided_patterns:
        k = len(avoided_pattern)
        while avoided_pattern in seq:
            start_index = seq.find(avoided_pattern)
            end_index = start_index + k
            kmer = "".join(random.choices("ACGT", k=k))  # Generate random k-mer
            seq = seq[:start_index] + kmer + seq[end_index:]  # Replace substring
            # Check if pattern should be avoided
            if random.random() < avoidance_probability:
                output_seq = output_seq[:start_index] + kmer + output_seq[end_index:]  # Replace substring

    return output_seq
