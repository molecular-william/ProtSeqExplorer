# From the paper: Classification of Protein Sequences by a Novel Alignment-Free Method on Bacterial and Virus Families (2022)
import numpy as np
from numba import jit

class AANaturalVector:
    """
    A class for analyzing amino acid sequences and converting them to natural vector representations.
    
    This class provides methods to convert amino acid sequences into feature vectors by:
    - Counting amino acid frequencies and positions
    - Computing cumulative distributions
    - Calculating statistical measures (theta, sigma, D, kesai)
    - Computing covariance matrices
    - Building final feature vectors
    """
    
    def __init__(self):
        """Initialize the analyzer with pre-computed amino acid mapping."""
        self.aa_mapping = self._create_aa_mapping()
        
    def _create_aa_mapping(self):
        """
        Create a fast mapping from character codes to amino acid indices.
        
        Returns:
            numpy.ndarray: Mapping array where ASCII character codes map to amino acid indices
        """
        aa_list = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
        mapping = np.full(128, -1, dtype=np.int32)  # ASCII range
        for idx, aa in enumerate(aa_list):
            mapping[ord(aa)] = idx
        return mapping

    @staticmethod
    @jit(nopython=True)
    def _count_amino_acids_numba(seq, aa_mapping, seq_len):
        """
        Count amino acids and record positions using pre-computed mapping.
        
        Args:
            seq: Amino acid sequence string
            aa_mapping: Pre-computed character-to-index mapping
            seq_len: Length of the sequence
            
        Returns:
            tuple: (miu, t, n) - position matrices and counts
        """
        miu = np.zeros((20, seq_len))
        t = np.zeros((20, seq_len), dtype=np.int32)
        n = np.zeros(20, dtype=np.int32)
        
        for i in range(seq_len):
            char_code = aa_mapping[ord(seq[i])]
            if char_code != -1:
                n[char_code] += 1
                t[char_code, n[char_code] - 1] = i + 1  # 1-based indexing
                miu[char_code, i] = 1
        
        return miu, t, n

    @staticmethod
    @jit(nopython=True)
    def _compute_cumulative_counts_numba(miu, t, n, seq_len):
        """
        Compute cumulative counts efficiently.
        
        Args:
            miu: Position indicator matrix
            t: Position tracking matrix
            n: Amino acid counts
            seq_len: Sequence length
            
        Returns:
            numpy.ndarray: Updated miu matrix with cumulative counts
        """
        for i in range(20):
            if n[i] > 0:
                positions = t[i, :n[i]] - 1  # Convert to 0-based indexing
                
                if positions[0] > 0:
                    miu[i, :positions[0]] = 0
                
                for j in range(1, n[i]):
                    start = positions[j-1]
                    end = positions[j]
                    miu[i, start:end] = j
                
                miu[i, positions[-1]:] = n[i]
        
        return miu

    @staticmethod
    @jit(nopython=True)
    def _compute_statistics_numba(miu, n, seq_len):
        """
        Compute theta, sigma, D, kesai efficiently.
        
        Args:
            miu: Cumulative count matrix
            n: Amino acid counts
            seq_len: Sequence length
            
        Returns:
            tuple: (theta, sigma, D, kesai) - statistical measures
        """
        theta = np.sum(miu, axis=1) / seq_len
        sigma = np.sum(miu, axis=1)
        
        D = np.zeros(20)
        for i in range(20):
            if n[i] > 0:
                # Vectorized calculation of D
                diff = miu[i] - theta[i]
                D[i] = np.sum(diff * diff) / (n[i] * n[i])
        
        kesai = np.zeros(20)
        for i in range(20):
            if n[i] > 0:
                kesai[i] = sigma[i] / n[i]
        
        return theta, sigma, D, kesai

    @staticmethod
    @jit(nopython=True)
    def _compute_covariance_numba(miu, theta, n, seq_len):
        """
        Compute covariance matrix efficiently.
        
        Args:
            miu: Cumulative count matrix
            theta: Mean values for each amino acid
            n: Amino acid counts
            seq_len: Sequence length
            
        Returns:
            numpy.ndarray: 20x20 covariance matrix
        """
        cov = np.zeros((20, 20))
        
        for i in range(20):
            if n[i] > 0:
                miu_i_centered = miu[i] - theta[i]
                denom_i = n[i]
                
                for j in range(i):  # Only compute lower triangle (i > j)
                    if n[j] > 0:
                        miu_j_centered = miu[j] - theta[j]
                        cov_ij = np.sum(miu_i_centered * miu_j_centered) / (denom_i * n[j])
                        cov[i, j] = cov_ij
        
        return cov

    @staticmethod
    @jit(nopython=True)
    def _build_feature_vector_numba(n, kesai, D, cov, feature1, m):
        """
        Build feature vector efficiently.
        
        Args:
            n: Amino acid counts
            kesai: Sigma/n ratios
            D: D statistics
            cov: Covariance matrix
            feature1: Feature vector array
            m: Index for current sequence
            
        Returns:
            numpy.ndarray: Updated feature vector
        """
        # Direct assignment to avoid function calls
        for i in range(20):
            feature1[m, i] = n[i]
        
        for i in range(20):
            feature1[m, 20 + i] = kesai[i]
        
        for i in range(20):
            feature1[m, 40 + i] = D[i]
        
        mm = 0
        for i in range(20):
            for j in range(i):  # Only lower triangle
                feature1[m, 60 + mm] = cov[i, j]
                mm += 1
        
        return feature1

    def seq2vector(self, seq):
        """
        Convert amino acid sequence to accumulated natural vector.
        
        Args:
            seq (str): Amino acid sequence string
            
        Returns:
            numpy.ndarray: Feature vector representation of the sequence
            
        Raises:
            ValueError: If sequence contains invalid amino acid characters
        """
        # Validate sequence
        if not isinstance(seq, str):
            raise ValueError("Sequence must be a string")
        
        N = len(seq)
        if N == 0:
            raise ValueError("Sequence cannot be empty")
        
        # Check for valid amino acid characters
        invalid_chars = []
        for char in seq:
            if ord(char) >= 128 or self.aa_mapping[ord(char)] == -1:
                invalid_chars.append(char)
        
        if invalid_chars:
            unique_invalid = list(set(invalid_chars))
            raise ValueError(f"Sequence contains invalid amino acid characters: {unique_invalid}")
        
        feature1 = np.zeros((1, 250)) 
        
        try:
            # Step 1: Count amino acids and positions
            miu, t, n = self._count_amino_acids_numba(seq, self.aa_mapping, N)
            
            # Step 2: Compute cumulative counts
            miu = self._compute_cumulative_counts_numba(miu, t, n, N)
            
            # Step 3: Compute statistics
            theta, sigma, D, kesai = self._compute_statistics_numba(miu, n, N)
            
            # Step 4: Compute covariance
            cov = self._compute_covariance_numba(miu, theta, n, N)
            
            # Step 5: Build feature vector
            feature1 = self._build_feature_vector_numba(n, kesai, D, cov, feature1, 0)
            
        except Exception as e:
            raise RuntimeError(f"Error processing sequence: {str(e)}")
        
        # Remove all-zero rows
        feature1 = feature1[~np.all(feature1 == 0, axis=1)]

        return feature1
