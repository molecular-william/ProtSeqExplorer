import itertools
import math
from collections import Counter

class EnergyEntropy_1:
    def __init__(self, data_type="DNA", energy_values=2, mutual_information_energy=2):
        """
        data_type (str): "protein", "DNA" or "RNA"
        energy_values (int): 1,2,3,...   (E1, E2, E3 are computed for k = 1 .. energy_values-1)
        mutual_information_energy (int): 2,3,...  (size of combinations for E4)
        """
        self.data_type = data_type
        self.energy_values = energy_values
        self.mutual_information_energy = mutual_information_energy

        # Alphabet for the given data type
        self.alphabet = self._get_data_type(data_type)

        # Precompute all k‑combinations for E1, E2, E3 (k = 1 .. energy_values-1)
        self.combinations_by_k = {}
        if self.energy_values > 1:
            for k in range(1, self.energy_values):
                self.combinations_by_k[k] = list(itertools.combinations(self.alphabet, k))

        # Precompute combinations and a fast lookup set for mutual information (E4)
        if self.mutual_information_energy >= 1:
            self.mi_combinations = list(itertools.combinations(
                self.alphabet, self.mutual_information_energy))
            self.mi_combinations_set = {"".join(sorted(c)) for c in self.mi_combinations}
        else:
            self.mi_combinations = []
            self.mi_combinations_set = set()

    def _get_data_type(self, data_type):
        """Return the alphabet string for the given data type."""
        data_type_dict = {
            "dna": 'ACGT',
            "protein": 'ACDEFGHIKLMNPQRSTVWY',
            "rna": 'ACGU',
        }
        return data_type_dict.get(data_type.lower())

    def _compute_kcomb_features(self, entropy_dict, count_dict, k):
        """
        For all k‑combinations of the alphabet, compute:
            (sum of entropy_dict[letter]) * (sum of count_dict[letter])
        and return a list of these values in the order of itertools.combinations.
        """
        result = []
        for comb in self.combinations_by_k[k]:
            sum_entropy = 0.0
            sum_count = 0
            for letter in comb:
                sum_entropy += entropy_dict[letter]
                sum_count += count_dict[letter]
            result.append(sum_entropy * sum_count)
        return result

    def seq2vector(self, seq):
        seq_len = len(seq)
        if seq_len < 2:
            raise ValueError("Sequence length must be at least 2")

        # ---------- Single pass: count letters, positions, pairs, MI windows ----------
        letter_counts = Counter()          # counts of each letter
        position_sum = {c: 0 for c in self.alphabet}   # sum of 1‑based positions
        pair_counts = Counter()           # counts of adjacent pairs
        mi_counts = Counter()            # counts of valid sorted MI windows

        for i, ch in enumerate(seq, start=1):
            letter_counts[ch] += 1
            if ch in position_sum:
                position_sum[ch] += i

        # Adjacent pairs (length 2)
        for i in range(seq_len - 1):
            pair = seq[i:i+2]
            pair_counts[pair] += 1

        # Sliding windows for mutual information (length r)
        r = self.mutual_information_energy
        if r > 1 and r <= seq_len:
            for i in range(seq_len - r + 1):
                window = seq[i:i+r]
                sorted_win = "".join(sorted(window))
                if sorted_win in self.mi_combinations_set:
                    mi_counts[sorted_win] += 1

        # ---------- Precomputed values that are reused ----------
        total_pairs = seq_len - 1
        all_position = seq_len * (seq_len + 1) * 0.5   # sum of positions 1..seq_len

        # Counts for all alphabet letters (including zero counts)
        number_X = {c: letter_counts.get(c, 0) for c in self.alphabet}
        p_X = {c: number_X[c] / seq_len for c in self.alphabet}

        # ---------- E1 : entropy of single letters ----------
        H_X = {c: -v * math.log2(v) if v > 0 else 0.0 for c, v in p_X.items()}
        E1 = []
        if self.energy_values > 1:
            for k in range(1, self.energy_values):
                E1.extend(self._compute_kcomb_features(H_X, number_X, k))

        # ---------- E2 : conditional entropy based on pairs ----------
        # For each second letter, compute the entropy of the conditional distribution
        H_second = {c: 0.0 for c in self.alphabet}
        if total_pairs > 0:
            for pair, cnt in pair_counts.items():
                if cnt == 0:
                    continue
                b = pair[1]          # second letter
                prob = cnt / total_pairs
                H_second[b] += -prob * math.log2(prob + 1e-10)

        E2 = []
        if self.energy_values > 1:
            for k in range(1, self.energy_values):
                E2.extend(self._compute_kcomb_features(H_second, number_X, k))

        # ---------- E3 : entropy of relative positions ----------
        H_rel = {}
        for c in self.alphabet:
            pos_sum = position_sum[c]
            relative = all_position - pos_sum
            rel_p = relative / all_position
            if rel_p > 0:
                H_rel[c] = -rel_p * math.log2(rel_p)
            else:
                H_rel[c] = 0.0

        E3 = []
        if self.energy_values > 1:
            for k in range(1, self.energy_values):
                E3.extend(self._compute_kcomb_features(H_rel, number_X, k))

        # ---------- E4 : mutual information for size‑r combinations ----------
        E4 = []
        if self.mi_combinations and seq_len >= r:
            # Product of marginal probabilities under independence
            p_comb = {}
            for comb in self.mi_combinations:
                prod = 1.0
                for letter in comb:
                    prod *= p_X[letter]
                p_comb[comb] = prod

            for comb in self.mi_combinations:
                key = "".join(sorted(comb))
                count = mi_counts.get(key, 0)
                if count > 0:
                    prob = count / seq_len
                    mi_val = math.log2(prob / p_comb[comb]) * prob * count
                    E4.append(mi_val)
                else:
                    E4.append(0.0)

        # ---------- Combine all features ----------
        return [float(v) for v in E1 + E2 + E3 + E4]
