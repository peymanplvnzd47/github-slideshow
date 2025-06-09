import numpy as np

# --- QPSK Specific ---
def coherent_detect(signal: np.ndarray) -> np.ndarray:
    """
    Placeholder for ideal coherent detection.
    Currently returns the input signal without modification.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Input signal must be a NumPy array.")
    return signal

def apply_equalizer(signal: np.ndarray) -> np.ndarray:
    """
    Placeholder for DSP/Equalizer.
    Currently returns the input signal without modification.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Input signal must be a NumPy array.")
    return signal

def qpsk_demodulate(received_symbols: np.ndarray) -> np.ndarray:
    """
    Demodulates QPSK symbols to bits using minimum Euclidean distance.
    """
    if not isinstance(received_symbols, np.ndarray) or received_symbols.ndim != 1:
        raise ValueError("Input 'received_symbols' must be a 1D NumPy array.")
    if not np.issubdtype(received_symbols.dtype, np.complexfloating):
        raise ValueError("Input 'received_symbols' must be of complex type.")

    norm_factor_qpsk = 1 / np.sqrt(2)
    qpsk_constellation = {
        (0, 0): (1 + 1j) * norm_factor_qpsk,
        (0, 1): (-1 + 1j) * norm_factor_qpsk,
        (1, 1): (-1 - 1j) * norm_factor_qpsk,
        (1, 0): (1 - 1j) * norm_factor_qpsk,
    }

    ideal_points = np.array(list(qpsk_constellation.values()))
    bit_pairs = np.array(list(qpsk_constellation.keys()))

    demodulated_bits = []
    for symbol in received_symbols:
        distances_sq = np.abs(symbol - ideal_points)**2
        closest_idx = np.argmin(distances_sq)
        demodulated_bits.extend(bit_pairs[closest_idx])

    return np.array(demodulated_bits, dtype=int)

# --- 16-QAM Specific ---

# Gray mapping for 2 bits to {-3, -1, 1, 3} - Identical to transmitter.py
# MSB first for mapping: e.g., b1 b0
_GRAY_MAP_2_BIT_TO_LEVEL_QAM16 = {
    (0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3
}
_LEVEL_TO_GRAY_MAP_2_BIT_QAM16 = {v: k for k, v in _GRAY_MAP_2_BIT_TO_LEVEL_QAM16.items()}

# Normalization factor for 16-QAM - Identical to transmitter.py
_QAM16_NORMALIZATION_FACTOR = 1 / np.sqrt(10)

# Pre-generate ideal 16-QAM constellation and bit mapping for demodulator
_IDEAL_16QAM_CONSTELLATION_POINTS = []
_IDEAL_16QAM_BIT_SEQUENCES = []
# Iterate through all possible 4-bit combinations (b3 b2 b1 b0)
for b3 in [0, 1]:
    for b2 in [0, 1]:
        val_i = _GRAY_MAP_2_BIT_TO_LEVEL_QAM16[(b3, b2)]
        for b1 in [0, 1]:
            for b0 in [0, 1]:
                val_q = _GRAY_MAP_2_BIT_TO_LEVEL_QAM16[(b1, b0)]
                symbol = (val_i + 1j * val_q) * _QAM16_NORMALIZATION_FACTOR
                _IDEAL_16QAM_CONSTELLATION_POINTS.append(symbol)
                _IDEAL_16QAM_BIT_SEQUENCES.append([b3, b2, b1, b0])

_IDEAL_16QAM_POINTS_NP = np.array(_IDEAL_16QAM_CONSTELLATION_POINTS, dtype=np.complex128)
_IDEAL_16QAM_BITS_NP = np.array(_IDEAL_16QAM_BIT_SEQUENCES, dtype=int)


def qam16_demodulate(received_symbols: np.ndarray) -> np.ndarray:
    """
    Demodulates 16-QAM symbols to bits using minimum Euclidean distance.
    Assumes symbols are normalized as per qam16_modulate.

    Args:
        received_symbols: A NumPy array of complex 16-QAM symbols.

    Returns:
        A NumPy array of demodulated bits (0s and 1s).
    """
    if not isinstance(received_symbols, np.ndarray) or received_symbols.ndim != 1:
        raise ValueError("Input 'received_symbols' must be a 1D NumPy array.")
    if received_symbols.size == 0: # Handle empty array case
        return np.array([], dtype=int)
    if not np.issubdtype(received_symbols.dtype, np.complexfloating):
        raise ValueError("Input 'received_symbols' must be of complex type.")

    demodulated_bits_list = []
    for symbol in received_symbols:
        # Calculate squared Euclidean distances to each ideal 16-QAM point
        distances_sq = np.abs(symbol - _IDEAL_16QAM_POINTS_NP)**2

        # Find the index of the closest ideal point
        closest_idx = np.argmin(distances_sq)

        # Append the corresponding 4-bit sequence
        demodulated_bits_list.extend(_IDEAL_16QAM_BITS_NP[closest_idx])

    return np.array(demodulated_bits_list, dtype=int)


if __name__ == '__main__':
    print("--- QPSK Demodulation Test (from previous setup) ---")
    # (Existing QPSK test code can remain or be shortened)
    qpsk_norm = 1 / np.sqrt(2)
    qpsk_orig_bits = np.array([0,0, 0,1, 1,1, 1,0])
    qpsk_ideal_syms = np.array([(1+1j)*qpsk_norm, (-1+1j)*qpsk_norm, (-1-1j)*qpsk_norm, (1-1j)*qpsk_norm])
    qpsk_noise = (np.random.normal(0, 0.1, qpsk_ideal_syms.shape) +
                  1j * np.random.normal(0, 0.1, qpsk_ideal_syms.shape))
    qpsk_noisy_syms = qpsk_ideal_syms + qpsk_noise
    qpsk_demod_bits = qpsk_demodulate(qpsk_noisy_syms)
    print(f"QPSK Original bits: {qpsk_orig_bits}")
    print(f"QPSK Demodulated:   {qpsk_demod_bits}")
    print(f"QPSK BER: {np.mean(qpsk_orig_bits != qpsk_demod_bits):.3f}")


    print("\n--- 16-QAM Demodulation Test ---")
    # For testing, let's manually create some ideal symbols and add noise
    # Bit sequence: (b3 b2 b1 b0)
    # 0000 -> I=-3, Q=-3
    # 0101 -> I=-1, Q=-1
    # 1111 -> I=1,  Q=1
    # 1010 -> I=3,  Q=3
    # 0011 -> I=-3, Q=1
    original_bits_qam16 = np.array([
        0,0,0,0,  0,1,0,1,  1,1,1,1,  1,0,1,0,  0,0,1,1
    ])

    # Corresponding ideal symbols (normalized)
    # Need to generate these based on the mapping, or use transmitter's output if available
    # For now, let's reconstruct them for self-contained test
    # (0,0)->-3; (0,1)->-1; (1,1)->1; (1,0)->3
    s0000 = (-3 - 3j) * _QAM16_NORMALIZATION_FACTOR
    s0101 = (-1 - 1j) * _QAM16_NORMALIZATION_FACTOR
    s1111 = ( 1 + 1j) * _QAM16_NORMALIZATION_FACTOR
    s1010 = ( 3 + 3j) * _QAM16_NORMALIZATION_FACTOR
    s0011 = (-3 + 1j) * _QAM16_NORMALIZATION_FACTOR
    ideal_symbols_qam16 = np.array([s0000, s0101, s1111, s1010, s0011])

    print(f"Original 16-QAM bits: {original_bits_qam16}")
    # print(f"Ideal 16-QAM symbols: {np.round(ideal_symbols_qam16,3)}")

    # Simulate some received symbols with noise
    noise_std_dev_qam16 = 0.1 # Adjust noise level
    noise_qam16 = (np.random.normal(0, noise_std_dev_qam16, size=ideal_symbols_qam16.shape) +
                   1j * np.random.normal(0, noise_std_dev_qam16, size=ideal_symbols_qam16.shape))
    received_noisy_symbols_qam16 = ideal_symbols_qam16 + noise_qam16
    # print(f"Received noisy 16-QAM symbols: {np.round(received_noisy_symbols_qam16,3)}")

    # 1. Ideal Coherent Detection (placeholder)
    detected_symbols_qam16 = coherent_detect(received_noisy_symbols_qam16)
    # 2. DSP/Equalizer (placeholder)
    equalized_symbols_qam16 = apply_equalizer(detected_symbols_qam16)

    # 3. 16-QAM Demodulation
    demodulated_bits_qam16 = qam16_demodulate(equalized_symbols_qam16)
    print(f"Demodulated 16-QAM bits: {demodulated_bits_qam16}")

    num_errors_qam16 = np.sum(original_bits_qam16 != demodulated_bits_qam16)
    ber_qam16 = num_errors_qam16 / original_bits_qam16.size
    print(f"Number of bit errors (16-QAM): {num_errors_qam16} out of {original_bits_qam16.size}")
    print(f"Bit Error Rate (BER) (16-QAM): {ber_qam16:.4f}")

    # Test error handling for qam16_demodulate
    print("\nTesting error handling for qam16_demodulate...")
    try:
        qam16_demodulate(np.array([1,2,3], dtype=float)) # Not complex
    except ValueError as e:
        print(f"Caught expected error: {e}")
    try:
        qam16_demodulate(np.array([[1+1j],[0+0j]])) # Wrong dimensions
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Test empty array input
    print("Testing empty array input for qam16_demodulate:")
    empty_demod_bits = qam16_demodulate(np.array([], dtype=np.complex128))
    print(f"Result for empty input: {empty_demod_bits}, length: {len(empty_demod_bits)}")
    assert len(empty_demod_bits) == 0

    print("\nReceiver module tests complete.")
