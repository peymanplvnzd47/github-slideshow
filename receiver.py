import numpy as np

def coherent_detect(signal: np.ndarray) -> np.ndarray:
    """
    Placeholder for ideal coherent detection.
    Currently returns the input signal without modification.

    Args:
        signal: NumPy array of complex symbols (e.g., after optical channel).

    Returns:
        NumPy array of complex symbols (ideally, baseband electrical signal).
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Input signal must be a NumPy array.")
    return signal

def apply_equalizer(signal: np.ndarray) -> np.ndarray:
    """
    Placeholder for DSP/Equalizer.
    Currently returns the input signal without modification.

    Args:
        signal: NumPy array of complex symbols (e.g., after coherent detection).

    Returns:
        NumPy array of complex symbols (ideally, equalized signal).
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Input signal must be a NumPy array.")
    return signal

def qpsk_demodulate(received_symbols: np.ndarray) -> np.ndarray:
    """
    Demodulates QPSK symbols to bits using minimum Euclidean distance.

    Constellation mapping (symbol -> bits):
        (1+1j)/sqrt(2)  -> 00
        (-1+1j)/sqrt(2) -> 01
        (-1-1j)/sqrt(2) -> 11
        (1-1j)/sqrt(2)  -> 10

    Args:
        received_symbols: A NumPy array of complex QPSK symbols.

    Returns:
        A NumPy array of demodulated bits (0s and 1s).
    """
    if not isinstance(received_symbols, np.ndarray) or received_symbols.ndim != 1:
        raise ValueError("Input 'received_symbols' must be a 1D NumPy array.")
    if not np.issubdtype(received_symbols.dtype, np.complexfloating):
        raise ValueError("Input 'received_symbols' must be of complex type.")

    norm_factor = 1 / np.sqrt(2)
    constellation = {
        (0, 0): (1 + 1j) * norm_factor,
        (0, 1): (-1 + 1j) * norm_factor,
        (1, 1): (-1 - 1j) * norm_factor,
        (1, 0): (1 - 1j) * norm_factor,
    }
    
    # Ideal constellation points and their corresponding bit pairs
    # Order is important for indexing later if needed, but here we map directly
    ideal_points = np.array(list(constellation.values())) # Shape: (4,) complex numbers
    bit_pairs = np.array(list(constellation.keys()))     # Shape: (4, 2) integers

    demodulated_bits = []
    for symbol in received_symbols: # symbol is a complex scalar
        # Calculate squared Euclidean distances to each ideal point
        # np.abs(complex_number) gives magnitude
        # (magnitude)^2 is squared magnitude (which is fine for distance comparison)
        # (symbol - ideal_points) results in an array of 4 complex numbers
        # np.abs(array_of_complex)**2 results in an array of 4 real numbers (squared magnitudes)
        distances_sq = np.abs(symbol - ideal_points)**2
        
        # Find the index of the closest ideal point
        closest_idx = np.argmin(distances_sq) # Index from 0 to 3
        
        # Append the corresponding bit pair
        demodulated_bits.extend(bit_pairs[closest_idx])
        
    return np.array(demodulated_bits, dtype=int)

if __name__ == '__main__':
    print("--- QPSK Demodulation Test ---")
    
    norm = 1 / np.sqrt(2)
    # Ideal constellation points
    s00 = (1 + 1j) * norm
    s01 = (-1 + 1j) * norm
    s11 = (-1 - 1j) * norm
    s10 = (1 - 1j) * norm

    # Original bits and corresponding ideal symbols
    original_bit_sequence = np.array([0,0, 0,1, 1,1, 1,0, 0,0, 1,0])
    ideal_transmitted_symbols = np.array([s00, s01, s11, s10, s00, s10])
    
    print(f"Original bit sequence: {original_bit_sequence}")
    print(f"Ideal transmitted symbols: {np.round(ideal_transmitted_symbols,3)}")

    # Simulate some received symbols with noise
    # Noise power (variance for each dimension, real and imag)
    noise_variance_per_dim = 0.05 # Adjust to see impact of noise
    noise_std_dev = np.sqrt(noise_variance_per_dim)
    
    noise = np.random.normal(0, noise_std_dev, size=ideal_transmitted_symbols.shape) + \
            1j * np.random.normal(0, noise_std_dev, size=ideal_transmitted_symbols.shape)
    
    received_noisy_symbols = ideal_transmitted_symbols + noise
    print(f"Received noisy symbols: {np.round(received_noisy_symbols,3)}")

    # 1. Ideal Coherent Detection (placeholder)
    detected_symbols = coherent_detect(received_noisy_symbols)
    print(f"Symbols after coherent detect (placeholder): {np.round(detected_symbols,3)}")

    # 2. DSP/Equalizer (placeholder)
    equalized_symbols = apply_equalizer(detected_symbols)
    print(f"Symbols after equalizer (placeholder): {np.round(equalized_symbols,3)}")

    # 3. QPSK Demodulation
    demodulated_bits = qpsk_demodulate(equalized_symbols)
    print(f"Demodulated bits: {demodulated_bits}")

    # Check for bit errors
    num_errors = np.sum(original_bit_sequence != demodulated_bits)
    ber = num_errors / original_bit_sequence.size
    print(f"Number of bit errors: {num_errors} out of {original_bit_sequence.size}")
    print(f"Bit Error Rate (BER): {ber:.4f}")

    # Test with a symbol exactly at a decision boundary (e.g., real part is 0)
    # This symbol is equidistant from s00 and s01 (and s10 and s11 in y-dim)
    # Example: 0 + 1j * norm. Should map to 01 or 00 depending on tie-breaking in argmin
    # (np.argmin typically returns the first occurrence of the minimum).
    test_boundary_symbol = np.array([0 + 1j * norm]) 
    demod_boundary = qpsk_demodulate(test_boundary_symbol)
    print(f"\nDemodulating boundary symbol {np.round(test_boundary_symbol,3)} -> bits {demod_boundary}")
    # Expected: (0,1) i.e., [-1+1j]/sqrt(2) because -1 is closer to 0 than 1 if only considering x-axis,
    # but distance calculation is done to the full complex points.
    # Distances for 0 + 1j/sqrt(2):
    # to s00 (1+1j)/sqrt(2): |(0-1) + (1-1)j|^2 / 2 = 1/2
    # to s01 (-1+1j)/sqrt(2): |(0-(-1)) + (1-1)j|^2 / 2 = 1/2
    # to s11 (-1-1j)/sqrt(2): |(0-(-1)) + (1-(-1))j|^2 / 2 = |1+2j|^2/2 = (1+4)/2 = 5/2
    # to s10 (1-1j)/sqrt(2): |(0-1) + (1-(-1))j|^2 / 2 = |-1+2j|^2/2 = (1+4)/2 = 5/2
    # np.argmin will pick the first one, which is index 0 (s00 -> 00)
    # This behavior is fine.

    print("\n--- Testing Error Handling ---")
    try:
        qpsk_demodulate(np.array([1,2,3])) # Not complex
    except ValueError as e:
        print(f"Caught expected error for qpsk_demodulate (not complex): {e}")
    try:
        qpsk_demodulate(np.array([[1+1j],[0+0j]])) # Wrong dimensions
    except ValueError as e:
        print(f"Caught expected error for qpsk_demodulate (wrong dimensions): {e}")

    print("\nReceiver module tests complete.")
