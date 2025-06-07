import numpy as np

# --- QPSK Specific ---
def generate_bits(num_bits: int) -> np.ndarray:
    """
    Generates a NumPy array of random integers (0 or 1).

    Args:
        num_bits: The number of bits to generate.

    Returns:
        A NumPy array of length num_bits with random 0s and 1s.
    """
    if not isinstance(num_bits, int) or num_bits <= 0:
        raise ValueError("num_bits must be a positive integer.")
    return np.random.randint(0, 2, num_bits)

def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """
    Modulates a sequence of bits using QPSK with Gray coding.

    The mapping is:
        00 -> (1+1j)/sqrt(2)
        01 -> (-1+1j)/sqrt(2)
        11 -> (-1-1j)/sqrt(2)
        10 -> (1-1j)/sqrt(2)

    Args:
        bits: A NumPy array of 0s and 1s. Must have an even number of bits.

    Returns:
        A NumPy array of complex QPSK symbols.
    """
    if not isinstance(bits, np.ndarray) or bits.ndim != 1:
        raise ValueError("Input 'bits' must be a 1D NumPy array.")
    if bits.size % 2 != 0:
        raise ValueError("The number of bits must be even for QPSK modulation.")
    if not np.all(np.logical_or(bits == 0, bits == 1)):
        raise ValueError("All elements in 'bits' must be either 0 or 1.")

    num_symbols = bits.size // 2
    symbols = np.empty(num_symbols, dtype=np.complex128)

    norm_qpsk = 1 / np.sqrt(2)

    for i in range(num_symbols):
        bit1 = bits[2*i]
        bit2 = bits[2*i+1]

        if bit1 == 0 and bit2 == 0:  # 00
            symbols[i] = (1 + 1j) * norm_qpsk
        elif bit1 == 0 and bit2 == 1:  # 01
            symbols[i] = (-1 + 1j) * norm_qpsk
        elif bit1 == 1 and bit2 == 1:  # 11
            symbols[i] = (-1 - 1j) * norm_qpsk
        elif bit1 == 1 and bit2 == 0:  # 10
            symbols[i] = (1 - 1j) * norm_qpsk

    return symbols

# --- 16-QAM Specific ---

# Gray mapping for 2 bits to {-3, -1, 1, 3}
# MSB first for mapping: e.g., b1 b0
GRAY_MAP_2_BIT_TO_LEVEL = {
    (0, 0): -3,
    (0, 1): -1,
    (1, 1): 1,  # Note: (b1=1, b0=1) -> 1
    (1, 0): 3   # Note: (b1=1, b0=0) -> 3
}
# Reverse mapping for convenience if needed elsewhere, but not directly in modulator
LEVEL_TO_GRAY_MAP_2_BIT = {v: k for k, v in GRAY_MAP_2_BIT_TO_LEVEL.items()}

# Normalization factor for 16-QAM to achieve average power of 1
# Average energy of I/Q component ({-3,-1,1,3}) is (9+1+1+9)/4 = 5.
# Total average symbol energy is E[I^2] + E[Q^2] = 5 + 5 = 10.
QAM16_NORMALIZATION_FACTOR = 1 / np.sqrt(10)

def qam16_modulate(bits: np.ndarray) -> np.ndarray:
    """
    Modulates a sequence of bits using 16-QAM with Gray coding.
    The constellation is normalized to have an average power of 1.

    Bit mapping (b3 b2 b1 b0):
      - b3 b2 determine the In-phase (I) component.
      - b1 b0 determine the Quadrature (Q) component.
    Gray coding for I/Q components (msb lsb -> level):
      - 00 -> -3
      - 01 -> -1
      - 11 ->  1
      - 10 ->  3

    Args:
        bits: A NumPy array of 0s and 1s. Length must be a multiple of 4.

    Returns:
        A NumPy array of complex 16-QAM symbols, normalized.
    """
    if not isinstance(bits, np.ndarray) or bits.ndim != 1:
        raise ValueError("Input 'bits' must be a 1D NumPy array.")
    if bits.size == 0: # Handle empty array case
        return np.array([], dtype=np.complex128)
    if bits.size % 4 != 0:
        raise ValueError("The number of bits must be a multiple of 4 for 16-QAM modulation.")
    if not np.all(np.logical_or(bits == 0, bits == 1)):
        raise ValueError("All elements in 'bits' must be either 0 or 1.")

    num_symbols = bits.size // 4
    symbols = np.empty(num_symbols, dtype=np.complex128)

    for i in range(num_symbols):
        b3 = bits[4*i]     # I-axis MSB
        b2 = bits[4*i+1]   # I-axis LSB
        b1 = bits[4*i+2]   # Q-axis MSB
        b0 = bits[4*i+3]   # Q-axis LSB

        val_i = GRAY_MAP_2_BIT_TO_LEVEL[(b3, b2)]
        val_q = GRAY_MAP_2_BIT_TO_LEVEL[(b1, b0)]

        symbols[i] = (val_i + 1j * val_q) * QAM16_NORMALIZATION_FACTOR

    return symbols


if __name__ == '__main__':
    print("--- QPSK Modulation Example ---")
    num_bits_qpsk = 10
    random_bits_qpsk = generate_bits(num_bits_qpsk)
    print(f"Generated bits for QPSK: {random_bits_qpsk}")
    # Ensure even number of bits for QPSK
    if num_bits_qpsk % 2 != 0:
        random_bits_qpsk = random_bits_qpsk[:-1] # Truncate if odd
        print(f"Truncated bits for QPSK: {random_bits_qpsk}")

    if random_bits_qpsk.size > 0:
        qpsk_symbols = qpsk_modulate(random_bits_qpsk)
        print(f"QPSK modulated symbols: {np.round(qpsk_symbols,3)}")
        print(f"QPSK Avg power: {np.mean(np.abs(qpsk_symbols)**2):.3f}")

    print("\n--- 16-QAM Modulation Example ---")
    num_bits_qam16 = 16 # e.g. 4 symbols
    random_bits_qam16 = generate_bits(num_bits_qam16)
    print(f"Generated bits for 16-QAM: {random_bits_qam16}")

    # Test with a known sequence for 16-QAM
    # b3 b2 b1 b0
    # 00 00 -> I=-3, Q=-3 -> (-3-3j)*norm
    # 01 01 -> I=-1, Q=-1 -> (-1-1j)*norm
    # 11 11 -> I=1,  Q=1  -> (1+1j)*norm
    # 10 10 -> I=3,  Q=3  -> (3+3j)*norm
    # 00 01 -> I=-3, Q=-1 -> (-3-1j)*norm
    test_bits_qam16 = np.array([
        0,0,0,0,  0,1,0,1,  1,1,1,1,  1,0,1,0,  0,0,0,1
    ])
    print(f"Test bits for 16-QAM: {test_bits_qam16}")
    qam16_symbols = qam16_modulate(test_bits_qam16)
    print(f"16-QAM modulated symbols (normalized):")
    for k, s in enumerate(qam16_symbols):
        print(f"  Bits {test_bits_qam16[4*k:4*k+4]} -> Symbol {s.real/QAM16_NORMALIZATION_FACTOR:.0f}{s.imag/QAM16_NORMALIZATION_FACTOR:+.0f}j * C = {s:.3f}")

    print(f"16-QAM Avg power (test_bits): {np.mean(np.abs(qam16_symbols)**2):.3f} (Expected ~1.0)")

    # Test with random bits
    qam16_symbols_rand = qam16_modulate(random_bits_qam16)
    print(f"16-QAM modulated symbols (random, normalized): {np.round(qam16_symbols_rand,3)}")
    print(f"16-QAM Avg power (random_bits): {np.mean(np.abs(qam16_symbols_rand)**2):.3f} (Expected ~1.0)")

    # Test error handling for qam16_modulate
    print("\nTesting error handling for qam16_modulate...")
    try:
        qam16_modulate(np.array([0,1,0])) # Not multiple of 4
    except ValueError as e:
        print(f"Caught expected error: {e}")
    try:
        qam16_modulate(np.array([0,1,0,2])) # Invalid bit value
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Test empty array input
    print("Testing empty array input for qam16_modulate:")
    empty_syms = qam16_modulate(np.array([], dtype=int))
    print(f"Result for empty input: {empty_syms}, length: {len(empty_syms)}")
    assert len(empty_syms) == 0

    print("Transmitter module tests complete.")
