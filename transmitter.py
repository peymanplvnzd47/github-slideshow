import numpy as np

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
    
    # Normalization factor for power = 1
    norm = 1 / np.sqrt(2)

    for i in range(num_symbols):
        bit1 = bits[2*i]
        bit2 = bits[2*i+1]

        if bit1 == 0 and bit2 == 0:  # 00
            symbols[i] = (1 + 1j) * norm
        elif bit1 == 0 and bit2 == 1:  # 01
            symbols[i] = (-1 + 1j) * norm
        elif bit1 == 1 and bit2 == 1:  # 11
            symbols[i] = (-1 - 1j) * norm
        elif bit1 == 1 and bit2 == 0:  # 10
            symbols[i] = (1 - 1j) * norm
            
    return symbols

if __name__ == '__main__':
    # Example Usage
    num_bits_to_generate = 10
    random_bits = generate_bits(num_bits_to_generate)
    print(f"Generated bits: {random_bits}")

    # Ensure even number of bits for QPSK
    if num_bits_to_generate % 2 != 0:
        print(f"Number of bits ({num_bits_to_generate}) is odd. QPSK modulation requires an even number of bits.")
        # Optionally, pad or truncate here, or re-generate
        # For this example, let's try with an even number if it was odd
        if num_bits_to_generate > 1:
             random_bits_even = random_bits[:-1] # simple truncation for example
             print(f"Using truncated bits for QPSK: {random_bits_even}")
             if random_bits_even.size > 0 :
                qpsk_symbols = qpsk_modulate(random_bits_even)
                print(f"QPSK modulated symbols: {qpsk_symbols}")
                print(f"Average symbol power: {np.mean(np.abs(qpsk_symbols)**2)}")
        else:
            print("Not enough bits to form a symbol after truncation.")

    else:
        qpsk_symbols = qpsk_modulate(random_bits)
        print(f"QPSK modulated symbols: {qpsk_symbols}")
        print(f"Average symbol power: {np.mean(np.abs(qpsk_symbols)**2)}")

    # Test with a known sequence
    test_bits = np.array([0,0, 0,1, 1,1, 1,0])
    print(f"\nTest bits: {test_bits}")
    test_symbols = qpsk_modulate(test_bits)
    print(f"Test QPSK symbols: {test_symbols}")
    expected_symbols = np.array([
        (1+1j), (-1+1j), (-1-1j), (1-1j)
    ]) / np.sqrt(2)
    print(f"Expected symbols: {expected_symbols}")
    assert np.allclose(test_symbols, expected_symbols), "QPSK modulation for test sequence failed!"
    print("Test sequence modulation successful.")

    # Test error handling
    print("\nTesting error handling...")
    try:
        generate_bits(-5)
    except ValueError as e:
        print(f"Caught expected error for generate_bits: {e}")

    try:
        generate_bits(0)
    except ValueError as e:
        print(f"Caught expected error for generate_bits: {e}")
        
    try:
        qpsk_modulate(np.array([0,1,0])) # Odd number of bits
    except ValueError as e:
        print(f"Caught expected error for qpsk_modulate (odd bits): {e}")

    try:
        qpsk_modulate(np.array([0,2,1,0])) # Invalid bit value
    except ValueError as e:
        print(f"Caught expected error for qpsk_modulate (invalid bit value): {e}")
    
    try:
        qpsk_modulate(np.array([[0,1],[1,0]])) # Incorrect dimensions
    except ValueError as e:
        print(f"Caught expected error for qpsk_modulate (wrong dimensions): {e}")

    print("Error handling tests complete.")
