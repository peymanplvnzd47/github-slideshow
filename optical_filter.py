from scipy.special import erf
import numpy as np
import matplotlib.pyplot as plt

def filter_transfer_function(f: np.ndarray, B_0: float, sigma: float) -> np.ndarray:
    """
    Calculates the super-Gaussian filter transfer function.
    H(f) = 0.5 * [erf((B_0/2 + f) / (sqrt(2) * sigma)) - erf((f - B_0/2) / (sqrt(2) * sigma))]

    Args:
        f (np.ndarray): Frequency array (Hz).
        B_0 (float): 6-dB bandwidth of the filter (Hz).
        sigma (float): Sigma parameter defining the filter slope.

    Returns:
        np.ndarray: Complex filter transfer function H(f).
                    (This specific erf-based definition results in a real-valued H(f))
    """
    if sigma <= 0:
        # Handle sigma=0 case: ideal brickwall filter
        # This erf formulation might become problematic with sigma=0 due to division by zero.
        # A true brickwall would be 1 within B_0/2 and 0 outside.
        # For simplicity, if sigma is very small, erf will approximate this.
        # Let's prevent division by zero for now.
        # A very small sigma will lead to very steep slopes.
        sigma = 1e-9 # A very small number to approximate steep slope if sigma is zero or negative
        # print("Warning: sigma was <= 0, using a very small value instead.")

    term1_arg = (B_0 / 2 + f) / (np.sqrt(2) * sigma)
    term2_arg = (f - B_0 / 2) / (np.sqrt(2) * sigma)

    term1 = erf(term1_arg)
    term2 = erf(term2_arg)

    H = 0.5 * (term1 - term2)
    return H

def apply_filter(
    signal: np.ndarray,
    symbol_rate_hz: float,
    B_0: float,
    sigma: float,
    filter_center_offset_hz: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies a super-Gaussian optical filter to a signal in the frequency domain.

    Args:
        signal (np.ndarray): Complex time-domain input signal.
        symbol_rate_hz (float): Symbol rate of the signal (Hz).
        B_0 (float): 6-dB bandwidth of the filter (Hz).
        sigma (float): Sigma parameter for the filter slope.
        filter_center_offset_hz (float, optional): Offset of the filter's center
                                                   frequency from 0 Hz. Defaults to 0.0.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - signal_filtered (np.ndarray): Filtered complex time-domain signal.
            - H (np.ndarray): The filter transfer function (not shifted).
            - freq_axis (np.ndarray): The frequency axis corresponding to H (not shifted).
    """
    if signal.size == 0:
        return np.array([], dtype=signal.dtype), np.array([]), np.array([])

    # 1. FFT the input signal
    S = np.fft.fft(signal)

    # 2. Create frequency axis
    n_points = len(signal)
    freq_axis = np.fft.fftfreq(n_points, d=1/symbol_rate_hz)

    # 3. Calculate frequencies relative to filter center
    f_relative = freq_axis - filter_center_offset_hz

    # 4. Generate filter response H(f)
    H = filter_transfer_function(f_relative, B_0, sigma)

    # 5. Apply filter in frequency domain
    S_filtered = S * H

    # 6. IFFT back to time domain
    signal_filtered = np.fft.ifft(S_filtered)

    return signal_filtered, H, freq_axis

if __name__ == '__main__':
    print("--- Testing optical_filter.py ---")

    # Example parameters for filter_transfer_function
    sample_symbol_rate_hz = 32e9  # 32 GHz
    B_0_test = 0.8 * sample_symbol_rate_hz  # Example: 80% of symbol rate
    sigma_test_1 = B_0_test / 4.0 # Sharper filter
    sigma_test_2 = B_0_test / 10.0 # Steeper, closer to problem statement intent
    sigma_test_3 = B_0_test / 2.0 # Softer filter

    # Create a frequency axis for testing H(f)
    # For plotting H(f), we usually want a shifted view
    num_freq_points = 2048
    freq_plot_axis_hz = np.linspace(-sample_symbol_rate_hz * 1.5, sample_symbol_rate_hz * 1.5, num_freq_points)

    H_test_1 = filter_transfer_function(freq_plot_axis_hz, B_0_test, sigma_test_1)
    H_test_2 = filter_transfer_function(freq_plot_axis_hz, B_0_test, sigma_test_2)
    H_test_3 = filter_transfer_function(freq_plot_axis_hz, B_0_test, sigma_test_3)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(freq_plot_axis_hz / 1e9, np.abs(H_test_1), label=f'Sigma = B0/4 (Sharper)')
    plt.plot(freq_plot_axis_hz / 1e9, np.abs(H_test_2), label=f'Sigma = B0/10 (Steeper)')
    plt.plot(freq_plot_axis_hz / 1e9, np.abs(H_test_3), label=f'Sigma = B0/2 (Softer)')
    plt.title(f'Filter Transfer Function H(f) (Linear Magnitude) for B0 = {B_0_test/1e9:.1f} GHz')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('|H(f)|')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    # Add a small epsilon for log calculation to avoid log(0)
    epsilon = 1e-20
    plt.plot(freq_plot_axis_hz / 1e9, 20 * np.log10(np.abs(H_test_1) + epsilon), label=f'Sigma = B0/4')
    plt.plot(freq_plot_axis_hz / 1e9, 20 * np.log10(np.abs(H_test_2) + epsilon), label=f'Sigma = B0/10')
    plt.plot(freq_plot_axis_hz / 1e9, 20 * np.log10(np.abs(H_test_3) + epsilon), label=f'Sigma = B0/2')
    plt.title(f'Filter Transfer Function H(f) (dB) for B0 = {B_0_test/1e9:.1f} GHz')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('20log10|H(f)| (dB)')
    plt.ylim(-60, 5) # Typical dB range for filters
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("test_filter_transfer_function.png")
    print("Saved test_filter_transfer_function.png")
    # plt.show() # Commented out for non-interactive environments
    plt.close()

    # Test apply_filter
    print("\n--- Testing apply_filter ---")
    num_symbols_test = 1024
    # Create a simple test signal (e.g. sum of sines or random signal)
    # For optical context, a baseband complex signal is typical
    test_signal_time_domain = np.random.randn(num_symbols_test) + 1j * np.random.randn(num_symbols_test)

    B_0_apply = 0.9 * sample_symbol_rate_hz
    sigma_apply = B_0_apply / 10.0

    filtered_signal, H_applied, freq_axis_applied = apply_filter(
        test_signal_time_domain,
        sample_symbol_rate_hz,
        B_0_apply,
        sigma_apply
    )
    print(f"Apply filter: Input signal length: {len(test_signal_time_domain)}")
    print(f"Apply filter: Filtered signal length: {len(filtered_signal)}")
    print(f"Apply filter: H_applied length: {len(H_applied)}")
    print(f"Apply filter: freq_axis_applied length: {len(freq_axis_applied)}")

    # Plot PSD of original and filtered signal
    plt.figure(figsize=(10,6))

    # PSD of original
    S_orig = np.fft.fftshift(np.fft.fft(test_signal_time_domain)/len(test_signal_time_domain))
    psd_orig_db = 10*np.log10(np.maximum(np.abs(S_orig)**2, epsilon))
    psd_orig_db_norm = psd_orig_db - np.max(psd_orig_db)
    freq_axis_shifted_ghz = np.fft.fftshift(freq_axis_applied) / 1e9

    plt.plot(freq_axis_shifted_ghz, psd_orig_db_norm, label='Original Signal PSD')

    # PSD of filtered
    S_filt = np.fft.fftshift(np.fft.fft(filtered_signal)/len(filtered_signal))
    psd_filt_db = 10*np.log10(np.maximum(np.abs(S_filt)**2, epsilon))
    psd_filt_db_norm = psd_filt_db - np.max(psd_filt_db)
    plt.plot(freq_axis_shifted_ghz, psd_filt_db_norm, label='Filtered Signal PSD')

    # Plot filter shape (H_applied is not shifted, need to shift for plotting with shifted freq axis)
    H_applied_shifted_abs_db = 20*np.log10(np.abs(np.fft.fftshift(H_applied)) + epsilon)
    # Normalize H to plot on same scale as PSDs (e.g. 0dB peak)
    H_applied_shifted_abs_db_norm = H_applied_shifted_abs_db - np.max(H_applied_shifted_abs_db)
    plt.plot(freq_axis_shifted_ghz, H_applied_shifted_abs_db_norm, label='Filter Response H(f) (Normalized)', linestyle='--')

    plt.title(f'PSD Before and After Filtering (B0={B_0_apply/1e9:.1f}GHz, sigma=B0/10)')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Normalized PSD (dB)')
    plt.ylim(-60, 5)
    plt.legend()
    plt.grid(True)
    plt.savefig("test_apply_filter_psd.png")
    print("Saved test_apply_filter_psd.png")
    plt.close()

    print("Optical filter tests complete.")
