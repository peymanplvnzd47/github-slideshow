import numpy as np
import transmitter
import optical_channel
import receiver
import optical_filter
import matplotlib.pyplot as plt
import time

# --- System Parameters & Constants ---
NUM_BITS_TO_SIMULATE_DEFAULT = 2**18
BITS_PER_SYMBOL_QPSK = 2
BITS_PER_SYMBOL_QAM16 = 4
SYMBOL_RATE_GBD = 32.0
NUM_CHANNELS = 1

FIBER_SPAN_LENGTH_KM_DEFAULT = 80.0
ATTENUATION_DB_PER_KM_DEFAULT = 0.2
EDFA_GAIN_DB_DEFAULT = FIBER_SPAN_LENGTH_KM_DEFAULT * ATTENUATION_DB_PER_KM_DEFAULT # Gain compensates loss

FIXED_NUM_SPANS_FOR_SWEEP = 10 # For Es/N0 sweep
FIXED_EDFA_NF_DB_FOR_SWEEP = 5.0 # For Es/N0 sweep
NUM_BITS_FOR_SWEEP = 2**18 # For Es/N0 sweep

# Constants for OSNR Calculation
H_PLANCK = 6.62607015e-34  # J*Hz^-1
C_LIGHT = 299792458.0  # m/s
# Using 1550nm as reference wavelength for C-band center for OSNR BW calculation
REF_LAMBDA_M_FOR_OSNR_BW = 1550e-9 # meters
OPTICAL_FREQ_HZ_CENTER_C_BAND = C_LIGHT / REF_LAMBDA_M_FOR_OSNR_BW # approx 193.4 THz
OSNR_REF_BW_NM = 0.1e-9  # 0.1 nm in meters
# OSNR Reference Bandwidth in Hz: df = (c/lambda^2) * dlambda
OSNR_REF_BW_HZ = (C_LIGHT / (REF_LAMBDA_M_FOR_OSNR_BW**2)) * OSNR_REF_BW_NM # Approx 12.48 GHz


# --- Helper function for Ideal Constellations (unchanged) ---
def get_ideal_constellation(modulation_format: str) -> np.ndarray:
    if modulation_format == "QPSK":
        norm_qpsk = 1 / np.sqrt(2)
        points = [(1 + 1j) * norm_qpsk, (-1 + 1j) * norm_qpsk, (-1 - 1j) * norm_qpsk, (1 - 1j) * norm_qpsk]
        return np.array(points, dtype=np.complex128)
    elif modulation_format == "16-QAM":
        levels = [-3, -1, 1, 3]
        qam16_norm_factor = 1 / np.sqrt(10)
        points = [(i_level + 1j * q_level) * qam16_norm_factor for i_level in levels for q_level in levels]
        return np.array(points, dtype=np.complex128)
    else: raise ValueError(f"Unsupported modulation_format: {modulation_format}")

# --- Plotting Functions (plot_constellation, plot_psd unchanged) ---
def plot_constellation(
    symbols: np.ndarray, title: str, ideal_constellation: np.ndarray = None,
    filename: str = None, snr_value: float = None
):
    plt.figure(figsize=(8, 8)); plt.scatter(np.real(symbols), np.imag(symbols), s=10, alpha=0.3, label="Symbols")
    if ideal_constellation is not None:
        plt.scatter(np.real(ideal_constellation), np.imag(ideal_constellation), s=50, marker='x', color='red', label="Ideal Points")
    plot_title = title
    if snr_value is not None: plot_title += f" (Es/N0 ~ {snr_value:.1f} dB)"
    plt.title(plot_title); plt.xlabel("In-Phase"); plt.ylabel("Quadrature")
    plt.axhline(0, color='grey', lw=0.5); plt.axvline(0, color='grey', lw=0.5)
    plt.grid(True, linestyle=':', alpha=0.7); plt.axis('equal')
    max_val = np.max(np.abs(ideal_constellation)) * 1.5 if ideal_constellation is not None and ideal_constellation.size > 0 else \
              np.max(np.abs(symbols)) * 1.2 if symbols.size > 0 else 1.5
    if not (np.isfinite(max_val) and max_val > 0): max_val = 1.5
    plt.xlim(-max_val, max_val); plt.ylim(-max_val, max_val)
    if ideal_constellation is not None: plt.legend()
    if filename: plt.savefig(filename); print(f"Constellation plot saved to {filename}")
    else: plt.show()
    plt.close()

def plot_psd(
    signal: np.ndarray, symbol_rate_hz: float, title: str,
    filename: str = None, filter_response_hz: tuple = None
):
    if signal.size == 0: print(f"Warning: Cannot plot PSD for empty signal. Title: {title}"); return
    S = np.fft.fft(signal) / len(signal); S_shifted = np.fft.fftshift(S); psd = np.abs(S_shifted)**2
    epsilon = 1e-20; psd_db = 10 * np.log10(np.maximum(psd, epsilon)); psd_db_normalized = psd_db - np.max(psd_db)
    n_points = len(signal); freq_axis = np.fft.fftfreq(n_points, d=1/symbol_rate_hz)
    freq_axis_shifted = np.fft.fftshift(freq_axis); freq_axis_ghz = freq_axis_shifted / 1e9
    plt.figure(figsize=(10, 6)); plt.plot(freq_axis_ghz, psd_db_normalized, label="Signal PSD")
    if filter_response_hz is not None:
        H_shape, freq_axis_for_H_hz = filter_response_hz
        H_shifted_abs = np.abs(np.fft.fftshift(H_shape)); freq_axis_for_H_ghz = np.fft.fftshift(freq_axis_for_H_hz) / 1e9
        H_db = 20 * np.log10(np.maximum(H_shifted_abs, epsilon)); H_db_normalized = H_db - np.max(H_db)
        plt.plot(freq_axis_for_H_ghz, H_db_normalized, label="Filter Response (Norm.)", linestyle='--'); plt.legend()
    plt.title(title); plt.xlabel("Frequency (GHz)"); plt.ylabel("Normalized PSD (dB)")
    plt.ylim(-60, 5); plt.grid(True, which="both", ls="-")
    if filename: plt.savefig(filename); print(f"PSD plot saved to {filename}")
    else: plt.show()
    plt.close()

# --- Simulation Functions (run_simulation_single_point unchanged from previous correct version) ---
def run_simulation_single_point(
    num_bits: int, symbol_rate: float, num_spans: int, span_length_km: float, attenuation_db_km: float,
    edfa_gain_db: float, edfa_nf_db: float, modulation_format: str = "QPSK",
    add_awgn_at_receiver_snr_db: float = None, num_wdm_channels: int = 1, verbose: bool = True
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    if modulation_format == "QPSK": bits_per_symbol = BITS_PER_SYMBOL_QPSK
    elif modulation_format == "16-QAM": bits_per_symbol = BITS_PER_SYMBOL_QAM16
    else: raise ValueError(f"Unsupported modulation_format: {modulation_format}")

    if num_bits % bits_per_symbol != 0:
        original_num_bits = num_bits; num_bits = (num_bits // bits_per_symbol) * bits_per_symbol
        if verbose: print(f"Warning: num_bits adjusted from {original_num_bits} to {num_bits} for {modulation_format}.")
    if num_bits == 0:
        if verbose: print("Warning: num_bits is 0. Returning NaN BER and empty arrays.")
        empty_complex_array = np.array([], dtype=np.complex128)
        return np.nan, 0.0, 0.0, empty_complex_array, empty_complex_array, empty_complex_array

    if verbose:
        log_prefix = f"--- Running simulation (Mod: {modulation_format}, Bits: {num_bits}"
        if add_awgn_at_receiver_snr_db is not None: log_prefix += f", Target Rx SNR: {add_awgn_at_receiver_snr_db:.1f} dB"
        else: log_prefix += f", Channel NF: {edfa_nf_db:.1f} dB"
        log_prefix += ") ---"; print(log_prefix)

    transmitted_bits = transmitter.generate_bits(num_bits)
    if modulation_format == "QPSK": modulated_symbols = transmitter.qpsk_modulate(transmitted_bits)
    elif modulation_format == "16-QAM": modulated_symbols = transmitter.qam16_modulate(transmitted_bits)
    tx_power_avg = np.mean(np.abs(modulated_symbols)**2)
    if verbose: print(f"TX: Average signal power = {tx_power_avg:.4f}")

    symbols_after_channel = optical_channel.propagate_through_spans(
        signal=modulated_symbols, num_spans=num_spans, fiber_length_km=span_length_km,
        attenuation_db_per_km=attenuation_db_km, edfa_gain_db=edfa_gain_db,
        edfa_noise_figure_db=edfa_nf_db, symbol_rate_gbd=symbol_rate, num_channels=num_wdm_channels
    )
    detected_symbols = receiver.coherent_detect(symbols_after_channel)
    equalized_symbols_pre_awgn = receiver.apply_equalizer(detected_symbols)
    symbols_at_demod_input = equalized_symbols_pre_awgn.copy()

    if add_awgn_at_receiver_snr_db is not None:
        signal_power_at_rx = np.mean(np.abs(equalized_symbols_pre_awgn)**2)
        if verbose: print(f"RX: Signal power before adding AWGN = {signal_power_at_rx:.4e}")
        noise_power_target = signal_power_at_rx / (10**(add_awgn_at_receiver_snr_db / 10)) if signal_power_at_rx > 1e-12 else 1.0
        sigma_sq_per_quadrature = noise_power_target / 2.0
        if sigma_sq_per_quadrature < 0: sigma_sq_per_quadrature = 0
        std_dev_per_quadrature = np.sqrt(sigma_sq_per_quadrature)
        manual_awgn = np.random.normal(0, std_dev_per_quadrature, equalized_symbols_pre_awgn.shape) + \
                      1j * np.random.normal(0, std_dev_per_quadrature, equalized_symbols_pre_awgn.shape)
        symbols_at_demod_input = equalized_symbols_pre_awgn + manual_awgn
        if verbose: print(f"RX: Added AWGN for target SNR {add_awgn_at_receiver_snr_db:.1f} dB. "
                          f"Target Noise Power: {noise_power_target:.2e}, "
                          f"Actual Added Noise Power: {np.mean(np.abs(manual_awgn)**2):.2e}")

    rx_power_avg_after_all_noise = np.mean(np.abs(symbols_at_demod_input)**2)
    if verbose: print(f"RX: Avg power at demod input = {rx_power_avg_after_all_noise:.4f}")

    if modulation_format == "QPSK": demodulated_bits = receiver.qpsk_demodulate(symbols_at_demod_input)
    elif modulation_format == "16-QAM": demodulated_bits = receiver.qam16_demodulate(symbols_at_demod_input)

    errors = np.sum(transmitted_bits != demodulated_bits)
    total_compared_bits = transmitted_bits.size
    ber = errors / total_compared_bits if total_compared_bits > 0 else np.nan
    if verbose: print(f"BER Calculation: {errors} errors in {total_compared_bits} bits. BER = {ber:.3e}")
    return ber, tx_power_avg, rx_power_avg_after_all_noise, equalized_symbols_pre_awgn, modulated_symbols, symbols_at_demod_input

# run_ber_vs_snr_sweep (for Es/N0 sweep) remains unchanged from previous correct version
def run_ber_vs_esn0_sweep( # Renamed for clarity from generic "snr" to "esn0"
    modulation_format_sweep: str, num_bits_sim: int, s_rate_gbd: float, n_spans: int,
    span_len_km: float, atten_db_km: float, edfa_gn_db: float,
    target_esn0_db_values: np.ndarray, fixed_channel_nf_db: float, n_wdm_channels: int = 1
) -> tuple[list[float], list[float]]:
    esn0_plot_points, ber_results = [], []
    print(f"\n=== Starting BER vs. Es/N0 Sweep (Mod: {modulation_format_sweep}) ===")
    print(f"Sweep Type: Adding AWGN at Receiver (Target Es/N0: {target_esn0_db_values} dB)")
    print(f"Underlying Channel: {n_spans} spans, NF = {fixed_channel_nf_db} dB")
    print(f"Bits per point: {num_bits_sim}")

    for idx, current_esn0_db in enumerate(target_esn0_db_values):
        start_time_point = time.time()
        print(f"\nRunning point {idx+1}/{len(target_esn0_db_values)}: Target Es/N0 = {current_esn0_db:.1f} dB (Channel NF fixed at {fixed_channel_nf_db:.1f} dB)")
        result_tuple = run_simulation_single_point(
            num_bits=num_bits_sim, symbol_rate=s_rate_gbd, modulation_format=modulation_format_sweep,
            num_spans=n_spans, span_length_km=span_len_km, attenuation_db_km=atten_db_km,
            edfa_gain_db=edfa_gn_db, edfa_nf_db=fixed_channel_nf_db,
            add_awgn_at_receiver_snr_db=current_esn0_db, # This is Es/N0
            num_wdm_channels=n_wdm_channels, verbose=False
        )
        ber, _, _, _, _, _ = result_tuple
        end_time_point = time.time()
        print(f"Point {idx+1} completed. BER={ber:.3e}, Target Es/N0={current_esn0_db:.2f} dB. Time: {end_time_point - start_time_point:.2f}s")
        if not np.isnan(ber) and not np.isnan(current_esn0_db):
            esn0_plot_points.append(current_esn0_db); ber_results.append(ber)
        else: print(f"Skipping point due to invalid BER/EsN0 (BER={ber}, EsN0={current_esn0_db}).")
    print(f"=== Es/N0 Sweep Finished ===")
    return esn0_plot_points, ber_results


# --- New Sweep Function for BER vs. OSNR ---
def run_ber_vs_osnr_sweep(
    modulation_format_sweep: str,
    num_bits_sim: int,
    nf_db_sweep_values: np.ndarray, # NF values to sweep over
    fixed_num_spans: int,
    s_rate_gbd: float,
    span_len_km: float,      # Primary param for edfa_gain_db
    atten_db_km: float,      # Primary param for edfa_gain_db
    # edfa_gain_comp_loss_db is calculated from above
    n_wdm_channels: int = 1
) -> tuple[list[float], list[float]]:
    osnr_db_results = []
    ber_results = []

    edfa_gain_comp_loss_db = atten_db_km * span_len_km # Calculate EDFA gain that compensates span loss

    print(f"\n=== Starting BER vs. OSNR Sweep (Mod: {modulation_format_sweep}) ===")
    print(f"Sweeping EDFA NF from {nf_db_sweep_values[0]:.1f} dB to {nf_db_sweep_values[-1]:.1f} dB")
    print(f"Fixed parameters: Num Spans = {fixed_num_spans}, Span Length = {span_len_km} km, Atten = {atten_db_km} dB/km")
    print(f"Calculated EDFA Gain (to compensate loss) = {edfa_gain_comp_loss_db:.2f} dB")
    print(f"Bits per point: {num_bits_sim}, Symbol Rate: {s_rate_gbd} GBd")

    for idx, current_nf_db in enumerate(nf_db_sweep_values):
        start_time_point = time.time()
        print(f"\nRunning point {idx+1}/{len(nf_db_sweep_values)}: EDFA NF = {current_nf_db:.1f} dB")

        # Call run_simulation_single_point: add_awgn_at_receiver_snr_db must be None
        result_tuple = run_simulation_single_point(
            num_bits=num_bits_sim,
            symbol_rate=s_rate_gbd,
            modulation_format=modulation_format_sweep,
            num_spans=fixed_num_spans,
            span_length_km=span_len_km,
            attenuation_db_km=atten_db_km,
            edfa_gain_db=edfa_gain_comp_loss_db, # EDFAs compensate span loss
            edfa_nf_db=current_nf_db,
            add_awgn_at_receiver_snr_db=None, # Crucial: No AWGN at Rx for OSNR calc
            num_wdm_channels=n_wdm_channels,
            verbose=False
        )
        ber, tx_power_avg_sim, _, _, _, _ = result_tuple # tx_power_avg_sim should be ~1.0

        # OSNR Calculation
        tx_signal_power = 1.0 # Normalized constellation average power
        if not np.isclose(tx_power_avg_sim, 1.0, rtol=0.1): # Check if actual TX power is far from 1.0
            print(f"Warning: Simulated TX power {tx_power_avg_sim:.3f} is not close to 1.0. OSNR calc will use 1.0.")

        G_linear = 10**(edfa_gain_comp_loss_db / 10)
        NF_linear = 10**(current_nf_db / 10)

        # ASE PSD from a single EDFA, for one polarization (W/Hz)
        N_ase_psd_one_edfa_one_pol = (G_linear - 1) * NF_linear * H_PLANCK * OPTICAL_FREQ_HZ_CENTER_C_BAND

        # Total ASE PSD at receiver (sum of contributions from all EDFAs, assuming G compensates loss)
        N_ase_total_psd_at_rx_one_pol = fixed_num_spans * N_ase_psd_one_edfa_one_pol

        # Total ASE Power in the reference bandwidth (0.1 nm)
        P_ase_total_ref_bw = N_ase_total_psd_at_rx_one_pol * OSNR_REF_BW_HZ

        osnr_linear = np.inf # Default to infinite if P_ase is zero
        if P_ase_total_ref_bw > 0:
            osnr_linear = tx_signal_power / P_ase_total_ref_bw

        osnr_db = 10 * np.log10(osnr_linear) if osnr_linear > 0 else 60.0 # Represent inf OSNR as high dB

        end_time_point = time.time()
        print(f"Point {idx+1} completed. EDFA NF={current_nf_db:.1f} dB, BER={ber:.3e}, Calc. OSNR={osnr_db:.2f} dB. Time: {end_time_point - start_time_point:.2f}s")

        if not np.isnan(ber) and not np.isnan(osnr_db):
            osnr_db_results.append(osnr_db)
            ber_results.append(ber)
        else:
            print(f"Skipping point due to invalid BER/OSNR (BER={ber}, OSNR={osnr_db}).")

    print(f"=== OSNR Sweep Finished ===")
    return osnr_db_results, ber_results


if __name__ == '__main__':
    # --- Control Flags for Demos ---
    BER_ESN0_SWEEP_ENABLED = False
    CONSTELLATION_PLOT_ENABLED = False
    PSD_PLOT_ENABLED = False
    OPTICAL_FILTER_DEMO_ENABLED = False
    BER_OSNR_SWEEP_ENABLED = True # Enable the new OSNR sweep demo

    # Parameters for main execution block
    MODULATION_FORMAT_MAIN = "QPSK"

    if MODULATION_FORMAT_MAIN == "QPSK":
        target_esn0_range_main = np.arange(0.0, 15.1, 1.0) # For Es/N0 sweep
        bits_per_sym_main = BITS_PER_SYMBOL_QPSK
        constellation_snr_demo_point_main = 10.0
        # OSNR Sweep specific params for QPSK (Refined for non-zero BERs)
        NF_DB_RANGE_FOR_OSNR_MAIN = np.arange(20.0, 32.1, 2.0) # NF range: 20dB to 32dB
        FIXED_SPANS_FOR_OSNR_MAIN = 1000 # Increased spans
        NUM_BITS_FOR_OSNR_SWEEP_MAIN = 2**20 # Increased bits for better resolution
    elif MODULATION_FORMAT_MAIN == "16-QAM":
        target_esn0_range_main = np.arange(0.0, 20.1, 2.0) # For Es/N0 sweep
        bits_per_sym_main = BITS_PER_SYMBOL_QAM16
        constellation_snr_demo_point_main = 16.0
        # OSNR Sweep specific params for 16-QAM (Example, might need tuning)
        NF_DB_RANGE_FOR_OSNR_MAIN = np.arange(15.0, 27.1, 2.0)
        FIXED_SPANS_FOR_OSNR_MAIN = 1000
        NUM_BITS_FOR_OSNR_SWEEP_MAIN = 2**20
    else: raise ValueError("Unsupported MODULATION_FORMAT_MAIN")

    if BER_ESN0_SWEEP_ENABLED:
        print(f"--- Running BER vs. Es/N0 Sweep for {MODULATION_FORMAT_MAIN} ---")
        snrs_db, bers = run_ber_vs_esn0_sweep( # Call the renamed function
            modulation_format_sweep=MODULATION_FORMAT_MAIN,
            target_esn0_db_values=target_esn0_range_main,
            fixed_channel_nf_db=FIXED_EDFA_NF_DB_FOR_SWEEP,
            num_bits_sim=NUM_BITS_FOR_SWEEP, # Uses the general NUM_BITS_FOR_SWEEP
            s_rate_gbd=SYMBOL_RATE_GBD, n_spans=FIXED_NUM_SPANS_FOR_SWEEP,
            span_len_km=FIBER_SPAN_LENGTH_KM_DEFAULT, atten_db_km=ATTENUATION_DB_PER_KM_DEFAULT,
            edfa_gn_db=EDFA_GAIN_DB_DEFAULT, n_wdm_channels=NUM_CHANNELS
        )
        if snrs_db and bers:
            bers_plot = np.array(bers); snrs_plot = np.array(snrs_db)
            min_ber_for_plot = 1 / (NUM_BITS_FOR_SWEEP * bits_per_sym_main * 10)
            bers_plot[bers_plot == 0] = min_ber_for_plot
            plt.figure(figsize=(10, 6)); plt.semilogy(snrs_plot, bers_plot, marker='o', linestyle='-')
            plt.xlabel("Target Es/N0 at Receiver (dB)"); plt.ylabel("Bit Error Rate (BER)")
            plt.title(f"BER vs. Target Es/N0 for {MODULATION_FORMAT_MAIN} (AWGN @ Rx)\n"
                      f"({FIXED_NUM_SPANS_FOR_SWEEP} spans, NF {FIXED_EDFA_NF_DB_FOR_SWEEP}dB in channel, {NUM_BITS_FOR_SWEEP} bits/pt)")
            plt.grid(True, which="both", ls="-"); sensible_min_y = max(min_ber_for_plot, 1e-7); plt.ylim(bottom=sensible_min_y, top=0.5)
            plot_filename = f"ber_vs_esn0_{MODULATION_FORMAT_MAIN.lower()}.png" # Updated filename
            plt.savefig(plot_filename); print(f"\nPlot saved to {plot_filename}"); plt.close()
        else: print("\nNo valid data to plot for Es/N0 sweep.")
    else: print("\nBER vs. Es/N0 SWEEP SKIPPED.")

    # --- BER vs OSNR SWEEP ---
    if BER_OSNR_SWEEP_ENABLED:
        print(f"--- Running BER vs. OSNR Sweep for {MODULATION_FORMAT_MAIN} ---")
        osnr_results, ber_osnr_results = run_ber_vs_osnr_sweep(
            modulation_format_sweep=MODULATION_FORMAT_MAIN,
            num_bits_sim=NUM_BITS_FOR_OSNR_SWEEP_MAIN,
            nf_db_sweep_values=NF_DB_RANGE_FOR_OSNR_MAIN,
            fixed_num_spans=FIXED_SPANS_FOR_OSNR_MAIN,
            s_rate_gbd=SYMBOL_RATE_GBD,
            span_len_km=FIBER_SPAN_LENGTH_KM_DEFAULT,
            atten_db_km=ATTENUATION_DB_PER_KM_DEFAULT
            # edfa_gain_comp_loss_db is calculated inside the sweep function
        )
        if osnr_results and ber_osnr_results:
            bers_plot = np.array(ber_osnr_results)
            osnrs_plot = np.array(osnr_results)
            # Replace BER=0 with a small value for plotting on log scale
            min_ber_for_plot = 1 / (NUM_BITS_FOR_OSNR_SWEEP_MAIN * bits_per_sym_main * 10)
            bers_plot[bers_plot == 0] = min_ber_for_plot

            plt.figure(figsize=(10, 6))
            plt.semilogy(osnrs_plot, bers_plot, marker='o', linestyle='-')
            plt.xlabel("Calculated OSNR (dB in 0.1nm BW)")
            plt.ylabel("Bit Error Rate (BER)")
            plt.title(f"BER vs. OSNR for {MODULATION_FORMAT_MAIN}\n"
                      f"({FIXED_SPANS_FOR_OSNR_MAIN} spans, NF varied from {NF_DB_RANGE_FOR_OSNR_MAIN[0]} to {NF_DB_RANGE_FOR_OSNR_MAIN[-1]}dB)")
            plt.grid(True, which="both", ls="-")
            sensible_min_y = max(min_ber_for_plot, 1e-7)
            plt.ylim(bottom=sensible_min_y, top=0.5)
            plot_filename = f"ber_vs_osnr_{MODULATION_FORMAT_MAIN.lower()}.png"
            plt.savefig(plot_filename)
            print(f"\nPlot saved to {plot_filename}")
            plt.close()
        else:
            print("\nNo valid data to plot for OSNR sweep.")
    else:
        print("\nBER vs. OSNR SWEEP SKIPPED.")

    # --- Common setup for other plots (Constellation, PSD, Filter Demo) ---
    # This part now uses MODULATION_FORMAT_MAIN and constellation_snr_demo_point_main
    tx_symbols_for_plots, rx_symbols_for_plots = None, None
    # Only run if any of these demos are enabled
    if CONSTELLATION_PLOT_ENABLED or PSD_PLOT_ENABLED or OPTICAL_FILTER_DEMO_ENABLED:
        print(f"\n--- Generating Base Symbols for Constellation/PSD/Filter Demos ({MODULATION_FORMAT_MAIN}) ---")
        num_bits_for_demo_plots = min(NUM_BITS_FOR_SWEEP, 2**12)
        print(f"Running single point for plots (Mod: {MODULATION_FORMAT_MAIN}, Target SNR for AWGN: {constellation_snr_demo_point_main} dB, Bits: {num_bits_for_demo_plots})")
        # This run uses AWGN at receiver for demo SNR, not channel NF for OSNR
        _, _, _, _, tx_symbols_for_plots, rx_symbols_for_plots = run_simulation_single_point(
            num_bits=num_bits_for_demo_plots, symbol_rate=SYMBOL_RATE_GBD,
            modulation_format=MODULATION_FORMAT_MAIN, num_spans=FIXED_NUM_SPANS_FOR_SWEEP, # Using fixed EsN0 sweep params
            span_length_km=FIBER_SPAN_LENGTH_KM_DEFAULT, attenuation_db_km=ATTENUATION_DB_PER_KM_DEFAULT,
            edfa_gain_db=EDFA_GAIN_DB_DEFAULT, edfa_nf_db=FIXED_EDFA_NF_DB_FOR_SWEEP, # Fixed NF for this demo
            add_awgn_at_receiver_snr_db=constellation_snr_demo_point_main,
            num_wdm_channels=NUM_CHANNELS, verbose=True
        )

    if CONSTELLATION_PLOT_ENABLED:
        # ... (constellation plotting code unchanged, uses MODULATION_FORMAT_MAIN, constellation_snr_demo_point_main) ...
        ideal_const = get_ideal_constellation(MODULATION_FORMAT_MAIN)
        if tx_symbols_for_plots is not None and tx_symbols_for_plots.size > 0:
            plot_constellation(tx_symbols_for_plots, title=f"Transmitted {MODULATION_FORMAT_MAIN} Constellation (Demo)",
                               ideal_constellation=ideal_const, filename=f"tx_constellation_{MODULATION_FORMAT_MAIN.lower()}_demo.png")
        if rx_symbols_for_plots is not None and rx_symbols_for_plots.size > 0:
            plot_constellation(rx_symbols_for_plots, title=f"Received {MODULATION_FORMAT_MAIN} Constellation (Demo)",
                               ideal_constellation=ideal_const,
                               filename=f"rx_constellation_{MODULATION_FORMAT_MAIN.lower()}_snr{constellation_snr_demo_point_main:.0f}_demo.png",
                               snr_value=constellation_snr_demo_point_main)

    if PSD_PLOT_ENABLED:
        # ... (PSD plotting code unchanged, uses MODULATION_FORMAT_MAIN, constellation_snr_demo_point_main) ...
        symbol_rate_hz_psd = SYMBOL_RATE_GBD * 1e9
        if tx_symbols_for_plots is not None and tx_symbols_for_plots.size > 0:
            plot_psd(tx_symbols_for_plots, symbol_rate_hz_psd,
                     title=f"PSD of Transmitted {MODULATION_FORMAT_MAIN} Signal (Demo)",
                     filename=f"psd_tx_{MODULATION_FORMAT_MAIN.lower()}_demo.png")
        if rx_symbols_for_plots is not None and rx_symbols_for_plots.size > 0:
            plot_psd(rx_symbols_for_plots, symbol_rate_hz_psd,
                     title=f"PSD of Received {MODULATION_FORMAT_MAIN} Signal (Demo, Es/N0 ~ {constellation_snr_demo_point_main:.1f} dB)",
                     filename=f"psd_rx_{MODULATION_FORMAT_MAIN.lower()}_snr{constellation_snr_demo_point_main:.0f}_demo.png")

    if OPTICAL_FILTER_DEMO_ENABLED:
        # ... (Optical filter demo code unchanged, uses MODULATION_FORMAT_MAIN) ...
        print(f"\n--- Optical Filtering Demonstration ({MODULATION_FORMAT_MAIN}) ---")
        filter_demo_symbol_rate_gbd = 24.0; filter_demo_symbol_rate_hz = filter_demo_symbol_rate_gbd * 1e9
        num_bits_for_filter_demo = 2**12
        filter_demo_bits = transmitter.generate_bits(num_bits_for_filter_demo)
        if MODULATION_FORMAT_MAIN == "QPSK":
            tx_symbols_for_filter_demo = transmitter.qpsk_modulate(filter_demo_bits)
        elif MODULATION_FORMAT_MAIN == "16-QAM":
            if num_bits_for_filter_demo % 4 != 0: num_bits_for_filter_demo = (num_bits_for_filter_demo // 4) * 4; filter_demo_bits = filter_demo_bits[:num_bits_for_filter_demo]
            tx_symbols_for_filter_demo = transmitter.qam16_modulate(filter_demo_bits)

        if tx_symbols_for_filter_demo.size > 0:
            plot_psd(tx_symbols_for_filter_demo, filter_demo_symbol_rate_hz,
                     title=f"PSD of Original {MODULATION_FORMAT_MAIN} Signal (Filter Demo, {filter_demo_symbol_rate_gbd} GBd)",
                     filename=f"psd_filter_demo_original_{MODULATION_FORMAT_MAIN.lower()}.png")
            bw_factors = [0.9, 1.0, 1.2]; ideal_const_filter_demo = get_ideal_constellation(MODULATION_FORMAT_MAIN)
            for bw_factor in bw_factors:
                B_0 = bw_factor * filter_demo_symbol_rate_hz; current_sigma = B_0 / 10.0
                print(f"Applying filter: BW Factor = {bw_factor}, B_0 = {B_0/1e9:.2f} GHz, Sigma = {current_sigma/1e9:.2f} GHz")
                filtered_symbols, H_response, freq_axis_filter = optical_filter.apply_filter(
                    tx_symbols_for_filter_demo, filter_demo_symbol_rate_hz, B_0, current_sigma)
                plot_psd(filtered_symbols, filter_demo_symbol_rate_hz,
                         title=f"PSD of Filtered {MODULATION_FORMAT_MAIN} (BW Factor: {bw_factor})",
                         filename=f"psd_filter_demo_filtered_{MODULATION_FORMAT_MAIN.lower()}_bw{bw_factor:.1f}.png",
                         filter_response_hz=(H_response, freq_axis_filter))
                plot_constellation(filtered_symbols,title=f"Constellation of Filtered {MODULATION_FORMAT_MAIN} (BW Factor: {bw_factor})",
                                   ideal_constellation=ideal_const_filter_demo,
                                   filename=f"constellation_filter_demo_{MODULATION_FORMAT_MAIN.lower()}_bw{bw_factor:.1f}.png")
        else: print("Skipping filter demo plots as tx_symbols_for_filter_demo is empty.")
    print("--- End of Script ---")
