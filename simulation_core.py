import numpy as np
import transmitter
import optical_channel
import receiver
import matplotlib.pyplot as plt
import time

# --- System Parameters ---
NUM_BITS_TO_SIMULATE_DEFAULT = 2**18  # Default for single point, overridden for sweep
MODULATION_FORMAT = "QPSK"
BITS_PER_SYMBOL = 2
SYMBOL_RATE_GBD = 32.0
NUM_CHANNELS = 1

# Fiber Channel Parameters (can be fixed for AWGN-at-Rx controlled SNR sweep)
FIBER_SPAN_LENGTH_KM_DEFAULT = 80.0
ATTENUATION_DB_PER_KM_DEFAULT = 0.2
# EDFA gain is set to compensate for the loss of one fiber span
EDFA_GAIN_DB_DEFAULT = FIBER_SPAN_LENGTH_KM_DEFAULT * ATTENUATION_DB_PER_KM_DEFAULT

# --- Parameters for AWGN-at-Receiver SNR Sweep ---
# These will be used in the __main__ block for the new sweep type
FIXED_NUM_SPANS_FOR_SWEEP = 10
FIXED_EDFA_NF_DB_FOR_SWEEP = 5.0
TARGET_SNR_DB_RANGE = np.arange(0.0, 15.1, 1.0) # Target Es/N0 for QPSK
NUM_BITS_FOR_SWEEP = 2**18 # 262,144 bits


def run_simulation_single_point(
    num_bits: int,
    symbol_rate: float,
    # Channel parameters (will be fixed during AWGN-at-Rx SNR sweep)
    num_spans: int,
    span_length_km: float,
    attenuation_db_km: float,
    edfa_gain_db: float,
    edfa_nf_db: float,
    # New parameter for direct AWGN addition at receiver
    add_awgn_at_receiver_snr_db: float = None,
    num_wdm_channels: int = 1,
    verbose: bool = True
) -> tuple[float, float, float, np.ndarray]:
    """
    Runs a single point simulation of the optical link.
    Can now add specified AWGN at the receiver for direct SNR control.
    """
    if verbose:
        log_prefix = f"--- Running simulation (Bits: {num_bits}"
        if add_awgn_at_receiver_snr_db is not None:
            log_prefix += f", Target Rx SNR: {add_awgn_at_receiver_snr_db:.1f} dB"
        else:
            log_prefix += f", Channel NF: {edfa_nf_db:.1f} dB"
        log_prefix += ") ---"
        print(log_prefix)

    # --- TX Stage ---
    if verbose: print(f"TX: Generating {num_bits} bits...")
    transmitted_bits = transmitter.generate_bits(num_bits)
    if verbose: print(f"TX: Modulating bits to {MODULATION_FORMAT} symbols...")
    modulated_symbols = transmitter.qpsk_modulate(transmitted_bits)
    tx_power_avg = np.mean(np.abs(modulated_symbols)**2)
    if verbose: print(f"TX: Average signal power = {tx_power_avg:.4f}")

    # --- Channel Stage (Optical Noise) ---
    # This noise is always present, based on channel params
    if verbose: print(f"Channel: Propagating through {num_spans} spans ({edfa_nf_db=:.1f}dB)...")
    symbols_after_channel = optical_channel.propagate_through_spans(
        signal=modulated_symbols,
        num_spans=num_spans,
        fiber_length_km=span_length_km,
        attenuation_db_per_km=attenuation_db_km,
        edfa_gain_db=edfa_gain_db,
        edfa_noise_figure_db=edfa_nf_db,
        symbol_rate_gbd=symbol_rate,
        num_channels=num_wdm_channels
    )
    if verbose: print("Channel: Propagation complete.")

    # --- RX Stage (Pre-Demodulation) ---
    if verbose: print("RX: Performing coherent detection (placeholder)...")
    detected_symbols = receiver.coherent_detect(symbols_after_channel)
    if verbose: print("RX: Applying equalizer (placeholder)...")
    equalized_symbols = receiver.apply_equalizer(detected_symbols)
    
    # Store symbols before potential addition of controlled AWGN
    symbols_before_manual_awgn = equalized_symbols.copy() 
    
    # --- Optional: Add AWGN at Receiver for Direct SNR Control ---
    if add_awgn_at_receiver_snr_db is not None:
        signal_power_at_rx = np.mean(np.abs(equalized_symbols)**2)
        if verbose: print(f"RX: Signal power before adding AWGN = {signal_power_at_rx:.4e}")

        if signal_power_at_rx <= 0:
            print("Warning: Signal power at receiver is zero or negative. Cannot add AWGN for target SNR.")
            # In this case, equalized_symbols will proceed to demodulation without additional AWGN
        else:
            snr_linear = 10**(add_awgn_at_receiver_snr_db / 10)
            noise_power_target = signal_power_at_rx / snr_linear
            sigma_sq_per_quadrature = noise_power_target / 2.0
            
            if sigma_sq_per_quadrature < 0: # Should not happen if signal_power_at_rx > 0
                 print(f"Warning: Calculated noise variance per quadrature is negative ({sigma_sq_per_quadrature}). Skipping AWGN addition.")
            else:
                std_dev_per_quadrature = np.sqrt(sigma_sq_per_quadrature)
                
                noise_real = np.random.normal(0, std_dev_per_quadrature, equalized_symbols.shape)
                noise_imag = np.random.normal(0, std_dev_per_quadrature, equalized_symbols.shape)
                manual_awgn = noise_real + 1j * noise_imag
                
                equalized_symbols = equalized_symbols + manual_awgn
                if verbose: print(f"RX: Added AWGN for target SNR {add_awgn_at_receiver_snr_db:.1f} dB. "
                                  f"Target Noise Power: {noise_power_target:.2e}, "
                                  f"Actual Added Noise Power: {np.mean(np.abs(manual_awgn)**2):.2e}")
    
    rx_power_avg_after_all_noise = np.mean(np.abs(equalized_symbols)**2)
    if verbose: print(f"RX: Average signal power after all noise (at demod input) = {rx_power_avg_after_all_noise:.4f}")
    
    if verbose: print(f"RX: Demodulating {MODULATION_FORMAT} symbols to bits...")
    demodulated_bits = receiver.qpsk_demodulate(equalized_symbols)
    if verbose: print("RX: Demodulation complete.")

    # --- BER Calculation ---
    errors = np.sum(transmitted_bits != demodulated_bits)
    total_compared_bits = transmitted_bits.size
    ber = errors / total_compared_bits if total_compared_bits > 0 else np.nan
    if verbose: print(f"BER Calculation: {errors} errors in {total_compared_bits} bits. BER = {ber:.3e}")
    
    # Return tx_power_avg (original signal power), rx_power_avg_after_all_noise, and symbols_before_manual_awgn for other SNR estimations if needed
    return ber, tx_power_avg, rx_power_avg_after_all_noise, symbols_before_manual_awgn


def run_ber_vs_snr_sweep(
    # Required parameters (no defaults)
    sweep_type: str, 
    num_bits_sim: int,
    s_rate_gbd: float,
    n_spans: int, 
    span_len_km: float,
    atten_db_km: float,
    edfa_gn_db: float,
    # Optional parameters (with defaults)
    # Params for channel_nf sweep
    nf_db_range: np.ndarray = None,
    # Params for receiver_awgn sweep
    target_snr_db_values: np.ndarray = None,
    fixed_channel_nf_db: float = None, # Used when sweep_type is "receiver_awgn"
    n_wdm_channels: int = 1
) -> tuple[list[float], list[float]]:
    """
    Runs a BER vs SNR sweep.
    If sweep_type is "channel_nf": varies EDFA Noise Figure. SNR is estimated.
    If sweep_type is "receiver_awgn": varies target SNR by adding AWGN at Rx. SNR is the target SNR.
    """
    snr_plot_points = [] # SNR values for the x-axis of the plot
    ber_results = []
    
    if sweep_type == "channel_nf":
        iterator_values = nf_db_range
        print(f"\n=== Starting BER vs. SNR Sweep (Varying Channel NF) ===")
        print(f"NF range: {nf_db_range} dB")
    elif sweep_type == "receiver_awgn":
        iterator_values = target_snr_db_values
        print(f"\n=== Starting BER vs. SNR Sweep (Adding AWGN at Receiver) ===")
        print(f"Target SNR range: {target_snr_db_values} dB")
        print(f"Underlying Channel: {n_spans} spans, NF = {fixed_channel_nf_db} dB")
    else:
        raise ValueError("Invalid sweep_type. Must be 'channel_nf' or 'receiver_awgn'.")

    print(f"Bits per point: {num_bits_sim}")

    for idx, current_val in enumerate(iterator_values):
        start_time_point = time.time()
        
        awgn_snr_param = None
        channel_nf_param = None

        if sweep_type == "channel_nf":
            channel_nf_param = current_val
            print(f"\nRunning point {idx+1}/{len(iterator_values)}: Channel NF = {channel_nf_param:.1f} dB")
        elif sweep_type == "receiver_awgn":
            awgn_snr_param = current_val
            channel_nf_param = fixed_channel_nf_db # Use the fixed channel NF
            print(f"\nRunning point {idx+1}/{len(iterator_values)}: Target Rx SNR = {awgn_snr_param:.1f} dB (Channel NF fixed at {channel_nf_param:.1f} dB)")

        ber, tx_power, rx_power_final, symbols_pre_awgn = run_simulation_single_point(
            num_bits=num_bits_sim,
            symbol_rate=s_rate_gbd,
            num_spans=n_spans,
            span_length_km=span_len_km,
            attenuation_db_km=atten_db_km,
            edfa_gain_db=edfa_gn_db,
            edfa_nf_db=channel_nf_param, # This is varied for "channel_nf" or fixed for "receiver_awgn"
            add_awgn_at_receiver_snr_db=awgn_snr_param, # This is varied for "receiver_awgn"
            num_wdm_channels=n_wdm_channels,
            verbose=False # Keep single point simulation quiet during sweep
        )
        
        current_snr_for_plot = np.nan
        if sweep_type == "channel_nf":
            # SNR Estimation based on channel noise (original method)
            # Uses power of symbols *before* any manual AWGN (though none is added in this mode)
            signal_power_for_snr_est = np.mean(np.abs(symbols_pre_awgn)**2) # Should be close to tx_power if gain=loss
            if signal_power_for_snr_est > 0 and rx_power_final > signal_power_for_snr_est: # rx_power_final includes channel noise
                noise_power_estimated = rx_power_final - signal_power_for_snr_est
                if noise_power_estimated > 0:
                    snr_linear_estimated = signal_power_for_snr_est / noise_power_estimated
                    current_snr_for_plot = 10 * np.log10(snr_linear_estimated)
                else:
                    current_snr_for_plot = 60 # Effectively infinite
            else:
                 current_snr_for_plot = 60 # Problematic estimation or very high SNR
            print_snr_val = f"Est. SNR={current_snr_for_plot:.2f} dB"
        elif sweep_type == "receiver_awgn":
            current_snr_for_plot = awgn_snr_param # Use the target SNR directly for plotting
            print_snr_val = f"Target SNR={current_snr_for_plot:.2f} dB"

        end_time_point = time.time()
        print(f"Point {idx+1} completed. BER={ber:.3e}, {print_snr_val}. Time: {end_time_point - start_time_point:.2f}s")

        if not np.isnan(ber) and not np.isnan(current_snr_for_plot):
            snr_plot_points.append(current_snr_for_plot)
            ber_results.append(ber)
        else:
            print(f"Skipping point due to invalid BER/SNR.")

    print(f"=== {sweep_type.upper()} Sweep Finished ===")
    return snr_plot_points, ber_results


if __name__ == '__main__':
    print("=== Optical Link Simulation - BER vs. SNR Sweep (AWGN @ Receiver) ===")
    print(f"Sweep Parameters: Num Bits={NUM_BITS_FOR_SWEEP}, Symbol Rate={SYMBOL_RATE_GBD} GBd")
    print(f"Underlying Channel: {FIXED_NUM_SPANS_FOR_SWEEP} spans, "
          f"{FIBER_SPAN_LENGTH_KM_DEFAULT} km/span, "
          f"{ATTENUATION_DB_PER_KM_DEFAULT} dB/km loss")
    print(f"Underlying EDFA: Gain={EDFA_GAIN_DB_DEFAULT:.2f} dB, NF={FIXED_EDFA_NF_DB_FOR_SWEEP} dB")
    print(f"Target SNR Range for sweep (AWGN @ Rx): {TARGET_SNR_DB_RANGE} dB")
    print("------------------------------------")

    total_start_time = time.time()

    snrs_db, bers = run_ber_vs_snr_sweep(
        sweep_type="receiver_awgn",
        target_snr_db_values=TARGET_SNR_DB_RANGE,
        fixed_channel_nf_db=FIXED_EDFA_NF_DB_FOR_SWEEP,
        num_bits_sim=NUM_BITS_FOR_SWEEP,
        s_rate_gbd=SYMBOL_RATE_GBD,
        n_spans=FIXED_NUM_SPANS_FOR_SWEEP, # Fixed number of spans for this sweep type
        span_len_km=FIBER_SPAN_LENGTH_KM_DEFAULT,
        atten_db_km=ATTENUATION_DB_PER_KM_DEFAULT,
        edfa_gn_db=EDFA_GAIN_DB_DEFAULT,
        n_wdm_channels=NUM_CHANNELS
    )

    total_end_time = time.time()
    print(f"\nTotal sweep duration: {(total_end_time - total_start_time) / 60:.2f} minutes")

    if snrs_db and bers:
        # Filter out BER=0 points for log plot, or replace with a small value if desired
        # For now, let matplotlib handle it or use ylim.
        # For plotting, it's good to sort by SNR if not already sorted
        bers_plot = np.array(bers)
        snrs_plot = np.array(snrs_db)

        # Replace BER=0 with a small value for plotting on log scale, e.g., min_ber / 10 or a fixed small number
        # Ensure using the correct variable name: NUM_BITS_FOR_SWEEP
        min_ber_for_plot = 1 / (NUM_BITS_FOR_SWEEP * BITS_PER_SYMBOL * 10) # A bit below 1 error
        bers_plot[bers_plot == 0] = min_ber_for_plot 


        plt.figure(figsize=(10, 6))
        plt.semilogy(snrs_plot, bers_plot, marker='o', linestyle='-')
        plt.xlabel("Target Es/N0 at Receiver (dB)")
        plt.ylabel("Bit Error Rate (BER)")
        plt.title(f"BER vs. Target Es/N0 for QPSK (AWGN @ Rx)\n"
                  f"({FIXED_NUM_SPANS_FOR_SWEEP} spans, NF {FIXED_EDFA_NF_DB_FOR_SWEEP}dB in channel, {NUM_BITS_FOR_SWEEP} bits/pt)")
        plt.grid(True, which="both", ls="-")
        
        # Set y-axis limits to ensure BER=0 points (now min_ber_for_plot) are visible if they occur
        # And to prevent y-axis from going too high if some BERs are large.
        sensible_min_y = max(min_ber_for_plot, 1e-7) # Don't go below 1e-7 or min_ber_for_plot
        plt.ylim(bottom=sensible_min_y, top=0.5) # Max BER is 0.5
        
        plot_filename = "ber_vs_snr.png"
        plt.savefig(plot_filename)
        print(f"\nPlot saved to {plot_filename}")
    else:
        print("\nNo valid data to plot. SNR or BER list is empty.")

    print("--- End of Script ---")
