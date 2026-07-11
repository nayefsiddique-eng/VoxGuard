import os
import glob
import hashlib
import subprocess
import shutil
import numpy as np
import soundfile as sf
import scipy.signal as signal

def get_ffmpeg_path():
    """Locates the ffmpeg executable on the system, including winget local directories."""
    # 1. Try global path
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return "ffmpeg"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # 2. Try WinGet package directories in Local AppData
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
        winget_pkg_dir = os.path.join(local_appdata, "Microsoft", "WinGet", "Packages")
        if os.path.exists(winget_pkg_dir):
            for root, dirs, files in os.walk(winget_pkg_dir):
                if "ffmpeg.exe" in files:
                    return os.path.join(root, "ffmpeg.exe")
    return None

def is_ffmpeg_available():
    """Checks if ffmpeg is available in system or winget path."""
    return get_ffmpeg_path() is not None

def degrade_ffmpeg(input_path, output_path, codec="amr"):
    """
    Degrades audio using ffmpeg.
    - AMR-NB: Re-sample to 8kHz, mono, AMR encoding, and convert back to 16kHz WAV.
    - Opus: Re-encode to low-bitrate Opus (e.g., 6kbps) and convert back to 16kHz WAV.
    """
    if codec == "none":
        shutil.copy(input_path, output_path)
        return True

    ffmpeg_cmd = get_ffmpeg_path()
    if not ffmpeg_cmd:
        return False
        
    temp_encoded = output_path + ".temp"
    try:
        if codec == "amr":
            # Encode to AMR narrowband (requires 8000Hz sample rate)
            subprocess.run([
                ffmpeg_cmd, "-y", "-i", input_path,
                "-ar", "8000", "-ac", "1", "-ab", "12.2k",
                temp_encoded + ".amr"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Decode back to 16kHz WAV for feature extraction consistency
            subprocess.run([
                ffmpeg_cmd, "-y", "-i", temp_encoded + ".amr",
                "-ar", "16000", output_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if os.path.exists(temp_encoded + ".amr"):
                os.remove(temp_encoded + ".amr")
                
        elif codec == "opus":
            # Encode to low-bitrate Opus (6kbps) simulating low bandwidth VoIP
            subprocess.run([
                ffmpeg_cmd, "-y", "-i", input_path,
                "-c:a", "libopus", "-b:a", "6k", "-vbr", "on",
                temp_encoded + ".opus"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Decode back to 16kHz WAV
            subprocess.run([
                ffmpeg_cmd, "-y", "-i", temp_encoded + ".opus",
                "-ar", "16000", output_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if os.path.exists(temp_encoded + ".opus"):
                os.remove(temp_encoded + ".opus")
                
        elif codec == "gsm":
            # Encode to GSM full rate (requires 8000Hz sample rate, mono)
            subprocess.run([
                ffmpeg_cmd, "-y", "-i", input_path,
                "-codec:a", "gsm", "-ar", "8000", "-ac", "1",
                temp_encoded + ".gsm"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Decode back to 16kHz WAV
            subprocess.run([
                ffmpeg_cmd, "-y", "-i", temp_encoded + ".gsm",
                "-ar", "16000", output_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if os.path.exists(temp_encoded + ".gsm"):
                os.remove(temp_encoded + ".gsm")
        return True
    except Exception as e:
        print(f"  [ffmpeg degradation failed: {e}] Falling back to python degradation.")
        return False

def degrade_python_fallback(input_path, output_path, codec="amr"):
    """
    Python fallback simulator using numpy and scipy to approximate codec effects.
    - AMR-NB approximation: bandpass filter (300Hz-3400Hz) + downsample to 8kHz + decimation + 8-bit quantization.
    - Opus approximation: downsample to 12kHz + sub-band compression + slight high frequency attenuation.
    """
    if codec == "none":
        shutil.copy(input_path, output_path)
        return

    data, sr = sf.read(input_path)
    
    if codec == "amr" or codec == "gsm":
        # Standard telephone line filter (300Hz - 3400Hz)
        nyq = 0.5 * sr
        low = 300 / nyq
        high = 3400 / nyq
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, data)
        
        # Quantize audio to simulate aggressive low bitrate compaction
        steps = 32 # 5-bit depth approximation
        quantized = np.round(filtered * steps) / steps
        degraded = quantized
    else:
        # Opus approximation: high-frequency roll-off (lowpass filter at 6000Hz)
        nyq = 0.5 * sr
        high = 6000 / nyq
        b, a = signal.butter(4, high, btype='low')
        degraded = signal.filtfilt(b, a, data)
        
        # Add subtle compression artifact noise
        noise = np.random.normal(0, 0.005, len(degraded))
        degraded = degraded + noise
        
    sf.write(output_path, degraded, sr)

def apply_network_degradations(audio_path, output_path, packet_loss_rate=0.05, jitter_ms=10):
    """
    Simulates packet loss and jitter on WAV audio.
    - Packet loss: zero out audio blocks of 20ms at the specified rate.
    - Jitter: introduces micro-silences or overlaps to simulate jitter buffer delay spikes.
    """
    data, sr = sf.read(audio_path)
    length = len(data)
    
    # 20ms block size
    block_size = int(sr * 0.02) 
    num_blocks = length // block_size
    
    modified_data = data.copy()
    
    # Simulate packet loss
    if packet_loss_rate > 0:
        loss_mask = np.random.rand(num_blocks) < packet_loss_rate
        for i in range(num_blocks):
            if loss_mask[i]:
                start = i * block_size
                end = start + block_size
                modified_data[start:end] = 0.0 # Packet dropped
                
    # Simulate jitter (delay variation / buffer reordering / gaps)
    if jitter_ms > 0:
        # We simulate jitter by inserting small variable silent gaps (jitter buffer underflow)
        jitter_samples = int(sr * (jitter_ms / 1000.0))
        if jitter_samples > 0:
            jittered_blocks = []
            for i in range(num_blocks):
                start = i * block_size
                end = start + block_size
                block = modified_data[start:end]
                
                # Randomly inject a jitter gap (10% chance per block)
                if np.random.rand() < 0.1:
                    gap_size = np.random.randint(1, jitter_samples)
                    # Insert silence before the block
                    jittered_blocks.append(np.zeros(gap_size))
                jittered_blocks.append(block)
            
            # Reassemble and trim to original length
            jittered_audio = np.concatenate(jittered_blocks)
            if len(jittered_audio) > length:
                modified_data = jittered_audio[:length]
            else:
                modified_data = np.pad(jittered_audio, (0, length - len(jittered_audio)))
                
    sf.write(output_path, modified_data, sr)

_PRINT_COUNTER = 0

def process_file(input_path, output_dir, codec="amr", packet_loss=0.05, jitter=10):
    """Degrades a single audio file and applies network channel effects."""
    global _PRINT_COUNTER
    base_name = os.path.basename(input_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Encode degradation parameters into the filename to prevent caching collisions
    degraded_file = os.path.join(output_dir, f"degraded_{codec}_L{packet_loss}_J{jitter}_{base_name}")
    
    # Fetch input audio stats
    try:
        in_data, in_sr = sf.read(input_path)
        in_duration = len(in_data) / in_sr
        in_variance = float(np.var(in_data))
    except Exception as e:
        in_duration = 0.0
        in_variance = 0.0
    
    # 1. Apply codec compression
    ffmpeg_ok = False
    if is_ffmpeg_available():
        ffmpeg_ok = degrade_ffmpeg(input_path, degraded_file, codec)
        
    if not ffmpeg_ok:
        degrade_python_fallback(input_path, degraded_file, codec)
        
    # 2. Apply channel network degradation on top of codec compression
    apply_network_degradations(degraded_file, degraded_file, packet_loss_rate=packet_loss, jitter_ms=jitter)
    
    # Fetch output audio stats
    try:
        out_data, out_sr = sf.read(degraded_file)
        out_duration = len(out_data) / out_sr
        out_variance = float(np.var(out_data))
    except Exception as e:
        out_duration = 0.0
        out_variance = 0.0

    # Print diagnostics for the first 3 processed files
    if _PRINT_COUNTER < 3:
        print(f"\n[DEGRADATION DIAGNOSTIC #{_PRINT_COUNTER}] File: {base_name}")
        print(f"  Codec: {codec} | Loss Rate: {packet_loss} | Jitter: {jitter}ms")
        print(f"  Before: Duration = {in_duration:.3f}s, Spectral Checksum (Var) = {in_variance:.6f}")
        print(f"  After : Duration = {out_duration:.3f}s, Spectral Checksum (Var) = {out_variance:.6f}")
        _PRINT_COUNTER += 1
        
    return degraded_file

def main():
    placeholder_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/placeholder"))
    degraded_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/degraded"))
    
    if not os.path.exists(placeholder_dir):
        print(f"Error: Placeholder directory does not exist: {placeholder_dir}")
        return
        
    print(f"Processing placeholder audios from: {placeholder_dir}")
    print(f"Saving degraded results in: {degraded_dir}")
    
    for filename in os.listdir(placeholder_dir):
        if filename.endswith(".wav"):
            input_path = os.path.join(placeholder_dir, filename)
            
            # Degrade through AMR + Network Loss
            amr_out = process_file(input_path, degraded_dir, codec="amr", packet_loss=0.08, jitter=15)
            print(f"  Degraded AMR-NB: {os.path.basename(amr_out)}")
            
            # Degrade through Opus + Network Loss
            opus_out = process_file(input_path, degraded_dir, codec="opus", packet_loss=0.04, jitter=8)
            print(f"  Degraded Opus: {os.path.basename(opus_out)}")

if __name__ == "__main__":
    main()
