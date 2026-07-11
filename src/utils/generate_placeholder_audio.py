import os
import numpy as np
import soundfile as sf

def generate_sine_wave(frequency, duration, sample_rate=16000, amplitude=0.5):
    """Generates a simple sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * t)

def generate_class_a_clean(filename, duration=3.0, sample_rate=16000):
    """
    Class A: Clean, voice-like synthetic tone (base sine wave with subtle frequency modulation).
    PLACEHOLDER: Replace with real ASVspoof/cloned sample here.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Subtle frequency modulation around 220Hz (typical pitch)
    fm_frequency = 220 + 20 * np.sin(2 * np.pi * 2.0 * t)
    audio = 0.5 * np.sin(2 * np.pi * fm_frequency * t)
    
    # Save audio
    sf.write(filename, audio, sample_rate)

def generate_class_b_fake(filename, duration=3.0, sample_rate=16000):
    """
    Class B: Distorted synthetic tone simulating a fake/cloned signature
    (base sine wave with aggressive pitch shifts, robotic modulation, or noise).
    PLACEHOLDER: Replace with real ASVspoof/cloned sample here.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Aggressive frequency modulation simulating robotic pitch artifact
    fm_frequency = 220 + 80 * np.sin(2 * np.pi * 15.0 * t)
    # Add robotic amplitude modulation (tremolo)
    am = 0.5 + 0.5 * np.sin(2 * np.pi * 20.0 * t)
    audio = 0.5 * np.sin(2 * np.pi * fm_frequency * t) * am
    
    # Add a bit of white noise
    noise = np.random.normal(0, 0.05, len(t))
    audio = audio + noise
    # Clip to valid range
    audio = np.clip(audio, -1.0, 1.0)
    
    # Save audio
    sf.write(filename, audio, sample_rate)

def main():
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/placeholder"))
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating synthetic placeholder files in: {output_dir}")
    
    # Generate 5 clean (Class A) samples
    for i in range(5):
        filepath = os.path.join(output_dir, f"class_a_clean_{i}.wav")
        generate_class_a_clean(filepath)
        print(f"  Created: {filepath}")
        
    # Generate 5 fake (Class B) samples
    for i in range(5):
        filepath = os.path.join(output_dir, f"class_b_fake_{i}.wav")
        generate_class_b_fake(filepath)
        print(f"  Created: {filepath}")

    print("Placeholder audio generation complete.")

if __name__ == "__main__":
    main()
