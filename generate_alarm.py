import numpy as np
import wave
import struct

def generate_alarm_sound(filename="alert.wav", duration=3, sample_rate=44100):
    """
    Generate a 3-second alarm sound with alternating high and low frequency tones
    """
    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create alternating frequency pattern (like a siren)
    frequency1 = 800  # Hz
    frequency2 = 1200  # Hz
    
    # Create the waveform - alternating between two frequencies every 0.5 seconds
    wave_data = np.zeros_like(t)
    
    for i, time_val in enumerate(t):
        # Alternate frequency every 0.5 seconds
        if int(time_val * 2) % 2 == 0:
            wave_data[i] = np.sin(2 * np.pi * frequency1 * time_val)
        else:
            wave_data[i] = np.sin(2 * np.pi * frequency2 * time_val)
    
    # Apply envelope to prevent clicking
    envelope = np.ones_like(wave_data)
    fade_samples = int(0.1 * sample_rate)  # 0.1 second fade
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    
    wave_data *= envelope
    
    # Normalize and convert to 16-bit integers
    wave_data = np.int16(wave_data * 32767 * 0.5)  # 50% volume
    
    # Save as WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(wave_data.tobytes())
    
    print(f"Alarm sound saved as {filename}")

if __name__ == "__main__":
    generate_alarm_sound()