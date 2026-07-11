import os
import numpy as np
import matplotlib.pyplot as plt

def generate_and_save_plot():
    # Set seed for reproducible point replication matching the fuser dataset
    np.random.seed(42)
    
    # 1. Re-generate fuser training scores dataset
    # Authentic cases (Label = 1)
    auth_passive = np.random.uniform(0.70, 0.99, 100)
    auth_challenge = np.random.uniform(0.75, 0.98, 100)
    
    # Spoofed cases (Label = 0)
    spoof_passive = np.random.uniform(0.01, 0.35, 100)
    spoof_challenge = np.random.uniform(0.05, 0.45, 100)
    
    # Evasion cases (Label = 0)
    evade_passive = np.random.uniform(0.01, 0.35, 50)
    evade_challenge = np.random.uniform(0.70, 0.95, 50)
    
    # Injection cases (Label = 0)
    inject_passive = np.random.uniform(0.70, 0.95, 50)
    inject_challenge = np.random.uniform(0.05, 0.40, 50)
    
    # Setup plot styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Plot points
    ax.scatter(auth_passive, auth_challenge, color='#10b981', alpha=0.8, edgecolors='none', s=60, label='Authentic Call (Human)')
    ax.scatter(spoof_passive, spoof_challenge, color='#ef4444', alpha=0.8, edgecolors='none', s=60, label='Spoofed Call (Fake)')
    ax.scatter(evade_passive, evade_challenge, color='#8b5cf6', alpha=0.8, edgecolors='none', s=60, label='Evasion Attempt (Fake)')
    ax.scatter(inject_passive, inject_challenge, color='#f59e0b', alpha=0.8, edgecolors='none', s=60, label='Injection Attempt (Fake)')
    
    # Draw Decision Boundary: 4.1575 * x + 4.6424 * y - 6.2060 = 0 -> y = (6.2060 - 4.1575 * x) / 4.6424
    x_vals = np.linspace(0.0, 1.0, 100)
    y_vals = (6.2060 - 4.1575 * x_vals) / 4.6424
    
    # Clip values to visual scale
    ax.plot(x_vals, y_vals, color='#06b6d4', linestyle='--', linewidth=2.5, label='Fuser Decision Boundary')
    
    # Label and Title
    ax.set_title("GOTCHA Score Fusion Boundary Visualization", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Passive Call Authenticity Score", fontsize=11, fontweight='semibold', labelpad=10)
    ax.set_ylabel("Challenge Adherence Scorer Score", fontsize=11, fontweight='semibold', labelpad=10)
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Save Image
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../docs"))
    os.makedirs(docs_dir, exist_ok=True)
    plot_path = os.path.join(docs_dir, "fuser_scatter_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated and saved fuser boundary plot to: {plot_path}")

if __name__ == "__main__":
    generate_and_save_plot()
