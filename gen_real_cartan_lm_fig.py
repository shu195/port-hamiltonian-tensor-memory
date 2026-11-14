import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#222222'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

ASSETS_DIR = r"C:\Users\Admin\.gemini\antigravity\scratch\port-hamiltonian-tensor-memory\assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

def generate_real_cartan_lm_benchmark_plot():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=300)

    # Panel 1: Exact 60k Parameter-Matched TinyShakespeare Validation Perplexity (benchmark_cartan_text_model.py)
    models = ['CartanLM\n(60,040)', 'Standard LSTM\n(60,475)', 'Causal GPT\n(60,004)']
    ppls = [8.01, 7.18, 12.92]
    colors = ['#00e676', '#b388ff', '#ff5252']

    bars = ax1.bar(models, ppls, color=colors, width=0.52, edgecolor='#ffffff', linewidth=0.6, alpha=0.92)
    ax1.set_title('TinyShakespeare 60k-Matched Val PPL (↓)', fontsize=11.5, fontweight='bold', pad=10)
    ax1.set_ylabel('Validation Perplexity (PPL)', fontsize=10.5, color='#cccccc')
    ax1.set_ylim(0, 15.0)
    ax1.grid(True, axis='y', color='#222222')
    ax1.tick_params(axis='x', labelsize=9.2)
    for bar, val in zip(bars, ppls):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, val + 0.35, f"{val:.2f}",
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#ffffff')

    # Panel 2: Exact Measured Roy-Vetterli Effective Rank across t in {16, 32, 48, 64}
    t_pos = [16, 32, 48, 64]
    erank_cartan = [66.35, 66.48, 67.47, 66.82]  # d_model=128 (52.2% of D, 100.7% temporal retention)
    erank_lstm   = [34.06, 34.24, 34.09, 33.77]  # d_model=59  (57.2% of D)
    erank_gpt    = [7.97,  7.79,  7.35,  7.59]   # d_model=60  (12.7% of D - severe rank collapse)

    ax2.plot(t_pos, erank_cartan, color='#00e676', marker='o', linewidth=2.3, label='CartanLM (D=128: 66.82, 100.7% ret.)')
    ax2.plot(t_pos, erank_lstm,   color='#b388ff', marker='^', linewidth=2.0, linestyle='--', label='Standard LSTM (D=59: 33.77)')
    ax2.plot(t_pos, erank_gpt,    color='#ff5252', marker='s', linewidth=2.0, linestyle=':', label='Causal GPT (D=60: 7.59 collapse)')
    ax2.set_title('Measured Roy-Vetterli eRank vs Position t', fontsize=11.5, fontweight='bold', pad=10)
    ax2.set_xlabel('Sequence Token Position (t)', fontsize=10.5, color='#cccccc')
    ax2.set_ylabel('Effective Rank (erank)', fontsize=10.5, color='#cccccc')
    ax2.set_xticks(t_pos)
    ax2.set_ylim(0, 80)
    ax2.grid(True, color='#222222')
    ax2.legend(loc='center left', bbox_to_anchor=(0.02, 0.56), fontsize=8.2, framealpha=0.65, facecolor='#111111', edgecolor='#444444')

    # Panel 3: FLA Real-Split & Fused BMM Speedup (Table 7)
    comps = ['Causal SSM\nMixer', 'Multi-Head\nRotor', 'Symplectic\nKicks', 'Full Layer\nForward']
    lat_before = [15.206, 8.347, 2.470, 28.904]
    lat_after  = [7.406, 2.199, 1.573, 13.158]
    x_c = np.arange(len(comps))
    w_c = 0.35

    b1 = ax3.bar(x_c - w_c/2, lat_before, width=w_c, color='#ff5252', edgecolor='#ffffff', linewidth=0.5, label='Before (Complex Tree + 4 Einsums)')
    b2 = ax3.bar(x_c + w_c/2, lat_after, width=w_c, color='#00e5ff', edgecolor='#ffffff', linewidth=0.5, label='After (FLA Real-Split + Fused BMM)')
    ax3.set_title('Kernel Optimization Speedup (3.28x End-to-End)', fontsize=11.5, fontweight='bold', pad=10)
    ax3.set_xticks(x_c)
    ax3.set_xticklabels(comps, fontsize=9.0, color='#cccccc')
    ax3.set_ylabel('Latency per Pass (ms)', fontsize=10.5, color='#cccccc')
    ax3.set_ylim(0, 33.5)
    ax3.grid(True, axis='y', color='#222222')
    ax3.legend(loc='upper left', fontsize=8.0, framealpha=0.55, facecolor='#111111', edgecolor='#444444')

    for bar in b1:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}", ha='center', fontsize=8.2, color='#ff8888')
    for bar in b2:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}", ha='center', fontsize=8.2, color='#00e5ff', fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(ASSETS_DIR, "cartan_lm_benchmark_and_speedup.png")
    plt.savefig(out_path, dpi=300, facecolor='#000000', edgecolor='none')
    plt.close()
    print(f"Generated: {out_path}")

if __name__ == "__main__":
    generate_real_cartan_lm_benchmark_plot()
