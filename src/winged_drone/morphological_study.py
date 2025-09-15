# v11: Niente α_{L=0}; gruppi = Aerodynamic surfaces + Stability-related
# Due figure: (i) solo Velocity & Efficiency, (ii) tutte e tre
# Estetica invariata: bande colorate, legenda boxed, label gruppi in basso, x-ticks piccoli.

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

# === Parametri I/O ===
CSV_PATH = "/home/andrea/Documents/genesis/Genesis/src/winged_drone/deap_temp.csv"
OUT_DIR = Path("ea_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === Utilità parsing ===
NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

def parse_chromosome(cell):
    if pd.isna(cell):
        return None
    s = str(cell).strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    try:
        return [float(x) for x in s.split(",")]
    except Exception:
        nums = re.findall(NUM_RE, str(cell))
        return [float(x) for x in nums] if nums else None

def parse_exp_name_numbers(cell):
    if pd.isna(cell):
        return None
    nums = re.findall(NUM_RE, str(cell))
    return [float(x) for x in nums] if nums else None

# === Morphology builder ===
def build_morphology_df(df):
    chromos = df["chromosome"].apply(parse_chromosome)
    expnums = df["exp_name"].apply(parse_exp_name_numbers)

    cols = [
        "wing_span","wing_chord","wing_attach_x","fus_cg_x","fus_length",
        "elevator_span","elevator_chord","rudder_span","rudder_chord",
        "density_scale","dihedral_deg","prop_radius","hinge_le_ratio",
        "sweep_multi","twist_multi","cl_alpha_2d","alpha0_2d"
    ]
    morph = pd.DataFrame(index=df.index, columns=cols, dtype=float)

    for i in df.index:
        en = expnums.iloc[i]
        ch = chromos.iloc[i]

        if en and len(en) >= 17:
            for j, name in enumerate(cols):
                morph.at[i, name] = en[j]
        elif en:
            for j in range(min(len(en), len(cols))):
                morph.at[i, cols[j]] = en[j]

        if pd.isna(morph.at[i, "wing_span"]) and ch and len(ch) >= 1:
            morph.at[i, "wing_span"] = ch[0]

    # Wing AR preso da chromosome[1]
    wing_ar = []
    for i in df.index:
        ch = chromos.iloc[i]
        wing_ar.append(float(ch[1]) if (ch and len(ch) > 1) else np.nan)
    morph["wing_AR"] = wing_ar

    # Tail AR (per completezza)
    with np.errstate(divide='ignore', invalid='ignore'):
        morph["elevator_AR"] = morph["elevator_span"] / morph["elevator_chord"]
        morph["rudder_AR"]   = morph["rudder_span"] / morph["rudder_chord"]
        morph.loc[~np.isfinite(morph["elevator_AR"]), "elevator_AR"] = np.nan
        morph.loc[~np.isfinite(morph["rudder_AR"]),   "rudder_AR"]   = np.nan

    # Aree di superficie
    morph["wing_area"]     = morph["wing_span"]     * morph["wing_chord"]
    morph["elevator_area"] = morph["elevator_span"] * morph["elevator_chord"]
    morph["rudder_area"]   = morph["rudder_span"]   * morph["rudder_chord"]

    return morph

# === Scelta colonne di fitness ===
def choose_metric_columns(df):
    mapping = {}
    mapping["Velocity"]        = "vel_v"  if "vel_v"  in df.columns else ("ff_0" if "ff_0" in df.columns else None)
    mapping["Efficiency"]      = "eff_E"  if "eff_E"  in df.columns else ("ff_1" if "ff_1" in df.columns else None)
    mapping["Maneuverability"] = "prog_P" if "prog_P" in df.columns else ("ff_2" if "ff_2" in df.columns else None)
    return {k: v for k, v in mapping.items() if v is not None}

# === Media normalizzata sulle top solutions ===
def normalized_means(df, morph, metric_col, params, top_frac=0.10):
    mask_minp = df["minimal_p"].apply(lambda x: np.isfinite(x) and np.isclose(x, 500.0))
    y = df.loc[mask_minp, metric_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(y) == 0:
        return pd.Series({p: np.nan for p in params}, index=params)

    thr = np.nanpercentile(y, 100*(1-top_frac))
    top_mask = (df[metric_col] >= thr) & mask_minp

    ref_min = morph.loc[mask_minp, params].min(axis=0, skipna=True)
    ref_max = morph.loc[mask_minp, params].max(axis=0, skipna=True)

    means = {}
    for p in params:
        raw_mean = morph.loc[top_mask, p].astype(float).mean(skipna=True)
        mn, mx = ref_min[p], ref_max[p]
        means[p] = np.nan if (pd.isna(raw_mean) or pd.isna(mn) or pd.isna(mx) or mx == mn) else (raw_mean - mn)/(mx - mn)

    return pd.Series(means, index=params)

# === Pretty labels ===
def prettify(name: str) -> str:
    mapping = {
        "wing_AR": "wing AR",
        "elevator_area": "elevator area",
        "rudder_area": "rudder area",
        "dihedral_deg": "dihedral",
        # α_{L=0} RIMOSSO DAL PLOT
    }
    return mapping.get(name, name.replace("_", " "))

# === Funzione di plotting riusabile ===
def make_plot(series_dict, ordered_params, title, out_path, metrics_to_show=None, show_groups=True):
    # estetica & dimensioni come richiesto
    fig = plt.figure(figsize=(11.5, 7.2))
    ax = plt.gca()
    x = np.arange(len(ordered_params))

    # Bande di gruppo: solo 2 gruppi ora
    # 0..2 = Aerodynamic surfaces (AR & tail areas)
    # 3    = Stability-related (dihedral setting)
    group_spans = [
        (-0.5, 2.5, ("Aerodynamic surfaces", "AR & tail areas"), "#e8f1ff"),
        ( 2.5, 3.5, ("Stability-related",    "dihedral setting"), "#e9ffe8"),
    ]

    # ======= NOVITÀ: disattivabili per il plot "tutte e tre" =======
    if show_groups:
        for x0, x1, _, color in group_spans:
            ax.axvspan(x0, x1, color=color, alpha=0.6, zorder=0)

    # Selezione metriche da tracciare
    if metrics_to_show is None:
        metrics_to_show = list(series_dict.keys())

    marker_map = {"Velocity": "o", "Efficiency": "s", "Maneuverability": "^"}
    color_map  = {"Velocity": "tab:orange", "Efficiency": "tab:green", "Maneuverability": "tab:blue"}

    for fit_name in metrics_to_show:
        if fit_name not in series_dict:
            continue
        s = series_dict[fit_name]
        label = "Obstacle Avoidance" if fit_name == "Maneuverability" else fit_name
        ax.plot(
            x,
            s.values.astype(float),
            linestyle="None",
            marker=marker_map.get(fit_name, "o"),
            color=color_map.get(fit_name, None),
            alpha=0.7,
            markersize=7,
            label=label,
            zorder=2
        )

    # Assi, ticks, griglia
    tick_labels = [prettify(p) for p in ordered_params]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, len(x)-0.5)
    ax.set_ylabel("Normalized mean value")
    ax.set_title(title)
    ax.grid(True, which="major", axis="both", alpha=0.25, zorder=1)

    # Legenda boxed
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.68, 0.98),
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="black"
    )
    for text in legend.get_texts():
        text.set_fontsize(10)

    # ======= NOVITÀ: etichette di gruppo mostrate solo se richieste =======
    if show_groups:
        plt.subplots_adjust(bottom=0.42)
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for x0, x1, (gtitle, subtitle), _ in group_spans:
            xm = (x0 + x1) / 2.0
            ax.text(xm, -0.23, gtitle,   ha="center", va="top", fontsize=12, fontweight="semibold", transform=trans)
            ax.text(xm, -0.30, subtitle, ha="center", va="top", fontsize=10, fontstyle="italic", transform=trans)
    else:
        # margine inferiore più contenuto quando non ci sono le label di gruppo
        plt.subplots_adjust(bottom=0.18)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

# === MAIN ===
if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    morph = build_morphology_df(df)
    metrics = choose_metric_columns(df)

    # === SOLO LE 4 GRANDEZZE RICHIESTE (α_{L=0} eliminato) ===
    ordered_params = [
        "wing_AR",        # Wing Aspect Ratio
        "elevator_area",  # Elevator surface area
        "rudder_area",    # Rudder surface area
        "dihedral_deg",   # Dihedral angle  (stability-related)
    ]

    # Serie per ciascuna fitness (top 10%)
    series = {
        fit: normalized_means(df, morph, col, ordered_params, top_frac=0.10)
        for fit, col in metrics.items()
    }

    # --- PLOT 1: SOLO Velocity & Efficiency ---
    only_vel_eff = [m for m in ["Velocity", "Efficiency"] if m in series]
    title1 = "Selected Morphology Means (top 10%): AR, Tail Areas, Dihedral — Velocity & Efficiency"
    out1 = OUT_DIR / "morph_means_top10_minp500_v11_vel_eff.png"

    # --- PLOT 2: TUTTE E TRE (se presenti) ---
    title2 = "Selected Morphology Means (top 10%): AR, Tail Areas, Dihedral — All metrics"
    out2 = OUT_DIR / "morph_means_top10_minp500_v11_all.png"
    # --- PLOT 1: SOLO Velocity & Efficiency (invariato: bande e scritte presenti) ---
    make_plot(series, ordered_params, title1, out1, metrics_to_show=only_vel_eff)

    # --- PLOT 2: TUTTE E TRE — senza divisione né scritte di gruppo ---
    make_plot(
        series,
        ordered_params,
        title2,
        out2,
        metrics_to_show=[k for k in ["Velocity","Efficiency","Maneuverability"] if k in series],
        show_groups=False  # <— rimuove bande opache e label "Aerodynamic surfaces / Stability-related"
    )


    print({"plot_vel_eff": str(out1), "plot_all": str(out2)})
