#!/usr/bin/python3

"""
    ==============================
    = Plot the floor comparisons =
    ==============================

    Read in the median and width files, and then create a scatter
    plot that shows how each of the selections affect the results
"""

# Import packages
import orbit_io
import halo_analysis as halo
import gizmo_analysis as gizmo
import utilities as ut
import numpy as np
import pandas as pd
import satellite_io
import matplotlib
from matplotlib import pyplot as plt
import time
import matplotlib.cm as cm
print('Read in the tools')

### Set path and initial parameters
loc = 'mac'
sim_data = satellite_io.SatelliteRead(gal1='m12i', location=loc)
sat_analysis = satellite_io.SatelliteAnalysis(gal1='m12i', location=loc)
#
print('Set paths')

mw_sats_1Mpc =     ['Antlia II', 'Aquarius II', 'Aquarius III', 'Bootes I', 'Bootes II', 'Bootes III', \
                    'Bootes IV', 'Bootes V', 'Canes Venatici I', 'Canes Venatici II', 'Carina', 'Carina II', \
                    'Carina III', 'Centaurus I', 'Cetus II', 'Cetus III', 'Columba I', 'Coma Berenices', \
                    'Crater II', 'Draco', 'Draco II', 'Eridanus II', 'Eridanus III', 'Eridanus IV', \
                    'Fornax', 'Grus I', 'Grus II', 'Hercules', 'Horologium I', 'Horologium II', \
                    'Hydra II', 'Hydrus I', 'Indus I', 'Leo I', 'Leo II', 'Leo IV', \
                    'Leo V', 'Leo VI', 'Leo A', 'Leo T', 'Leo Minor I', 'Pegasus III', \
                    'Pegasus IV', 'Phoenix I', 'Phoenix II', 'Pictor I', 'Pictor II', 'Pisces II', \
                    'Reticulum II', 'Reticulum III', 'Sagittarius', 'Sagittarius II', 'Sculptor', 'Segue 1', \
                    'Segue 2', 'Sextans', 'Sextans II', 'Triangulum II', 'Tucana I', 'Tucana II', \
                    'Tucana III', 'Tucana IV', 'Tucana V', 'Ursa Major I', 'Ursa Major II', 'Ursa Minor', \
                    'Virgo I', 'Virgo II', 'Virgo III', 'Willman 1']

cols = ['Satellite', '1/1/1', '1/3/3', '1/3/5', '1/5/5', '1/7/7', '1/10/10',
       '3/1/1', '3/3/3', '3/3/5', '3/5/5', '3/7/7', '3/10/10', '5/1/1',
       '5/3/3', '5/3/5', '5/5/5', '5/7/7', '5/10/10', '7/1/1', '7/3/3',
       '7/3/5', '7/5/5', '7/7/7', '7/10/10', '10/1/1', '10/3/3', '10/3/5',
       '10/5/5', '10/7/7', '10/10/10']

orbit_properties = ['dperi_min', 'dperi_rec', 'tperi_min', 'tperi_rec', 'vperi_min', 'vperi_rec', 'nperi', 'dapo', 'tapo', 'infall', 'ke', 'ell']

finalists = ['10/5/5', '10/7/7', '10/10/10']
pair_tests = [('10/5/5', '10/7/7'),
              ('10/5/5', '10/10/10')]

# colour helpers
cmap        = cm.get_cmap('tab20')
marker_map  = dict(zip(finalists, ['o', 's', '^']))

results_dir = sim_data.home_dir+f'/orbit_data/plots/summary/paper_3/combined_floors_physical/MW_population'

property = 'infall'
median_data = pd.read_csv(f"/Users/isaiahsantistevan/simulation/orbit_data/paper_III/floor_tests_headers_physical/{property}_median.csv", index_col=0, usecols=cols).replace(-1, np.nan)
#median_data = pd.read_csv(f"/Users/isaiahsantistevan/simulation/orbit_data/paper_III/floor_tests_headers_physical/{property}_median.csv", index_col=0).replace(-1, np.nan)
median_data_finalists = median_data[finalists]

"""
    Working case for one satellite, Antlia II
"""
# # Working on Antlia II right now
# #comparison = median_data['3/3/3'][0] # Comparing against a particular selection criteria
# infall_antlia = median_data.iloc[0]
# median_all = np.median(infall_antlia)

# # Compute the raw difference between the choices and the comparison array (median over all)
# median_diff = np.abs(infall_antlia - median_all)

# # Compute the mean absolute deviation
# #MAD = np.sum(np.abs(infall_antlia - median_all))/len(infall_antlia)

# # Compute the median absolute deviation
# MAD = np.median(median_diff)


"""
    Now scale up the previous code for all satellites for the infall time comparisons
"""
results = np.zeros((len(cols), len(mw_sats_1Mpc))) # I really want len(cols)-1, but then also add the row for the MAD value
results_diff = np.zeros((len(cols), len(mw_sats_1Mpc)))

# Loop over all satellites
for i in range(len(mw_sats_1Mpc)):
    #
    # Get the list of infall times for the current satellite
    infall_satellite = median_data.iloc[i]
    #
    # Calculate the median over all selections
    median_all_selection = np.median(infall_satellite)
    #
    # Calculate the median difference between each selection and the overall median
    median_diff = np.abs(infall_satellite - median_all_selection)
    #
    # Calculate the median absolute deviation
    MAD = np.median(median_diff)
    #
    # First save the data from the raw differences
    results_diff[0,i] = MAD
    results_diff[1:, i] = median_diff
    #
    if MAD == 0:
        #
        nonzero = median_diff[median_diff > 0]
        nonzeroMAD = np.median(nonzero)
        #
        denominator = np.where(median_diff > 0, median_diff, nonzeroMAD)
        results[0, i] = MAD
        results[1:, i] = median_diff/denominator
    else:
        results[0, i] = MAD
        results[1:, i] = median_diff/MAD

row_labels = [
    "MAD",
    "1/1/1", "1/3/3", "1/3/5", "1/5/5", "1/7/7", "1/10/10",
    "3/1/1", "3/3/3", "3/3/5", "3/5/5", "3/7/7", "3/10/10",
    "5/1/1", "5/3/3", "5/3/5", "5/5/5", "5/7/7", "5/10/10",
    "7/1/1", "7/3/3", "7/3/5", "7/5/5", "7/7/7", "7/10/10",
    "10/1/1", "10/3/3", "10/3/5", "10/5/5", "10/7/7", "10/10/10"
]

# Save the data for the MAD table first
df = pd.DataFrame(results, index=row_labels, columns=mw_sats_1Mpc)
df.index.name = "Satellite"
#
csv_path = "mad_infall_comaprisons.csv"
df.to_csv(results_dir+"/"+csv_path)

# Now save the data for the raw differences
df = pd.DataFrame(results_diff, index=row_labels, columns=mw_sats_1Mpc)
df.index.name = "Satellite"
#
csv_path = "raw_infall_comaprisons.csv"
df.to_csv(results_dir+"/"+csv_path)




# Trying to make sense of all of the data
df_full = pd.DataFrame(results_diff, index=row_labels, columns=mw_sats_1Mpc)
df = df_full.loc[finalists]
df_other = df_full.drop(index=finalists)
#
tolerance = 0.25
verdict = (pd.DataFrame({
        'Δ_10/5/5'   : df.loc['10/5/5'],
        'Δ_10/7/7'   : df.loc['10/7/7'],
        'Δ_10/10/10' : df.loc['10/10/10'],}).assign(
        n_fail = lambda d: (d.abs() > tolerance).sum(axis=1),
        pass_  = lambda d: d['n_fail'] == 0))

full_spread = np.nanmax(np.abs(df_other), axis=0)
verdict['max_Δ_other_selection'] = full_spread

verdict.to_csv(results_dir+"/per_satellite_infall_verdict.csv")









# Try creating a heatmap to see if certain selections would be okay
tolerance = 0.25
df = pd.DataFrame(results_diff, index=row_labels, columns=mw_sats_1Mpc)
within = np.abs(df) <= tolerance 
#
fig, ax = plt.subplots(figsize=(14,8))
plot_matrix = within.astype(float).where(within, 0.0).fillna(-1.0)

map = plt.cm.get_cmap('RdYlGn')       # red→yellow→green
#cmap.set_under('lightgrey')            # for NaNs / -1

im = ax.imshow(plot_matrix,
               cmap=cmap,
               vmin=0, vmax=1)      # values <0.001 get 'under' colour

# tick labels
ax.set_xticks(np.arange(len(df.columns)))
ax.set_xticklabels(df.columns, rotation=90, fontsize=8)
ax.set_yticks(np.arange(len(df.index)))
ax.set_yticklabels(df.index, fontsize=8)

# colour-bar as legend
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label(f"|Δ| ≤ {tolerance} Gyr   (green = pass, red = fail)", fontsize=9)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['fail', 'pass'])

ax.set_title("Infall-time median bias test (traffic-light view)")
plt.tight_layout()
plt.show()




































def load_stat(prop: str, kind: str, keep_finalists=False) -> pd.DataFrame:
    """
    kind = 'median' | 'width'
    Replaces -1 with NaN and sets satellite names as index.
    """
    fname = (f"/Users/isaiahsantistevan/simulation/orbit_data/paper_III/"
             f"floor_tests_headers_physical/{prop}_{kind}.csv")
    df = (pd.read_csv(fname, index_col=0)
            .replace(-1, np.nan))         # sentinel → NaN
    if keep_finalists:
        df = df[finalists]
    return df

# ---------------------------------------------
def make_range_track(df_median: pd.DataFrame,
                     ndf: pd.DataFrame,
                     prop: str) -> None:
    """
    Grey min–max line + coloured markers for each finalist.
    Saves PDF '<results_dir>/median_<prop>_each_line-n-point.pdf'
    """
    fig_h = 0.42 * len(df_median) + 1        # auto-scale figure height
    fig, ax = plt.subplots(figsize=(8, fig_h))

    for i, (sat, row) in enumerate(df_median.iterrows()):
        vals = row.dropna()
        if vals.empty:
            continue
        ax.hlines(i, vals.min(), vals.max(),
                  color='lightgray', linewidth=2, zorder=1)

        base_color = cmap(i % cmap.N)
        for sel in finalists:
            y = i
            x = row[sel]
            if pd.isna(x):
                continue
            n = ndf.at[sat, sel] if not pd.isna(ndf.at[sat, sel]) else 0
            ax.scatter(x, y,
                       marker=marker_map[sel],
                       facecolor=base_color,
                       edgecolor='k', linewidth=0.4,
                       s=18 + 4*np.sqrt(n),
                       label=sel if i == 0 else None)

    ax.set_yticks(range(len(df_median)))
    ax.set_yticklabels(df_median.index, fontsize=6)
    ax.set_xlabel(label_for(prop, kind='median'))
    ax.invert_yaxis()
    ax.margins(x=0.04)

    # single legend
    if fig.legends:
        fig.legends.clear()
    ax.legend(title='Selection', ncol=len(finalists),
              loc='upper center', bbox_to_anchor=(0.5, 1.02),
              frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(results_dir+f"/median_{prop}_each_line-n-point.pdf")
    plt.close(fig)

# ---------------------------------------------
def make_dumbbells(df_median: pd.DataFrame,
                   ndf: pd.DataFrame,
                   prop: str) -> None:
    """
    Two-panel dumbbell (10/7/7 vs 10/5/5, 10/10/10 vs 10/5/5).
    Saves PDF '<results_dir>/median_<prop>_each_dumbbell.pdf'
    """
    fig_h = 0.42 * len(df_median) + 1
    fig, axes = plt.subplots(1, len(pair_tests),
                             figsize=(5.5*len(pair_tests), fig_h),
                             sharey=True)
    for ax, (left, right) in zip(axes, pair_tests):
        for i, sat in enumerate(df_median.index):
            if pd.isna(df_median.at[sat, left]) or pd.isna(df_median.at[sat, right]):
                continue
            x1 = df_median.at[sat, left]
            x2 = df_median.at[sat, right]
            ax.plot([x1, x2], [i, i], 'grey', lw=0.7, zorder=1)
            ax.scatter([x1, x2], [i, i],
                       s=[marker_size(ndf.at[sat, left]),
                          marker_size(ndf.at[sat, right])],
                       c=['tab:blue','tab:orange'], zorder=2)
        ax.set_title(f'{right}  vs  {left}', fontsize=10)
        ax.set_xlabel(label_for(prop, kind='median'), fontsize=9)
    axes[0].set_yticks(range(len(df_median)))
    axes[0].set_yticklabels(df_median.index, fontsize=6)
    fig.tight_layout()
    fig.savefig(results_dir+f"/median_{prop}_each_dumbbell.pdf")
    plt.close(fig)

# ---------------------------------------------
def make_spread_hist(df_median: pd.DataFrame,
                     prop: str) -> None:##### Add the threshold here and draw it as a vertical line?
    """
    Histogram of max–min across finalists.
    Saves PDF '<results_dir>/median_<prop>_each_spread.pdf'
    """
    spread = df_median.max(axis=1) - df_median.min(axis=1)
    spread = spread.dropna()
    thr = bias_threshold(prop)
    perc90 = np.percentile(spread, 90)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(spread, bins='auto', edgecolor='k', alpha=0.75)
    ax.axvline(thr, ls='--', lw=1.4, c='r', label=f"Threshold ({thr})")
    ax.axvline(perc90, ls='-.', lw=1.4, c='navy', label="90% of satellties")
    ax.legend(fontsize=8)
    ax.set_xlabel(f'Max–Min among finalists ({label_unit(prop)})')
    ax.set_ylabel('Number of satellites')
    ax.set_title(f'Spread of {label_short(prop)} across three selections')
    fig.tight_layout()
    fig.savefig(results_dir+f"/median_{prop}_each_spread.pdf")
    plt.close(fig)

# ---------------------------------------------
# tiny helpers
# ---------------------------------------------
def bias_threshold(prop: str) -> float:
    """Return absolute bias threshold for a given orbit property."""
    if prop.endswith(('tperi_min','tperi_rec','tapo','infall')):   # time [Gyr]
        return 0.24                   # one MW rotation
    if prop.endswith(('dperi_min','dperi_rec','dapo')):            # distance [kpc]
        return 10.0
    if prop.endswith(('vperi_min','vperi_rec')):                   # speed [km/s]
        return 10.0
    # everything else → no automatic limit
    return np.inf

def marker_size(n):             # tweak once, use everywhere
    return 20 + 6*np.sqrt(n)

def label_unit(prop):
    # hard-code units once:
    units = {'infall':'Gyr', 'tperi_min':'Gyr', 'tperi_rec':'Gyr',
             'tapo':'Gyr', 'ke':'km² s⁻²', 'dapo':'kpc', 'dperi_min':'kpc',
             'dperi_rec':'kpc', 'vperi_min':'km s⁻¹', 'vperi_rec':'km s⁻¹',
             'ell':'—', 'nperi':'—'}
    return units.get(prop, '')

def label_short(prop):
    nice = {'infall':'$t_{\\rm infall}$', 'nperi':'$N_{\\rm peri}$', 'ke':'KE'}
    return nice.get(prop, prop)

def label_for(prop, *, kind):
    base = label_short(prop)
    unit = label_unit(prop)
    suffix = 'median' if kind=='median' else '68% width'
    return f'{base} ({unit}) — {suffix}' if unit else f'{base} — {suffix}'

# ---------------------------------------------
# main driver
# ---------------------------------------------
results_thr_dict = {}
results_spread_dict = {}
flagged_sats = {}
delta_to_mid = {}
for prop in orbit_properties:
    print(f'▶  Processing {prop}')
    med_fin = load_stat(prop, 'median', keep_finalists=True)
    width_fin = load_stat(prop, 'width', keep_finalists=True)     # units match the medians
    # If you already have an analogue-count table, load it instead of ​​ones_like:
    ndf  = pd.DataFrame(1, index=med_fin.index, columns=med_fin.columns)
    #
    thr = bias_threshold(prop)
    spread = med_fin.max(axis=1) - med_fin.min(axis=1)
    midpoint = (med_fin.max(axis=1) + med_fin.min(axis=1))/2
    offset = (med_fin['10/5/5'] - midpoint).abs()
    sigma68 = width_fin['10/5/5']/2
    flag_bad = (spread > thr)|(offset > sigma68)
    #flag_spread = (spread > sigma68['10/5/5'])
    #
    sats_keep = med_fin.index[~flag_bad]
    sats_raise = med_fin.index[flag_bad]
    #
    flagged_sats[prop] = sats_raise
    ##results_spread_dict[prop] = med.index[flag_spread]
    ##delta_to_mid[prop] = abs(med['10/5/5'] - midpoint)
    ##print(f"Property: {prop} - Threshold check: {results_spread_dict[prop]}")
    # ----------------------------
    #make_range_track(med_fin, ndf, prop)
    #make_dumbbells(  med_fin, ndf, prop)
    #make_spread_hist(med_fin, prop, thr)
    #
    df_all_med = load_stat(prop,'median',keep_finalists=False)
    min_f,max_f = df_all_med[finalists].min(axis=1), df_all_med[finalists].max(axis=1)
    others      = [c for c in df_all_med.columns if c not in finalists]
    def all_inside(row):
        sat=row.name
        return ((row>=min_f[sat])&(row<=max_f[sat])).all()
    inside_mask = df_all_med.loc[sats_keep,others].apply(all_inside,axis=1)
    violations  = inside_mask.eq(False)
    if violations.any():
        bad_sats = violations[violations].index.tolist()
        print(f'   ⚠️  Envelope breached for {bad_sats}')
        #df_all_med.loc[bad_sats, others]\
        #          .to_csv(results_root/f'envelope_violations_{prop}.csv')
    else:
        print('   ✅  All exploratory floors stay inside finalist envelope.')









prop = 'infall'
fname_med = (f"/Users/isaiahsantistevan/simulation/orbit_data/paper_III/"
             f"floor_tests_headers_physical/{prop}_median.csv")
fname_width = (f"/Users/isaiahsantistevan/simulation/orbit_data/paper_III/"
             f"floor_tests_headers_physical/{prop}_width.csv")

df_all_med = (pd.read_csv(fname_med, index_col=0).replace(-1, np.nan)) 
df_all_width = (pd.read_csv(fname_width, index_col=0).replace(-1, np.nan)) 
finalists = ['10/5/5', '10/7/7', '10/10/10']
others    = [c for c in df_all_med.columns if c not in finalists]

min_f = df_all_med[finalists].min(axis=1)   # grey-bar left end
max_f = df_all_med[finalists].max(axis=1)   # grey-bar right end


thr = bias_threshold(prop) 
#
spread = df_all_med.max(axis=1) - df_all_med.min(axis=1)  
#
sigma68 = df_all_width['10/5/5'] / 2                  # same index as spread
midpoint = (df_all_med.min(axis=1) + df_all_med.max(axis=1)) / 2
offset   = (df_all_med['10/5/5'] - midpoint).abs()
flag_bad = (spread > thr) | (offset > sigma68)

sats_keep   = df_all_med.index[~flag_bad]   # keep 10/5/5
sats_raise  = df_all_med.index[flag_bad]   # move to stricter floor

# 7. (Optional) save or print for the memo
print(f'Keeping 10/5/5 for {len(sats_keep)} satellites; '
      f'raising floor for {len(sats_raise)}.')

# True if *every* 'other' median lies within the finalist range
inside = df_all_med.loc[sats_keep, others].apply(
            lambda row: (row >= min_f[row.name]) &
                        (row <= max_f[row.name]),
            axis=1
         )

violations = inside.eq(False).any(axis=1)   # any out-of-bounds entry?

if not violations.any():
    print('✅  All exploratory floors stay inside the finalist bar '
          'for satellites you keep at 10/5/5.')
else:
    print('⚠️  The following satellites exceed the finalist bar:\n',
          violations[violations].index.tolist())
    # optional: print which floor and by how much
    bad = df_all_med.loc[violations[violations].index, others]
    diff_lo = (bad.subtract(min_f, axis=0) < 0).stack().loc[lambda s: s]
    diff_hi = (bad.subtract(max_f, axis=0) > 0).stack().loc[lambda s: s]
    print('  Below min  ', diff_lo)
    print('  Above max  ', diff_hi)














# Plot the data
fig, ax = plt.subplots(figsize=(8, 0.45*len(df)))

for i, (sat, row) in enumerate(df.iterrows()):
    vals = row.dropna()
    if vals.empty: 
        continue

    # grey backbone = full min‑to‑max range across all selections
    ax.hlines(i, vals.min(), vals.max(), color='lightgray', linewidth=2, zorder=1)

    # coloured points = your three finalist selections
    base_color = colors[i]
    for sel in finalists:
        if sel in df.columns and not pd.isna(row[sel]):
            x = row[sel]
            n = ndf.at[sat, sel] if not pd.isna(ndf.at[sat, sel]) else 0
            ax.scatter(x, i,
                       marker   = marker_map[sel],
                       facecolor= base_color,
                       edgecolor= 'k',
                       s        = 20 + 4*np.sqrt(n),    # size ∝ √N (tweak as you like)
                       linewidth= 0.5,
                       label    = sel if i == 0 else None)  # one legend entry per sel

# cosmetics
ax.set_yticks(range(len(df)))
ax.set_yticklabels(df.index, fontsize=7)
ax.set_xlabel('Median $t_{\\rm infall}$ [Gyr]')           # change label for other stats
ax.invert_yaxis()                                         # optional: oldest at top
ax.margins(x=0.05)

# single legend keyed by marker shape (colour is irrelevant there)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title='Selection', loc='upper center',
          bbox_to_anchor=(0.5, 1.02), ncol=len(finalists), frameon=False, fontsize=8)

plt.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+f'/orbit_data/plots/summary/paper_3/combined_floors_physical/MW_population/median_{prop}_each_line-n-point.pdf')





### Dumbbell plot


ndf = pd.DataFrame(1, index=df.index, columns=df.columns)

fig, axes = plt.subplots(1, len(pairs),figsize=(6*len(pairs), 0.45*len(df)), sharey=True)

for ax, (left, right) in zip(axes, pairs):
    for i, sat in enumerate(df.index):
        if pd.isna(df.at[sat, left]) or pd.isna(df.at[sat, right]): continue
        x1, x2 = df.at[sat, left], df.at[sat, right]
        ax.plot([x1, x2], [i, i], 'k--', alpha=0.3)
        ax.scatter([x1, x2], [i, i],
                   s=[ndf.at[sat, left], ndf.at[sat, right]],
                   c=['tab:blue','tab:orange'], zorder=3)
    ax.set_title(f'{right}  vs  {left}')
    ax.set_xlabel('Median $t_{\\rm infall}$  [Gyr]')

axes[0].set_yticks(range(len(df)))
axes[0].set_yticklabels(df.index)
plt.tight_layout()#; plt.show()
plt.savefig(sim_data.home_dir+f'/orbit_data/plots/summary/paper_3/combined_floors_physical/MW_population/median_{prop}_each_dumbbell.pdf')




# Plots that show the spread
sub = df[finalists] 

min_sel = sub.min(axis=1, skipna=True)
max_sel = sub.max(axis=1, skipna=True)
spread  = max_sel - min_sel       # absolute range

# gather in one table (optional)
spread_df = pd.DataFrame({
    'min_finalists': min_sel,
    'max_finalists': max_sel,
    'range':         spread
})

spread_df = spread_df.dropna(subset=['range'])

fig, ax = plt.subplots(figsize=(12,8))
ax.hist(spread_df['range'], bins='auto', edgecolor='k', alpha=0.7)
ax.set_xlabel('Max–Min across finalists')
ax.set_ylabel('Number of galaxies')
ax.set_title('Spread among three finalist selections')
plt.tight_layout()#; plt.show()
plt.savefig(sim_data.home_dir+f'/orbit_data/plots/summary/paper_3/combined_floors_physical/MW_population/median_{prop}_each_spread.pdf')


spread_df[spread_df['range'] > 0.5]










# ----------------- 0. LOAD & MASK SENTINEL -----------------------
df_raw = pd.read_csv(f"/Users/isaiahsantistevan/simulation/orbit_data/paper_III/floor_tests_headers_physical/{file_name_median}.csv", index_col=0)


# Any entry == –1 is really “missing” → convert to NaN
df = df_raw.replace(-1, np.nan)

# ----------------- 1. MAIN Δ (needs fiducial) --------------------
fid = df.iloc[:, 0]                      # first column = fiducial
mask = fid.notna()                       # rows that *do* have fiducial

delta_main = (df[mask].sub(fid[mask], axis=0).div(fid[mask], axis=0))

# ----------------- 2. SECONDARY Δ (row‑median baseline) ----------
row_median = df.median(axis=1, skipna=True)   # NaNs (incl. the former –1s) are ignored
delta_rowmed = (df.sub(row_median, axis=0)
                  .div(row_median, axis=0))

# keep true zeros where a proper fiducial exists
delta_rowmed.loc[mask, df.columns[0]] = 0

# rows with *all* NaNs now have row_median = NaN → drop if desired
delta_rowmed = delta_rowmed.dropna(how='all')

# ----------------- 3. QUICK HEAT‑MAP FUNCTION --------------------
mat = delta_main
title = 'Bias vs. fiducial (rows with fiducial only)'
vmax = np.nanmax(np.abs(mat.values))
fig, ax = plt.subplots(figsize=(1.4*mat.shape[1], 0.7*mat.shape[0]))
im = ax.imshow(mat, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
ax.set_xticks(range(len(mat.columns)))
ax.set_xticklabels(mat.columns, rotation=45, ha='right')
ax.set_yticks(range(len(mat.index)))
ax.set_yticklabels(mat.index)
ax.set_title(title)
fig.colorbar(im, ax=ax, pad=0.02,
                label='(value − baseline)/baseline')
fig.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+f'/orbit_data/plots/summary/paper_3/combined_floors_physical/MW_population/median_{prop}_each_heatmap_fid.pdf')


mat = delta_rowmed
title = 'Deviation from row‑median (rows lacking fiducial)'
vmax = np.nanmax(np.abs(mat.values))
fig, ax = plt.subplots(figsize=(1.4*mat.shape[1], 0.7*mat.shape[0]))
im = ax.imshow(mat, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
ax.set_xticks(range(len(mat.columns)))
ax.set_xticklabels(mat.columns, rotation=45, ha='right')
ax.set_yticks(range(len(mat.index)))
ax.set_yticklabels(mat.index)
ax.set_title(title)
fig.colorbar(im, ax=ax, pad=0.02,
                label='(value − baseline)/baseline')
fig.tight_layout()
#plt.show()
plt.savefig(sim_data.home_dir+f'/orbit_data/plots/summary/paper_3/combined_floors_physical/MW_population/median_{prop}_each_heatmap_rowmed.pdf')















# ---------------------------------------------
# config & imports
# ---------------------------------------------
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --- user settings ---
orbit_properties = [
    'dperi_min','dperi_rec','tperi_min','tperi_rec','vperi_min','vperi_rec',
    'nperi','dapo','tapo','infall','ke','ell'
]
# three finalist floor sets, left→right determines “baseline” in dumbbells
finalists   = ['10/5/5', '10/7/7', '10/10/10']
pair_tests  = [('10/5/5','10/7/7'), ('10/5/5','10/10/10')]
results_dir = Path('/Users/isaiahsantistevan/simulation/orbit_data/plots/'
                   'summary/paper_3/combined_floors_physical/MW_population')
results_dir.mkdir(parents=True, exist_ok=True)

# colour helpers
cmap        = cm.get_cmap('tab20')
marker_map  = dict(zip(finalists, ['o', 's', '^']))

# optional: nice Matplotlib defaults
plt.rcParams.update({'axes.spines.right': False,
                     'axes.spines.top'  : False,
                     'savefig.dpi'      : 300})
# ---------------------------------------------
def load_stat(prop: str, kind: str) -> pd.DataFrame:
    """
    kind = 'median' | 'width'
    Replaces -1 with NaN and sets satellite names as index.
    """
    fname = (f"/Users/isaiahsantistevan/simulation/orbit_data/paper_III/"
             f"floor_tests_headers_physical/{prop}_{kind}.csv")
    df = (pd.read_csv(fname, index_col=0)
            .replace(-1, np.nan)         # sentinel → NaN
            .loc[:, finalists])          # keep only finalists here
    return df

# ---------------------------------------------
def make_range_track(df_median: pd.DataFrame,
                     ndf: pd.DataFrame,
                     prop: str) -> None:
    """
    Grey min–max line + coloured markers for each finalist.
    Saves PDF '<results_dir>/median_<prop>_each_line-n-point.pdf'
    """
    fig_h = 0.42 * len(df_median) + 1        # auto-scale figure height
    fig, ax = plt.subplots(figsize=(8, fig_h))

    for i, (sat, row) in enumerate(df_median.iterrows()):
        vals = row.dropna()
        if vals.empty:
            continue
        ax.hlines(i, vals.min(), vals.max(),
                  color='lightgray', linewidth=2, zorder=1)

        base_color = cmap(i % cmap.N)
        for sel in finalists:
            y = i
            x = row[sel]
            if pd.isna(x):
                continue
            n = ndf.at[sat, sel] if not pd.isna(ndf.at[sat, sel]) else 0
            ax.scatter(x, y,
                       marker=marker_map[sel],
                       facecolor=base_color,
                       edgecolor='k', linewidth=0.4,
                       s=18 + 4*np.sqrt(n),
                       label=sel if i == 0 else None)

    ax.set_yticks(range(len(df_median)))
    ax.set_yticklabels(df_median.index, fontsize=6)
    ax.set_xlabel(label_for(prop, kind='median'))
    ax.invert_yaxis()
    ax.margins(x=0.04)

    # single legend
    if fig.legends:
        fig.legends.clear()
    ax.legend(title='Selection', ncol=len(finalists),
              loc='upper center', bbox_to_anchor=(0.5, 1.02),
              frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(results_dir+f"/median_{prop}_each_line-n-point.pdf")
    plt.close(fig)

# ---------------------------------------------
def make_dumbbells(df_median: pd.DataFrame,
                   ndf: pd.DataFrame,
                   prop: str) -> None:
    """
    Two-panel dumbbell (10/7/7 vs 10/5/5, 10/10/10 vs 10/5/5).
    Saves PDF '<results_dir>/median_<prop>_each_dumbbell.pdf'
    """
    fig_h = 0.42 * len(df_median) + 1
    fig, axes = plt.subplots(1, len(pair_tests),
                             figsize=(5.5*len(pair_tests), fig_h),
                             sharey=True)
    for ax, (left, right) in zip(axes, pair_tests):
        for i, sat in enumerate(df_median.index):
            if pd.isna(df_median.at[sat, left]) or pd.isna(df_median.at[sat, right]):
                continue
            x1 = df_median.at[sat, left]
            x2 = df_median.at[sat, right]
            ax.plot([x1, x2], [i, i], 'grey', lw=0.7, zorder=1)
            ax.scatter([x1, x2], [i, i],
                       s=[marker_size(ndf.at[sat, left]),
                          marker_size(ndf.at[sat, right])],
                       c=['tab:blue','tab:orange'], zorder=2)
        ax.set_title(f'{right}  vs  {left}', fontsize=10)
        ax.set_xlabel(label_for(prop, kind='median'), fontsize=9)
    axes[0].set_yticks(range(len(df_median)))
    axes[0].set_yticklabels(df_median.index, fontsize=6)
    fig.tight_layout()
    fig.savefig(results_dir+f"/median_{prop}_each_dumbbell.pdf")
    plt.close(fig)

# ---------------------------------------------
def make_spread_hist(df_median: pd.DataFrame,
                     prop: str) -> None:
    """
    Histogram of max–min across finalists.
    Saves PDF '<results_dir>/median_<prop>_each_spread.pdf'
    """
    spread = df_median.max(axis=1) - df_median.min(axis=1)
    spread = spread.dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(spread, bins='auto', edgecolor='k', alpha=0.75)
    ax.set_xlabel(f'Max–Min among finalists ({label_unit(prop)})')
    ax.set_ylabel('Number of satellites')
    ax.set_title(f'Spread of {label_short(prop)} across three selections')
    fig.tight_layout()
    fig.savefig(results_dir+f"/median_{prop}_each_spread.pdf")
    plt.close(fig)

# ---------------------------------------------
# tiny helpers
# ---------------------------------------------
def marker_size(n):             # tweak once, use everywhere
    return 20 + 6*np.sqrt(n)

def label_unit(prop):
    # hard-code units once:
    units = {'infall':'Gyr', 'tperi_min':'Gyr', 'tperi_rec':'Gyr',
             'tapo':'Gyr', 'ke':'km² s⁻²', 'dapo':'kpc', 'dperi_min':'kpc',
             'dperi_rec':'kpc', 'vperi_min':'km s⁻¹', 'vperi_rec':'km s⁻¹',
             'ell':'—', 'nperi':'—'}
    return units.get(prop, '')

def label_short(prop):
    nice = {'infall':'$t_{\\rm infall}$', 'nperi':'$N_{\\rm peri}$', 'ke':'KE'}
    return nice.get(prop, prop)

def label_for(prop, *, kind):
    base = label_short(prop)
    unit = label_unit(prop)
    suffix = 'median' if kind=='median' else '68% width'
    return f'{base} ({unit}) — {suffix}' if unit else f'{base} — {suffix}'

# ---------------------------------------------
# main driver
# ---------------------------------------------
for prop in orbit_properties:
    print(f'▶  Processing {prop}')
    med  = load_stat(prop, 'median')
    # If you already have an analogue-count table, load it instead of ​​ones_like:
    ndf  = pd.DataFrame(1, index=med.index, columns=med.columns)
    # ----------------------------
    make_range_track(med, ndf, prop)
    make_dumbbells(  med, ndf, prop)
    make_spread_hist(med, prop)
