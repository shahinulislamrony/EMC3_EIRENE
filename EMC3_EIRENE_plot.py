
##Shahinul, 10022025


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



# Set working directory
os.chdir(r'C:\W7X\t2120ms-20250929T105025Z-1-001\t2120ms\OUTPUT')
inputgeofile = r'C:\W7X\20171207_21-20250921T133000Z-1-001\20171207_21\input.geo'
gridfile = r'C:\W7X\20171207_21-20250921T133000Z-1-001\20171207_21\grid3D.dat'
Bfile = r'C:\W7X\20171207_21-20250921T133000Z-1-001\20171207_21\bfield.dat'
EMC3folder = r'C:\W7X\t2120ms-20250929T105025Z-1-001'

def load_flexible(filename):
    """Reads all numbers from a file, regardless of columns per row."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            for val in line.strip().split():
                try:
                    data.append(float(val))
                except ValueError:
                    continue  # skip non-numeric values
    return np.array(data)

def load_cell_geo(filename):
    """Reads CELL_GEO file: returns data (as float array) and header (as int list)."""
    with open(filename, 'r') as f:
        header = f.readline()
        data = []
        for line in f:
            for val in line.strip().split():
                try:
                    data.append(float(val))
                except ValueError:
                    continue
    return np.array(data), [int(x) for x in header.split()]

# Read grid info from input.geo
print('read grid info...')
with open(inputgeofile, 'r') as f:
    for _ in range(7):
        a = f.readline()
    NZ = int(a)
    IZ = np.arange(1, NZ+1)
    Nr, Nth, Nphi = [], [], []
    for i in range(NZ):
        f.readline()
        a = f.readline()
        vals = list(map(int, a.split()))
        Nr.append(vals[0])
        Nth.append(vals[1])
        Nphi.append(vals[2])
Nr = np.array(Nr)
Nth = np.array(Nth)
Nphi = np.array(Nphi)

# Mesh and grid offsets
Mesh_p_os = [0]
Grid_p_os = [0]
for iz in range(len(IZ)-1):
    Mesh_p_os.append((Nr[iz]-1)*(Nth[iz]-1)*(Nphi[iz]-1) + Mesh_p_os[iz])
    Grid_p_os.append((Nr[iz])*(Nth[iz])*(Nphi[iz]) + Grid_p_os[iz])

# Read the grid
print('read grid ...')
with open(gridfile, 'r') as f:
    n = list(map(int, f.readline().split()))
    R, Z, PHI = [], [], []
    i, step, iz = 0, 1, 0
    R1, Z1, Phi = [], [], []
    while True:
        line = f.readline()
        if not line:
            break
        a = list(map(float, line.split()))
        if len(a) == 1 and step == 1:
            Phi.append(a[0])
            step = 2
        elif step == 2 and len(a) == 4 and np.mean(a) < 200:
            Z1.extend(a)
            step = 3
        elif step == 2 and len(a) == 4:
            R1.extend(a)
        elif step == 2 and len(a) < 4:
            R1.extend(a)
            step = 3
        elif step == 3 and len(a) == 4:
            Z1.extend(a)
            if f.tell() == os.fstat(f.fileno()).st_size:
                R.append(R1)
                Z.append(Z1)
                PHI.append([Phi[i]]*len(R1))
        elif step == 3 and len(a) < 4:
            b = Z1 + a
            if len(b) == len(R1):
                Z1.extend(a)
                step = 1
            else:
                if len(a) == 1:
                    Phi.append(a[0])
                    step = 2
                else:
                    n.append(a)
                    step = 1
            R.append(R1)
            Z.append(Z1)
            PHI.append([Phi[i]]*len(R1))
            i += 1
            R1, Z1 = [], []
            if i > n[2]:
                i = 1
                iz += 1

# Flatten grid arrays
RG, ZG, PHIG = [], [], []
Mesh_p_os[0] = 0
for iz in range(len(R)):
    RG.extend(R[iz])
    ZG.extend(Z[iz])
    PHIG.extend(PHI[iz])

RG = np.array(RG)
ZG = np.array(ZG)
PHIG = np.array(PHIG)

# Read B field
print('read B field...')
Bf = np.loadtxt(Bfile)
cond = PHIG == 36

# Read cell geo
print('read cell geo...')
IDCELL_data, Ncell = load_cell_geo('../CELL_GEO')
IDCELL = IDCELL_data  # Already skips header

# Read cell length
print('read cell length...')
LGCELL = load_flexible('../LG_CELL')
LGCELL = LGCELL[~np.isnan(LGCELL)]

# Read temperature
print("read Temperature...")
TE_TI = load_flexible('TE_TI')
TE = TE_TI[:Ncell[1]]
TI = TE_TI[Ncell[1]:2*Ncell[1]]

# Read density
print("read density...")
try:
    N = load_flexible('DENSITY')
except:
    N = np.zeros(Ncell[1])  # or shape as needed

# Read Mach number
print("read mach number...")
M = load_flexible('MACH_NUMBER')

# Read impurity radiation per ion
print("read impurity radiation separated by ions...")
RAD_files = glob.glob('../RADIATION*')
RAD = []

if len(RAD_files) == 0:
    print("No impurity radiation files found. Setting RAD_padded = 0")
    RAD_padded = np.zeros((1, Ncell[1]))  # one species, zero for all cells
else:
    for f in RAD_files:
        out = load_flexible(f)
        RAD.append(out)
    shapes = [x.shape[0] for x in RAD]
    max_len = max(shapes)
    RAD_padded = np.vstack([
        np.pad(x, (0, max_len-len(x)), mode='constant', constant_values=np.nan) 
        if len(x) < max_len else x[:max_len] 
        for x in RAD
    ])

# Read connection length
try:
    print('read connection length...')
    CL = load_flexible('../CONNECTION_LENGTH')
    CL = CL[~np.isnan(CL)]
except:
    CL = 0

n_species = 9
n_cells = Ncell[1]
expected_size = n_species * n_cells

if N.size != expected_size:
    print(f"Warning: Trimming N from {N.size} to {expected_size} elements.")
    N = N[:expected_size]

N = N.reshape((n_species, n_cells))
print("New N.shape:", N.shape)  # Should be (9, n_cells)

I1_list, I2_list, I3_list, I4_list, I5_list, I6_list, I7_list, I8_list = [], [], [], [], [], [], [], []

for iz in range(len(Nr)):
    for K in range(Nphi[iz] - 1):
        for J in range(Nth[iz] - 1):
            for I in range(Nr[iz] - 1):
                idx_base = Grid_p_os[iz]
                nR = Nr[iz]
                nTh = Nth[iz]
                # Calculate the 8 indices for this cell
                I1 = I   + J    * nR + K    * nR * nTh + idx_base
                I2 = I+1 + J    * nR + K    * nR * nTh + idx_base
                I3 = I+1 + (J+1)* nR + K    * nR * nTh + idx_base
                I4 = I   + (J+1)* nR + K    * nR * nTh + idx_base
                I5 = I   + J    * nR + (K+1)* nR * nTh + idx_base
                I6 = I+1 + J    * nR + (K+1)* nR * nTh + idx_base
                I7 = I+1 + (J+1)* nR + (K+1)* nR * nTh + idx_base
                I8 = I   + (J+1)* nR + (K+1)* nR * nTh + idx_base
                # Append to lists
                I1_list.append(I1)
                I2_list.append(I2)
                I3_list.append(I3)
                I4_list.append(I4)
                I5_list.append(I5)
                I6_list.append(I6)
                I7_list.append(I7)
                I8_list.append(I8)

# Convert lists to arrays
I1 = np.array(I1_list)
I2 = np.array(I2_list)
I3 = np.array(I3_list)
I4 = np.array(I4_list)
I5 = np.array(I5_list)
I6 = np.array(I6_list)
I7 = np.array(I7_list)
I8 = np.array(I8_list)

# Stack for easy cell-vertex addressing
cell_vertex_indices = np.stack([I1, I2, I3, I4, I5, I6, I7, I8], axis=1)  # shape (n_cells, 8)

# --- Compute cell-centered coordinates ---
PHI_vertices = PHIG[cell_vertex_indices]  # shape (n_cells, 8)
R_vertices   = RG[cell_vertex_indices]
Z_vertices   = ZG[cell_vertex_indices]

PHIp = PHI_vertices.mean(axis=1)  # shape (n_cells,)
Rp   = R_vertices.mean(axis=1)
Zp   = Z_vertices.mean(axis=1)

print("N.shape:", N.shape)           # (9, n_cells)
print("Rp.shape:", Rp.shape)         # (n_cells,)
print("Zp.shape:", Zp.shape)         # (n_cells,)
print("PHIp.shape:", PHIp.shape)     # (n_cells,)

# --- Select toroidal slice ---
# 1. Find active cell indices
active_mask = IDCELL >= 0
active_indices = np.where(active_mask)[0]  # This is (n_cells,)

# 2. Subset the full mesh arrays to get only active cells
Rp_active = Rp[active_indices]         # (n_cells,)
Zp_active = Zp[active_indices]         # (n_cells,)
PHIp_active = PHIp[active_indices]     # (n_cells,)

# 3. Now make your mask for the toroidal slice
phi0 = 36
closest_phi_value = PHIp_active[np.argmin(np.abs(PHIp_active - phi0))]
cond_cell = np.isclose(PHIp_active, closest_phi_value)  # (n_cells,)

print("Rp_active:", Rp_active.shape)
print("Zp_active:", Zp_active.shape)
print("PHIp_active:", PHIp_active.shape)
print("N:", N.shape)
print("TE:", TE.shape)
print("cond_cell:", cond_cell.shape)
print("Points to plot:", np.sum(cond_cell))

print(f"Closest phi value to {phi0} is {closest_phi_value}")



# --- Prepare polygons and color data for active cells in the selected toroidal slice ---
# Only use the first 4 vertices for each cell (as in MATLAB patch)
polygons = []
density_vals = []
te_vals = []

for idx in np.where(cond_cell)[0]:
    cell_idx = active_indices[idx]      # mesh cell index (0..3151871)
    data_idx = int(IDCELL[cell_idx])    # data index (0..522060), must be int!
    if data_idx < 0 or data_idx >= N.shape[1]:
        continue  # skip invalid
    verts_R = R_vertices[cell_idx, :4] / 100  # cm to m
    verts_Z = Z_vertices[cell_idx, :4] / 100
    poly = np.column_stack((verts_R, verts_Z))
    polygons.append(poly)
    density_vals.append(np.log10(N[0, data_idx] * 1e6))
    te_vals.append(TE[data_idx])
    



fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

# Density patch plot
coll = PolyCollection(polygons, array=np.array(density_vals), cmap='turbo', edgecolor='none')
axs[0].add_collection(coll)
axs[0].autoscale()
axs[0].set_aspect('equal')
axs[0].set_xlabel('R [m]')
axs[0].set_ylabel('Z [m]')
axs[0].set_title('log$_{10}$ n$_e$ [m$^{-3}$]')

# Electron temperature patch plot
coll2 = PolyCollection(polygons, array=np.array(te_vals), cmap='turbo', edgecolor='none')
axs[1].add_collection(coll2)
axs[1].autoscale()
axs[1].set_aspect('equal')
axs[1].set_xlabel('R [m]')
axs[1].set_title('T$_e$ [eV]')
# Hide redundant y-label and y-ticks on right subplot
axs[1].set_ylabel('')
axs[1].tick_params(axis='y', labelleft=False)

# Adjust horizontal space between subplots
plt.subplots_adjust(wspace=0.25)  # Increase for more space if needed


# Place colorbars next to each subplot, matching their height, with more pad
for ax, coll_item in zip(axs, [coll, coll2]):   # extend list if you have more
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.3)  # increase pad for more spacing
    plt.colorbar(coll_item, cax=cax)  # no label


plt.show()




def make_patch_plot(polygons, values, title, cmap='turbo', cbar_label='', ax=None):
    coll = PolyCollection(polygons, array=np.array(values), cmap=cmap, edgecolor='none')
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    ax.add_collection(coll)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title(title)
    
    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.18)
    plt.colorbar(coll, cax=cax, label=cbar_label)
    return ax



density_vals, te_vals, ti_vals, mach_vals, cl_vals = [], [], [], [], []
rad_vals = [[] for _ in range(RAD_padded.shape[0])]  # one list per ion species


polygons = []
for idx in np.where(cond_cell)[0]:
    cell_idx = active_indices[idx]
    data_idx = int(IDCELL[cell_idx])
    if data_idx < 0 or data_idx >= N.shape[1]:
        continue
    
    verts_R = R_vertices[cell_idx, :4] / 100  # cm → m
    verts_Z = Z_vertices[cell_idx, :4] / 100
    poly = np.column_stack((verts_R, verts_Z))
    polygons.append(poly)

    density_vals.append(np.log10(N[0, data_idx] * 1e6))   # log10 ne [m^-3]
    te_vals.append(TE[data_idx])
    ti_vals.append(TI[data_idx])
    mach_vals.append(M[data_idx])
    if isinstance(CL, np.ndarray) and len(CL) > data_idx:
        cl_vals.append(CL[data_idx])
    for i in range(RAD_padded.shape[0]):
        if data_idx < RAD_padded.shape[1]:
            rad_vals[i].append(RAD_padded[i, data_idx])


fig, axs = plt.subplots(2, 3, figsize=(18, 10))
axs = axs.flatten()

make_patch_plot(polygons, density_vals, 'log$_{10}$ n$_e$', cmap='turbo', cbar_label='log$_{10}$ n$_e$ [m$^{-3}$]', ax=axs[0])
make_patch_plot(polygons, te_vals, 'T$_e$', cmap='plasma', cbar_label='T$_e$ [eV]', ax=axs[1])
make_patch_plot(polygons, ti_vals, 'T$_i$', cmap='plasma', cbar_label='T$_i$ [eV]', ax=axs[2])
make_patch_plot(polygons, mach_vals, 'Mach number', cmap='coolwarm', cbar_label='M', ax=axs[3])

# Total radiation
if len(rad_vals) > 0:
    rad_sum = np.sum([np.array(rv) for rv in rad_vals], axis=0)
    make_patch_plot(polygons, rad_sum, 'Total Radiation',
                    cmap='inferno', cbar_label='ΣRad [a.u.]', ax=axs[4])

# Hide the last unused subplot
axs[5].axis('off')

plt.tight_layout()
plt.show()



#now for toroidal slices and save in the folder




# List of phi0 values to plot
angles = [0, 5, 10, 15, 20, 25, 30, 36]

for phi0 in angles:
    # Make folder for each angle
    outdir = f"phi_{phi0}"
    os.makedirs(outdir, exist_ok=True)

    # Pick slice closest to phi0
    closest_phi_value = PHIp_active[np.argmin(np.abs(PHIp_active - phi0))]
    cond_cell = np.isclose(PHIp_active, closest_phi_value)

    print(f"phi0 = {phi0}, closest phi = {closest_phi_value}, cells = {np.sum(cond_cell)}")

    # --- Gather values for this slice ---
    polygons, density_vals, te_vals, ti_vals, mach_vals, cl_vals = [], [], [], [], [], []
    rad_vals = [[] for _ in range(RAD_padded.shape[0])]

    for idx in np.where(cond_cell)[0]:
        cell_idx = active_indices[idx]
        data_idx = int(IDCELL[cell_idx])
        if data_idx < 0 or data_idx >= N.shape[1]:
            continue

        verts_R = R_vertices[cell_idx, :4] / 100  # cm → m
        verts_Z = Z_vertices[cell_idx, :4] / 100
        poly = np.column_stack((verts_R, verts_Z))
        polygons.append(poly)

        density_vals.append(np.log10(N[0, data_idx] * 1e6))
        te_vals.append(TE[data_idx])
        ti_vals.append(TI[data_idx])
        mach_vals.append(M[data_idx])
        if isinstance(CL, np.ndarray) and len(CL) > data_idx:
            cl_vals.append(CL[data_idx])
        for i in range(RAD_padded.shape[0]):
            if data_idx < RAD_padded.shape[1]:
                rad_vals[i].append(RAD_padded[i, data_idx])

    # --- Make plots ---
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()

    make_patch_plot(polygons, density_vals, 'log$_{10}$ n$_e$', cmap='turbo',
                    cbar_label='log$_{10}$ n$_e$ [m$^{-3}$]', ax=axs[0])
    make_patch_plot(polygons, te_vals, 'T$_e$', cmap='plasma',
                    cbar_label='T$_e$ [eV]', ax=axs[1])
    make_patch_plot(polygons, ti_vals, 'T$_i$', cmap='plasma',
                    cbar_label='T$_i$ [eV]', ax=axs[2])
    make_patch_plot(polygons, mach_vals, 'Mach number', cmap='coolwarm',
                    cbar_label='M', ax=axs[3])

    # Total radiation
    if len(rad_vals) > 0:
        rad_sum = np.sum([np.array(rv) for rv in rad_vals], axis=0)
        make_patch_plot(polygons, rad_sum, 'Total Radiation',
                        cmap='inferno', cbar_label='ΣRad [a.u.]', ax=axs[4])

    axs[5].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"slice_phi{phi0}.png"), dpi=300)
    plt.close(fig)

