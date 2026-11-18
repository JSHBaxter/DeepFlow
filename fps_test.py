from pathlib import Path

import nrrd
import numpy as np
import torch
from matplotlib import pyplot as plt

s_bat = 2
s_img = 128

d_b = torch.zeros((s_bat,s_img,s_img,s_img))-1
d_u = torch.zeros((s_bat,s_img,s_img,s_img))-1
d_r = torch.zeros((s_bat,s_img,s_img,s_img))-1

si = torch.arange(s_img).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)/s_img
ap = torch.arange(s_img).float().unsqueeze(0).unsqueeze(0).unsqueeze(-1)/s_img
lr = torch.arange(s_img).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)/s_img


#create simple sphere for rectum
d_r += 2 * ( (lr-1/2)**2 + (ap-(3/4+1.5/16))**2 < (1.5/16)**2 )

#create simple sphere for bladder
d_b += 2 * ( (lr-1/2)**2 + (ap-(1/4))**2 + (si-3/16)**2 < (3/16)**2 )

# Create a curved, croissant-shaped bladder

# Center of the bladder
lr_center = 0.5
ap_center = 0.28
si_center = 0.20

# Dimensions
lr_radius = 0.27
ap_radius = 0.105
si_radius = 0.15

# Controls how strongly the bladder bends
curvature = 0.13

# Normalized left-right position
x = (lr - lr_center) / lr_radius

# Curved centerline:
# The middle is displaced along the AP axis,
# while the extremities remain closer to ap_center.
curved_ap_center = ap_center + curvature * (1 - x**2)

# Curved ellipsoid
bladder = (
    x**2
    + ((ap - curved_ap_center) / ap_radius)**2
    + ((si - si_center) / si_radius)**2
) < 1

d_b += 2 * bladder


#create simple sphere for uterus
d_u += 2 * ( (lr-1/2)**2 + (ap-(1/2+1/16))**2 + (si-3/8)**2 < (2.5/16)**2 )

# s_b, s_u, s_r = FPSF_axial_segmenter.apply(d_b,d_u,d_r)

plt.imshow((d_r+d_b+d_u)[0,int(3/16 * s_img),:,:])
plt.show()
plt.imshow((d_r+d_b+d_u)[0,int(3/8 * s_img),:,:])
plt.show()
plt.imshow((d_r+d_b+d_u)[0,int(4.5/16 * s_img),:,:])
plt.show()


output_dir = Path("synthetic_pelvis")
output_dir.mkdir(parents=True, exist_ok=True)

# First batch element: 3D binary volumes (SI, AP, LR)
volumes = {
    "bladder": (d_b[0] > 0).cpu().numpy().astype(np.uint8),
    "uterus": (d_u[0] > 0).cpu().numpy().astype(np.uint8),
    "rectum": (d_r[0] > 0).cpu().numpy().astype(np.uint8),
}

header = {
    "space": "left-posterior-superior",
    "space directions": [
        [0.0, 0.0, 1.0],  # SI axis
        [0.0, 1.0, 0.0],  # AP axis
        [1.0, 0.0, 0.0],  # LR axis
    ],
    "space origin": [0.0, 0.0, 0.0],
}

for name, volume in volumes.items():
    np.save(output_dir / f"{name}.npy", volume)
    nrrd.write(
        str(output_dir / f"{name}.nrrd"),
        volume,
        header=header,
        index_order="F",
    )

print("Saved .npy and .nrrd volumes.")
