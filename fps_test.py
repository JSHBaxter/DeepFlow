import torch
# from female_pelvis_segmenter_function import FPSF_axial_segmenter
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

#create simple sphere for uterus
d_u += 2 * ( (lr-1/2)**2 + (ap-(1/2+1/16))**2 + (si-3/8)**2 < (2.5/16)**2 )

# s_b, s_u, s_r = FPSF_axial_segmenter.apply(d_b,d_u,d_r)

plt.imshow((d_r+d_b+d_u)[0,int(3/16 * s_img),:,:])
plt.show()
plt.imshow((d_r+d_b+d_u)[0,int(3/8 * s_img),:,:])
plt.show()
plt.imshow((d_r+d_b+d_u)[0,int(4.5/16 * s_img),:,:])
plt.show()


# plt.imshow((s_r+s_b+s_u)[0,int(3/16 * s_img),:,:])
# plt.show()
# plt.imshow((s_r+s_b+s_u)[0,int(3/8 * s_img),:,:])
# plt.show()
# plt.imshow((s_r+s_b+s_u)[0,int(4.5/16 * s_img),:,:])
# plt.show()