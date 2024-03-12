import os
import numpy as np 
import pandas as pd
import time 
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Check if GPU is available 
is_gpu = torch.cuda.is_available()
device = torch.device("cuda" if is_gpu else "cpu")
print("Using GPU:", is_gpu)

# Function to generate a structured grid
def create_grid(domain_size, cell_size):
    num_cells = int(domain_size / cell_size)
    grid = np.zeros((num_cells, num_cells), dtype=int)
    return grid

# Input Parameters
domain_size = 2000  # Size of the square domain
half_domain_size = 1000
cell_size = 1   # Cell size and particle radius
simulation_time = 1
kn = 500000  # Normal stiffness of the spring
dn = 0.5  # Normal damping coefficient
particle_mass = 1
K_graph = 19.2*10000*1
S_graph = K_graph * (cell_size / domain_size) ** 2
restitution_coefficient = 0.7 # coefficient of restitution

damping_coefficient_Alpha = -1*math.log(restitution_coefficient)/math.pi
damping_coefficient_Gamma = damping_coefficient_Alpha/math.sqrt(damping_coefficient_Alpha**2+1)
damping_coefficient_Eta = 2 * damping_coefficient_Gamma * math.sqrt(kn * particle_mass)


# Module 1: Domain discretisation and initial particle insertion
# Create grid
grid = create_grid(domain_size, cell_size)
grid_shape = grid.shape

# Generate particles
npt = int(domain_size ** 2)
x_grid = np.zeros((1, 1, grid_shape[0], grid_shape[1]))
y_grid = np.zeros((1, 1, grid_shape[0], grid_shape[1]))
vx_grid = np.zeros((1, 1, grid_shape[0], grid_shape[1]))
vy_grid = np.zeros((1, 1, grid_shape[0], grid_shape[1]))
mask = np.zeros((1, 1, grid_shape[0], grid_shape[1]))

'''
xp_s = np.zeros(half_domain_size)
yp_s = np.zeros(half_domain_size)

x_idx = np.round(xp_s/cell_size).astype(int)
y_idx = np.round(yp_s/cell_size).astype(int)

for i in range(1, half_domain_size):
    xp_s [i] = i * 2
    yp_s [i] = i * 2

x_idx = np.round(xp_s/cell_size).astype(int)
y_idx = np.round(yp_s/cell_size).astype(int)


for i, j in zip(x_idx, y_idx):
    x_grid[0, 0, j, i] = i * cell_size
    y_grid[0, 0, j, i] = j * cell_size
    
'''   
# vx_grid[0, 0, 18, 16] = 10
print(x_grid[0,0,:,:])
print(x_grid[0,0,19,19])
print(x_grid[0,0,0,0])
print(x_grid[0,0,19,19])


for i in range(1, half_domain_size-1):
    x_grid[0, 0, domain_size-3*cell_size, i*2] = i * cell_size *2
    y_grid[0, 0, domain_size-3*cell_size, i*2] = (domain_size-3*cell_size) * cell_size
    vx_grid[0, 0, domain_size-3*cell_size, i*2] = random.uniform(-1.0,1.0)
    vy_grid[0, 0, domain_size-3*cell_size, i*2] = 20
    mask[0, 0, domain_size-3*cell_size, i*2] = 1
''''''

mask = np.where(x_grid != 0, 1, 0) 
print('Number of particles:', np.count_nonzero(mask))

# Define the AI4DEM model
class AI4DEM(nn.Module):
    """AI4DEM model for particle interaction detection and force calculation"""
    def __init__(self):
        super(AI4DEM, self).__init__()

    def detector(self, grid, i, j):
        """Detect neighboring particles and calculate the distance between them"""
        diff = grid - torch.roll(grid, shifts=(j - 2, i - 2), dims=(2, 3))
        return diff

    def forward(self, x_grid, y_grid, vx_grid, vy_grid, fx_grid_collision, fy_grid_collision, fx_grid_damping, fy_grid_damping, mask, d, kn, damping_coefficient_Eta, diffx, diffy, diffvx, diffvy, dt, input_shape, filter_size):
        cell_xold = x_grid / d
        cell_yold = y_grid / d   
        cell_xold = torch.round(cell_xold).long()
        cell_yold = torch.round(cell_yold).long()
        # calculate distance between the two particles 
        fx_grid_collision = torch.zeros(input_shape, device=device) 
        fy_grid_collision = torch.zeros(input_shape, device=device) 
        
        fx_grid_damping = torch.zeros(input_shape, device=device) 
        fy_grid_damping = torch.zeros(input_shape, device=device) 
        
        
        for i in range(filter_size):
            for j in range(filter_size):
                diffx = self.detector(x_grid, i, j) # individual
                diffy = self.detector(y_grid, i, j) # individual
                dist = torch.sqrt(diffx**2 + diffy**2)   
                fx_grid_collision =  fx_grid_collision + torch.where(torch.lt(dist,2*d), kn * (dist - 2 * d ) * diffx / torch.maximum(eplis, dist), zeros) # individual
                fy_grid_collision =  fy_grid_collision + torch.where(torch.lt(dist,2*d), kn * (dist - 2 * d ) * diffy / torch.maximum(eplis, dist), zeros) # individual
                
                
                diffvx = self.detector(vx_grid, i, j) # individual
                diffvy = self.detector(vy_grid, i, j) # individual
                
                diffvx_Vn = diffvx * diffx /  torch.maximum(eplis, dist)
                diffvy_Vn = diffvy * diffy /  torch.maximum(eplis, dist)
                
                diffv_Vn = diffvx_Vn + diffvy_Vn
                
                diffv_Vn_x = diffv_Vn * diffx /  torch.maximum(eplis, dist)
                diffv_Vn_y = diffv_Vn * diffy /  torch.maximum(eplis, dist)
                
                fx_grid_damping = fx_grid_damping + torch.where(torch.lt(dist,2*d), damping_coefficient_Eta * diffv_Vn_x, zeros) # individual        
                fy_grid_damping = fy_grid_damping + torch.where(torch.lt(dist,2*d), damping_coefficient_Eta * diffv_Vn_y, zeros) # individual 
                ''' '''

        is_bottom_overlap = torch.ne(y_grid, 0.0000) & torch.lt(y_grid, d) # Overlap with bottom wall
        is_top_overlap = torch.gt(y_grid,domain_size-2*d ) # Overlap with bottom wall      
        is_left_overlap =  torch.ne(x_grid, 0.0000) & torch.lt(x_grid, d) # Overlap with bottom wall
        is_right_overlap = torch.gt(x_grid,domain_size-2*d)# Overlap with bottom wall


        is_bottom_overlap = is_bottom_overlap.to(device)
        is_top_overlap = is_top_overlap.to(device)
        is_left_overlap = is_left_overlap.to(device)
        is_right_overlap = is_right_overlap.to(device)
        
        
        fy_grid_boundary_bottom = kn * torch.where(is_bottom_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))*mask*( d - y_grid )
        fy_grid_boundary_top = kn * torch.where(is_top_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask * (y_grid - domain_size + 2*d)
        fx_grid_boundary_left = kn * torch.where(is_left_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))*mask*(d - x_grid)
        fx_grid_boundary_right = kn * torch.where(is_right_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)) * mask * (x_grid - domain_size + 2*d)
        
        fy_grid_bottom_damping = damping_coefficient_Eta * vy_grid *torch.where(is_bottom_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))*mask
        fy_grid_top_damping = damping_coefficient_Eta * vy_grid *torch.where(is_top_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))*mask
        fx_grid_left_damping = damping_coefficient_Eta * vx_grid *torch.where(is_left_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))*mask
        fx_grid_right_damping = damping_coefficient_Eta * vx_grid *torch.where(is_right_overlap, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))*mask
        
        
        vx_grid = vx_grid  + (dt / particle_mass) * ( fx_grid_boundary_left - fx_grid_boundary_right - fx_grid_collision - fx_grid_damping - fx_grid_left_damping - fx_grid_right_damping) * mask
        vy_grid = vy_grid  + (dt / particle_mass) * ( - 9.8 * particle_mass) * mask  + (dt / particle_mass)* ( -fy_grid_boundary_top + fy_grid_boundary_bottom - fy_grid_collision-fy_grid_damping - fy_grid_bottom_damping - fy_grid_top_damping) * mask 

        
        
        # Update particle coordniates
        x_grid = x_grid + dt * vx_grid
        y_grid = y_grid + dt * vy_grid
        
        x_grid_merge = x_grid.clone()
        y_grid_merge = y_grid.clone()
        vx_grid_merge = vx_grid.clone()
        vy_grid_merge = vy_grid.clone()
        # update new index tensor 
        cell_x = x_grid / d 
        cell_y = y_grid / d     
        cell_x = torch.round(cell_x).long()
        cell_y = torch.round(cell_y).long()    
        # extract index (previous and new) from sparse index tensor (previous and new)
        cell_x = cell_x[cell_x!=0]
        cell_y = cell_y[cell_y!=0]         
        cell_xold = cell_xold[cell_xold!=0]
        cell_yold = cell_yold[cell_yold!=0]   
        # get rid of values at previous index 
        mask[0,0,cell_yold.long(), cell_xold.long()] = 0
        x_grid[0,0,cell_yold.long(),cell_xold.long()] = 0 
        y_grid[0,0,cell_yold.long(),cell_xold.long()] = 0 
        vx_grid[0,0,cell_yold.long(),cell_xold.long()] = 0 
        vy_grid[0,0,cell_yold.long(),cell_xold.long()] = 0 
        # update new values based on new index         
        mask[0,0,cell_y.long(), cell_x.long()] = 1
        x_grid[0,0,cell_y.long(), cell_x.long()] = x_grid_merge[0,0,cell_yold.long(),cell_xold.long()] 
        y_grid[0,0,cell_y.long(), cell_x.long()] = y_grid_merge[0,0,cell_yold.long(),cell_xold.long()] 
        vx_grid[0,0,cell_y.long(), cell_x.long()] = vx_grid_merge[0,0,cell_yold.long(),cell_xold.long()] 
        vy_grid[0,0,cell_y.long(), cell_x.long()] = vy_grid_merge[0,0,cell_yold.long(),cell_xold.long()]

        return x_grid, y_grid, vx_grid, vy_grid, mask

model = AI4DEM().to(device)

# Module 2: Contact detection and force calculation
t = 0
dt = 0.0001  # 0.0001
ntime = 20000000
# Convert np.array into torch.tensor and transfer it to GPU
filter_size = 5 
input_shape_global = (1, 1, grid_shape[0], grid_shape[1])

# Initialize tensors
diffx = torch.zeros(input_shape_global, device=device)
diffy = torch.zeros(input_shape_global, device=device)
diffvx = torch.zeros(input_shape_global, device=device)
diffvy = torch.zeros(input_shape_global, device=device)
zeros = torch.zeros(input_shape_global, device=device)
eplis = torch.ones(input_shape_global, device=device) * 1e-04
fx_grid_collision = torch.zeros(input_shape_global, device=device)
fy_grid_collision = torch.zeros(input_shape_global, device=device)
fx_grid_damping = torch.zeros(input_shape_global, device=device)
fy_grid_damping = torch.zeros(input_shape_global, device=device)
# vx_grid = torch.zeros(input_shape_global, device=device)
# vy_grid = torch.zeros(input_shape_global, device=device)

mask = torch.from_numpy(mask).float().to(device)
x_grid = torch.from_numpy(x_grid).float().to(device)
y_grid = torch.from_numpy(y_grid).float().to(device)

vx_grid = torch.from_numpy(vx_grid).float().to(device)
vy_grid = torch.from_numpy(vy_grid).float().to(device)

# Main simulation loop
start = time.time()
with torch.no_grad():
    for itime in range(1, ntime + 1):
        [x_grid, y_grid, vx_grid, vy_grid, mask] = model(x_grid, y_grid, vx_grid, vy_grid, fx_grid_collision, fy_grid_collision, fx_grid_damping, fy_grid_damping, mask, cell_size, kn, damping_coefficient_Eta, diffx, diffy, diffvx, diffvy, dt, input_shape_global, filter_size)
        print('Time step:', itime, 'Number of particles:', torch.count_nonzero(mask).item()) 
        
        if itime % 2000 == 0:
            
            # Visualize particles
            xp = x_grid[x_grid!=0].cpu() 
            yp = y_grid[y_grid!=0].cpu() 
            
            plt.scatter(xp, yp, c=vy_grid[vy_grid!=0].cpu(), cmap='turbo', s=S_graph, vmin=-200, vmax=200) 
            print(vy_grid[vy_grid!=0])
            cbar = plt.colorbar()
            cbar.set_label('$V_{p}$')
            ax = plt.gca()
            ax.set_xlim([0, domain_size-cell_size])
            ax.set_ylim([0, domain_size-cell_size])
            print(vx_grid[vx_grid!=0].cpu())          
            if itime < 200000:
                for i in range(1, half_domain_size-1):
                    x_grid[0, 0, domain_size-3*cell_size, i*2] = i * cell_size *2
                    y_grid[0, 0, domain_size-3*cell_size, i*2] = (domain_size-3*cell_size) * cell_size
                    vx_grid[0, 0, domain_size-3*cell_size, i*2] = random.uniform(-1.0,1.0)
                    vy_grid[0, 0, domain_size-3*cell_size, i*2] = 20
                    mask[0, 0, domain_size-3*cell_size, i*2] = 1

            # Save visualization
            if itime < 10:
                save_name = "packing/"+str(itime)+".jpg"
            elif itime >= 10 and itime < 100:
                save_name = "packing/"+str(itime)+".jpg"
            elif itime >= 100 and itime < 1000:
                save_name = "packing/"+str(itime)+".jpg"
            elif itime >= 1000 and itime < 10000:
                save_name = "packing/"+str(itime)+".jpg"
            else:
                save_name = "packing/"+str(itime)+".jpg"
            plt.savefig(save_name, dpi=200, bbox_inches='tight')
            plt.close()

end = time.time()
print('Elapsed time:', end - start)
