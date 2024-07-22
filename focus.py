"""Optimize a focusing model"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch

import spintorch
import spintorch.geom
import numpy as np
from spintorch.utils import tic, toc
from spintorch.plot import wave_integrated, wave_snapshot
from numpy import pi

import warnings
warnings.filterwarnings("ignore", message=".*Casting complex values to real.*")
torch.set_default_tensor_type('torch.FloatTensor')

"""Parameters"""
dx = 50e-9      # discretization (m)
dy = 50e-9      # discretization (m)
dz = 20e-9      # discretization (m)
nx = 100        # size x    (cells)
ny = 100       # size y    (cells)

Ms = 140e3      # saturation magnetization (A/m)
B0 = 60e-3      # bias field (T)
B0_theta = 0      # Angle of bias field in randian, e.g.: pi/4
Bt = 1e-3       # excitation field amplitude (T)

dt = 20e-12     # timestep (s)
f1 = 4e9        # source frequency (Hz)
timesteps = 1500 # number of timesteps for wave propagation


'''Directories'''
basedir = 'M0/'
plotdir = 'plots/' + basedir
if not os.path.isdir(plotdir):
    os.makedirs(plotdir)
savedir = 'models/' + basedir
if not os.path.isdir(savedir):
    os.makedirs(savedir)    

'''Geometry, sources, probes, model definitions'''
dev = torch.device('cpu')  # 'cuda' or 'cpu'
print('Running on', dev)
### Here are three geometry modules initialized, just uncomment one of them to try:
Ms_CoPt = 723e3 # saturation magnetization of the nanomagnets (A/m)
# r0_x, r0_y, dr, dm, z_off = 20, 20, 4, 2, 10  # starting pos, period, magnet size, z distance
# rx, ry = 4,4

r0_x, r0_y, dr, dm, z_off = 10, 15, 4, 2, 10  # starting pos, period, magnet size, z distance
rx, ry = 20,18

rho = torch.zeros((rx, ry))  # Design parameter array

## Define your mask here

mask = np.zeros((nx,ny))+0.25
# for i in range(nx):
#     for j in range(ny):
#         if j>20 and j<34:
#             mask[i][j]=1
#         if i>20 and i<34:
#             mask[i][j]=1

# right Vertical
for i in range(nx-24, nx-10):  # 假设ny是上限
    for j in range(22, ny-10):
        mask[i][j] = 1

# left Vertical
for i in range(nx-80, nx-66):  # 假设ny是上限
    for j in range(22, ny-10):
        mask[i][j] = 1

# top Horizontal
for i in range(22, nx-10):
    for j in range(ny-24, ny-10):  # 假设nx是上限
        mask[i][j] = 1

# bottom horizontal
for i in range(10, nx-10):
    for j in range(ny-80, ny-66):  # 假设nx是上限
        mask[i][j] = 1

# Apply the mask to the file
Msat = Ms*mask
Msat = Msat.astype(np.float32)
geom = spintorch.geom.WaveGeometryArray_Ms(rho, (nx, ny), (dx, dy, dz), Msat, B0, B0_theta, 
                                    r0_x, r0_y, dr, dm, z_off, rx, ry, Ms_CoPt,mask)
#geom = spintorch.geom.WaveGeometryArray_Ms(rho, (nx, ny), (dx, dy, dz), Ms, Msat, B0, 
                                    #r0, dr, dm, z_off, rx, ry, Ms_CoPt)
 




src = spintorch.WaveLineSource(10, 20, 10, 34, dim=2)
probes = []


# unwanted output
probes.append(spintorch.WaveIntensityProbeDisk(76, 31, 2))
probes.append(spintorch.WaveIntensityProbeDisk(76, 27, 2))
probes.append(spintorch.WaveIntensityProbeDisk(76, 23, 2))

# Desire output
# Top Left
probes.append(spintorch.WaveIntensityProbeDisk(23, 85, 2))
probes.append(spintorch.WaveIntensityProbeDisk(27, 85, 2))
probes.append(spintorch.WaveIntensityProbeDisk(31, 85, 2))
#Top Right
probes.append(spintorch.WaveIntensityProbeDisk(65, 86, 2))
probes.append(spintorch.WaveIntensityProbeDisk(65, 82, 2))
probes.append(spintorch.WaveIntensityProbeDisk(65, 78, 2))
#Lower Right
probes.append(spintorch.WaveIntensityProbeDisk(78, 60, 2))
probes.append(spintorch.WaveIntensityProbeDisk(82, 60, 2))
probes.append(spintorch.WaveIntensityProbeDisk(86, 60, 2))

model = spintorch.MMSolver(geom, dt, [src], probes)


model.to(dev)   # sending model to GPU/CPU


'''Define the source signal and output goal'''
t = torch.arange(0, timesteps*dt, dt, device=dev).unsqueeze(0).unsqueeze(2) # time vector
X = Bt*torch.sin(2*np.pi*f1*t)  # sinusoid signal at f1 frequency, Bt amplitude

INPUTS = X  # here we could cat multiple inputs

OUTPUTS = torch.tensor([3,4,5,6,7,8,9,10,11]).to(dev) # desired output

'''Define optimizer and lossfunction'''
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# def my_loss(output, target_index):
#     target_value = output[:,target_index]
#     loss = output.sum(dim=1)/target_value-1
#     return (loss.sum()/loss.size()[0]).log10()

def my_loss(output, target_index):
        a = torch.tensor([0,1,2]).to(dev) # Unwanted output
        b = torch.tensor([3,4,5]).to(dev)  # Desire output uppper left
        c = torch.tensor([6,7,8]).to(dev)  # Desire output upper right
        d = torch.tensor([9,10,11]).to(dev)  # Desire output lower right
        probea = output[:,a]
        probeb = output[:,b]
        probec = output[:,c]
        probed = output[:,d]  
        loss = torch.sum(probea)/(torch.sum(probeb)+torch.sum(probec)+torch.sum(probed))
        return loss

'''Load checkpoint'''
epoch = epoch_init = -1 # select previous checkpoint (-1 = don't use checkpoint)
if epoch_init>=0:
    checkpoint = torch.load(savedir + 'model_e%d.pt' % (epoch_init))
    epoch = checkpoint['epoch']
    loss_iter = checkpoint['loss_iter']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    loss_iter = []

'''Train the network'''
tic()
model.retain_history = True
epochs = 10
for epoch in range(epoch_init+1, epochs):
    optimizer.zero_grad()
    u = model(INPUTS).sum(dim=1) # integral!!!!!
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    
    
    spintorch.plot.plot_output(u[0,], OUTPUTS[0]+1, epoch, plotdir)
    loss = my_loss(u,OUTPUTS)
    loss_iter.append(loss.item())  # store loss values
    spintorch.plot.plot_loss(loss_iter, plotdir)
    # stat_cuda('after forward')
    loss.backward()
    optimizer.step()
    # stat_cuda('after backward')
    print("Epoch finished: %d -- Loss: %.6f" % (epoch, loss))
    toc()   

    '''Save model checkpoint'''
    torch.save({
                'epoch': epoch,
                'loss_iter': loss_iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, savedir + 'model_e%d.pt' % (epoch))
    
    '''Plot spin-wave propagation'''
    if model.retain_history:
        with torch.no_grad():
            spintorch.plot.geometry(model, epoch=epoch, plotdir=plotdir)
            mz = torch.stack(model.m_history, 1)[0,:,2,]-model.m0[0,2,].unsqueeze(0).cpu()
            wave_snapshot(model, mz[timesteps-1], (plotdir+'snapshot_time%d_epoch%d.png' % (timesteps,epoch)),r"$m_z$")
            wave_snapshot(model, mz[int(timesteps/2)-1], (plotdir+'snapshot_time%d_epoch%d.png' % (int(timesteps/2),epoch)),r"$m_z$")
            wave_integrated(model, mz, (plotdir+'integrated_epoch%d.png' % (epoch)))


  