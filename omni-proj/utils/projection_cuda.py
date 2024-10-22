import torch
import numpy as np

def rodrigues_torch(rvec):
    theta = torch.norm(rvec)
    if theta < 1e-6:
        return torch.eye(3, device=rvec.device)

    rvec = rvec / theta
    K = torch.tensor([[0, -rvec[2], rvec[1]],
                      [rvec[2], 0, -rvec[0]],
                      [-rvec[1], rvec[0], 0]], device=rvec.device)
    R = torch.eye(3, device=rvec.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(K, K)
    return R

# gridy2x
def gridy2x_fis2erp(gridy, HWy, HWx, THETA, PHI, FOVy, FOVx, device='cuda'):
    H, W, h, w = *HWy, *HWx
    hFOVy, wFOVy = FOVy * float(H) / W, FOVy
    hFOVx, wFOVx = FOVx * float(h) / w, FOVx

    # gridy2x
    ### onto sphere
    gridy = gridy.reshape(-1, 2).float()
    lat = gridy[:, 0] * np.pi / 2
    lon = gridy[:, 1] * np.pi

    z0 = torch.sin(lat)
    y0 = torch.cos(lat) * torch.sin(lon)
    x0 = torch.cos(lat) * torch.cos(lon)
    gridy = torch.stack((x0, y0, z0), dim=-1).double()

    ### rotation
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float64)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float64)
    R1 = rodrigues_torch(z_axis * np.radians(THETA))
    R2 = rodrigues_torch(torch.matmul(R1, y_axis) * np.radians(PHI))

    R1_inv = torch.inverse(R1)
    R2_inv = torch.inverse(R2)

    gridy = torch.mm(R2_inv, gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(R1_inv, gridy.permute(1, 0)).permute(1, 0)

    ### sphere to gridx
    r = torch.arccos(gridy[:, 0])
    norm = torch.norm(gridy[:, 1:], p=2, dim=-1)
    z0 = gridy[:, 2] * r / norm / np.radians(hFOVx / 2.0)
    y0 = gridy[:, 1] * r / norm / np.radians(wFOVx / 2.0)
    gridx = torch.stack((z0, y0), dim=-1)
    
    # masky
    dist = torch.norm(gridx, p=2, dim=-1)
    mask = torch.where(dist > 1, 0, 1)

    return gridx.to(torch.float32), mask.to(torch.float32)

def gridy2x_per2erp(gridy, HWy, HWx, THETA, PHI, FOVy, FOVx, device='cuda'):
    H, W, h, w = *HWy, *HWx
    hFOVy, wFOVy = FOVy * float(H) / W, FOVy
    hFOVx, wFOVx = FOVx * float(h) / w, FOVx

    # gridy2x
    ### onto sphere
    gridy = gridy.reshape(-1, 2).float()
    lat = gridy[:, 0] * np.pi / 2
    lon = gridy[:, 1] * np.pi

    z0 = torch.sin(lat)
    y0 = torch.cos(lat) * torch.sin(lon)
    x0 = torch.cos(lat) * torch.cos(lon)
    gridy = torch.stack((x0, y0, z0), dim=-1).double()

    ### rotation
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float64)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float64)
    R1 = rodrigues_torch(z_axis * np.radians(THETA))
    R2 = rodrigues_torch(torch.matmul(R1, y_axis) * np.radians(PHI))

    R1_inv = torch.inverse(R1)
    R2_inv = torch.inverse(R2)

    gridy = torch.mm(R2_inv, gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(R1_inv, gridy.permute(1, 0)).permute(1, 0)

    ### sphere to gridx
    z0 = gridy[:, 2] / gridy[:, 0]
    y0 = gridy[:, 1] / gridy[:, 0]
    gridx = torch.stack((z0, y0), dim=-1).float()

    # masky
    mask = torch.where(torch.abs(gridx) > 1, 0, 1)
    mask = mask[:, 0] * mask[:, 1]
    mask *= torch.where(gridy[:, 0] < 0, 0, 1)

    return gridx.to(torch.float32), mask.to(torch.float32)

def gridy2x_erp2fis(gridy, HWy, HWx, THETA, PHI, FOVy, FOVx, device='cuda'):
    H, W, h, w = *HWy, *HWx
    hFOVy, wFOVy = FOVy * float(H) / W, FOVy
    hFOVx, wFOVx = FOVx * float(h) / w, FOVx
    
    # gridy2x
    ### onto sphere
    gridy = gridy.reshape(-1, 2).float()
    dist = torch.norm(gridy, p=2, dim=-1)
    gridy[:, 0] *= np.radians(hFOVy / 2.0)
    gridy[:, 1] *= np.radians(wFOVy / 2.0)
    gridy = gridy.double().flip(-1)

    r = torch.norm(gridy, p=2, dim=-1, keepdim=True)
    x0 = torch.cos(r)
    gridy *= torch.sqrt(1 - x0**2) / r
    gridy = torch.cat((x0 , gridy), dim=-1)
    
    ### rotation
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float64)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float64)
    R1 = rodrigues_torch(z_axis * np.radians(THETA))
    R2 = rodrigues_torch(torch.matmul(R1, y_axis) * np.radians(PHI))

    gridy = torch.mm(R1, gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(R2, gridy.permute(1, 0)).permute(1, 0)

    ### sphere to gridx
    lat = torch.arcsin(gridy[:, 2].clamp_(-1+1e-6, 1-1e-6)) / np.pi * 2
    lon = torch.atan2(gridy[:, 1], gridy[:, 0]) / np.pi
    gridx = torch.stack((lat, lon), dim=-1)

    # masky
    mask = torch.where(dist > 1, 0, 1)

    return gridx.to(torch.float32), mask.to(torch.float32)

def gridy2x_per2fis(gridy, HWy, HWx, THETA, PHI, FOVy, FOVx, device='cuda'):
    H, W, h, w = *HWy, *HWx
    hFOVy, wFOVy = FOVy * float(H) / W, FOVy
    hFOVx, wFOVx = FOVx * float(h) / w, FOVx

    # gridy2x
    ### onto sphere
    gridy = gridy.reshape(-1, 2).float()
    dist = torch.norm(gridy, p=2, dim=-1)
    gridy[:, 0] *= np.radians(hFOVy / 2.0)
    gridy[:, 1] *= np.radians(wFOVy / 2.0)
    gridy = gridy.double().flip(-1)

    r = torch.norm(gridy, p=2, dim=-1, keepdim=True)
    x0 = torch.cos(r)
    gridy *= torch.sqrt(1 - x0**2) / r
    gridy = torch.cat((x0 , gridy), dim=-1)

    ### rotation
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float64)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float64)
    R1 = rodrigues_torch(z_axis * np.radians(THETA))
    R2 = rodrigues_torch(torch.matmul(R1, y_axis) * np.radians(PHI))

    R1_inv = torch.inverse(R1)
    R2_inv = torch.inverse(R2)

    gridy = torch.mm(R2_inv, gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(R1_inv, gridy.permute(1, 0)).permute(1, 0)

    ### sphere to gridx
    z0 = gridy[:, 2] / gridy[:, 0]
    y0 = gridy[:, 1] / gridy[:, 0]
    gridx = torch.stack((z0, y0), dim=-1).float()

    # masky
    mask = torch.where(torch.abs(gridx) > 1, 0, 1)
    mask = mask[:, 0] * mask[:, 1]
    mask *= torch.where(gridy[:, 0] < 0, 0, 1)
    mask *= torch.where(dist > 1, 0, 1)

    return gridx.to(torch.float32), mask.to(torch.float32)

def gridy2x_erp2per(gridy, HWy, HWx, THETA, PHI, FOVy, FOVx, device='cuda'):
    H, W, h, w = *HWy, *HWx
    hFOVy, wFOVy = FOVy * float(H) / W, FOVy
    hFOVx, wFOVx = FOVx * float(h) / w, FOVx
    
    # gridy2x
    ### onto sphere
    gridy = gridy.reshape(-1, 2).float()
    gridy[:, 0] *= np.tan(np.radians(hFOVy / 2.0))
    gridy[:, 1] *= np.tan(np.radians(wFOVy / 2.0))
    gridy = gridy.double().flip(-1)
    
    x0 = torch.ones(gridy.shape[0], 1, device=device)
    gridy = torch.cat((x0, gridy), dim=-1)
    gridy /= torch.norm(gridy, p=2, dim=-1, keepdim=True)
    
    ### rotation
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float64)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float64)
    R1 = rodrigues_torch(z_axis * np.radians(THETA))
    R2 = rodrigues_torch(torch.matmul(R1, y_axis) * np.radians(PHI))

    gridy = torch.mm(R1, gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(R2, gridy.permute(1, 0)).permute(1, 0)

    ### sphere to gridx
    lat = torch.arcsin(gridy[:, 2]) / np.pi * 2
    lon = torch.atan2(gridy[:, 1] , gridy[:, 0]) / np.pi
    gridx = torch.stack((lat, lon), dim=-1)

    # masky
    mask = torch.where(torch.abs(gridx) > 1, 0, 1)
    mask = mask[:, 0] * mask[:, 1]

    return gridx.to(torch.float32), mask.to(torch.float32)

def gridy2x_fis2per(gridy, HWy, HWx, THETA, PHI, FOVy, FOVx, device='cuda'):
    H, W, h, w = *HWy, *HWx
    hFOVy, wFOVy = FOVy * float(H) / W, FOVy
    hFOVx, wFOVx = FOVx * float(h) / w, FOVx

    # gridy2x
    ### onto sphere
    gridy = gridy.reshape(-1, 2).float()
    gridy[:, 0] *= np.tan(np.radians(hFOVy / 2.0))
    gridy[:, 1] *= np.tan(np.radians(wFOVy / 2.0))
    gridy = gridy.double().flip(-1)
    
    x0 = torch.ones(gridy.shape[0], 1, device=device)
    gridy = torch.cat((x0, gridy), dim=-1)
    gridy /= torch.norm(gridy, p=2, dim=-1, keepdim=True)
    
    ### rotation
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float64)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float64)
    R1 = rodrigues_torch(z_axis * np.radians(THETA))
    R2 = rodrigues_torch(torch.matmul(R1, y_axis) * np.radians(PHI))

    gridy = torch.mm(R1, gridy.permute(1, 0)).permute(1, 0)
    gridy = torch.mm(R2, gridy.permute(1, 0)).permute(1, 0)
    
    ### sphere to gridx
    r = torch.arccos(gridy[:, 0])
    norm = torch.norm(gridy[:, 1:], p=2, dim=-1)
    z0 = gridy[:, 2] * r / norm / np.radians(hFOVx / 2.0)
    y0 = gridy[:, 1] * r / norm / np.radians(wFOVx / 2.0)
    gridx = torch.stack((z0, y0), dim=-1)
    
    # masky
    dist = torch.norm(gridx, p=2, dim=-1)
    mask = torch.where(dist > 1, 0, 1)

    return gridx.to(torch.float32), mask.to(torch.float32)

# celly2x
def celly2x_fis2erp(celly, HWy, HWx, THETA, PHI, FOVy, FOVx, device='cuda'):
    cellx, _ = gridy2x_fis2erp(celly, HWy, HWx, THETA, PHI, FOVy, FOVx)
    return shape_estimation(cellx)

def celly2x_per2erp(celly, HWy, HWx, THETA, PHI, FOVy, FOVx, device='cuda'):
    cellx, _ = gridy2x_per2erp(celly, HWy, HWx, THETA, PHI, FOVy, FOVx, device)
    return shape_estimation(cellx)

def celly2x_erp2fis(celly, HWy, HWx, THETA, PHI, FOVy, FOVx, device='cuda'):
    cellx, _ = gridy2x_erp2fis(celly, HWy, HWx, THETA, PHI, FOVy, FOVx, device)
    return shape_estimation(cellx)

def celly2x_per2fis(celly, HWy, HWx, THETA, PHI, FOVy, FOVx, device='cuda'):
    cellx, _ = gridy2x_per2fis(celly, HWy, HWx, THETA, PHI, FOVy, FOVx, device)
    return shape_estimation(cellx)

def celly2x_erp2per(celly, HWy, HWx, THETA, PHI, FOVy, FOVx, device='cuda'):
    cellx, _ = gridy2x_erp2per(celly, HWy, HWx, THETA, PHI, FOVy, FOVx, device)
    return shape_estimation(cellx)

def celly2x_fis2per(celly, HWy, HWx, THETA, PHI, FOVy, FOVx, device='cuda'):
    cellx, _ = gridy2x_fis2per(celly, HWy, HWx, THETA, PHI, FOVy, FOVx, device)
    return shape_estimation(cellx)

def shape_estimation(cell):
    # Jacobian
    cell_1 = cell[7*cell.shape[0]//9:8*cell.shape[0]//9, :]\
           - cell[6*cell.shape[0]//9:7*cell.shape[0]//9, :]

    cell_2 = cell[5*cell.shape[0]//9:6*cell.shape[0]//9, :]\
           - cell[4*cell.shape[0]//9:5*cell.shape[0]//9, :]
    
    # Second-order derivatives in Hessian
    cell_3 = cell[7*cell.shape[0]//9:8*cell.shape[0]//9, :]\
           + cell[6*cell.shape[0]//9:7*cell.shape[0]//9, :]\
           - cell[8*cell.shape[0]//9:9*cell.shape[0]//9, :] * 2

    cell_4 = cell[5*cell.shape[0]//9:6*cell.shape[0]//9, :]\
           + cell[4*cell.shape[0]//9:5*cell.shape[0]//9, :]\
           - cell[8*cell.shape[0]//9:9*cell.shape[0]//9, :] * 2
           
    # Cross-term in Hessian
    cell_5 = cell[3*cell.shape[0]//9:4*cell.shape[0]//9, :]\
           - cell[0*cell.shape[0]//9:1*cell.shape[0]//9, :]\
           - cell[1*cell.shape[0]//9:2*cell.shape[0]//9, :]\
           + cell[0*cell.shape[0]//9:1*cell.shape[0]//9, :]\
           - cell[2*cell.shape[0]//9:3*cell.shape[0]//9, :]\
           + cell[0*cell.shape[0]//9:1*cell.shape[0]//9, :] 

    # cat[yaxis (q, 5), xaxis (q, 5)]
    shape = torch.cat((cell_1, cell_2, 4*cell_3, 4*cell_4, cell_5), dim=-1)
    return shape
