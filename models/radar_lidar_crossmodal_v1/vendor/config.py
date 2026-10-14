import os
import numpy as np

# --- PATH ---
DATA_ROOT = "/media/gandalf22/Data_storage/Davide/VoD/view_of_delft_PUBLIC"
BASE_OUTPUT_ROOT = "/media/gandalf22/Data_storage/Davide/Models/Diffusion_Model/AutoEncoder_ground"
PROCESSED_DATA_PATH = os.path.join(BASE_OUTPUT_ROOT, "dataset_10cm")


POINT_CLOUD_RANGE = [-51.2, -51.2, -2.0, 51.2, 51.2, 2.0]

# Voxel Size:
# Z = 0.5 (50cm) -> 8 Voxel totali in altezza.
# XY = 0.1 (10cm) -> Alta risoluzione orizzontale.
VOXEL_SIZE = [0.1, 0.1, 0.5] 

# Grid Size Atteso: [1024, 1024, 32]
# 1024 è 2^10 -> Perfetto per downsampling profondo (fino a 6 volte)
GRID_SIZE = [
    int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0]),
    int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1]),
    int((POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2]) / VOXEL_SIZE[2])
]

# --- MODEL PARAMETERS ---
MAX_POINTS_PER_VOXEL = 200    
MAX_VOXELS = 160000 
LATENT_DIM = 64    
