import torch
from torch.utils.data import TensorDataset, DataLoader
from aurora import AuroraSmallPretrained, Batch, Metadata
from aurora import rollout
import pickle
from huggingface_hub import hf_hub_download

import os 
import numpy as np
from tqdm import tqdm

in_path = '/mnt/data/sonia/aurora-data/date/input-natlantic-multivar-fullcontext/test'
out_path = '/mnt/data/sonia/aurora-out/date/raw-natlantic-multivar-fullcontext/test'
timesteps = 8
batch_size = 2

slp_channel = 0
u_channel = 1
v_channel = 2
t_channel = 3
q_channel = 4

# Initialize the model and load the weights from Hugging Face
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AuroraSmallPretrained().to(device)
model.load_checkpoint() # This will trigger a one-time download of the weights

from datetime import datetime

# Define your grid (example for 0.25 degree resolution)
lats = torch.linspace(90, -90, 721)
lons = torch.linspace(0, 360, 1440 + 1)[:-1]

print('Loading prompts...')
prompts = []
names = []
for sid in tqdm(os.listdir(in_path)):
    prompts.append(np.load(os.path.join(in_path, sid)))
    names.append(sid.split('.')[0])
prompts = np.stack(prompts, axis=0) # N V H W
print(prompts.shape)
print('Prompt loading complete')

# slp from hpa to pa
prompts[:, slp_channel, :, :] *= 100.0

# pressure level configuration 
prompts = np.expand_dims(prompts, 1).repeat(2, axis=1) # N P V H W (P==pressure level)
# prompts[:, 1, slp_channel, :, :] = np.nan #. no pressure level 
prompts[:, 1, u_channel, :, :] = np.nan # 500
prompts[:, 1, v_channel, :, :] = np.nan # 500
prompts[:, 0, t_channel, :, :] = np.nan # 925
prompts[:, 1, q_channel, :, :] = np.nan # 500

prompts_ds = TensorDataset(torch.from_numpy(prompts).float())
prompts_loader = DataLoader(prompts_ds, batch_size=batch_size, shuffle=False)

# Download the official static variables directly from Microsoft's repo
static_path = hf_hub_download(
    repo_id="microsoft/aurora", 
    filename="aurora-0.25-static.pickle" # Fetches the 0.25 degree resolution statics
)
# Load the dictionary
with open(static_path, "rb") as f:
    official_static = pickle.load(f) # lsm, z, slt

preds = []

for batch_prompts in tqdm(prompts_loader, total=len(prompts_loader)):
    batch_prompts = batch_prompts[0] # BS P V H W
    batch_prompts = batch_prompts.cuda()
    # 1. Surface Variables: Shape (Batch, Time, Lat, Lon)
    # You are only using Temperature 2m ("2t")
    surf_vars = {
        "msl": batch_prompts[:, 0, slp_channel, :, :].unsqueeze(1) # add in a time dimension, get rid of pressure dim
    }

    # 2. Atmospheric Variables: Shape (Batch, Time, Levels, Lat, Lon)
    # Levels index: 0 = 500hPa, 1 = 800hPa
    # If a variable doesn't exist at a certain level, you can pass zeros or NaNs depending on your preprocessing
    atmos_vars = {
        "u": batch_prompts[:, :, u_channel, :, :].unsqueeze(1), # U-wind (primarily 500)
        "v": batch_prompts[:, :, v_channel, :, :].unsqueeze(1), # V-wind (primarily 500)
        "t": batch_prompts[:, :, t_channel, :, :].unsqueeze(1), # Temperature (primarily 925)
        "q": batch_prompts[:, :, q_channel, :, :].unsqueeze(1), # Humidity (primarily 500)
    }

    # # 3. Static Variables: Shape (Lat, Lon)
    # # Topography/land-sea mask
    static_vars = {
        "z": torch.as_tensor(official_static["z"]),       
        "lsm": torch.as_tensor(official_static["lsm"]),   
        "slt": torch.as_tensor(official_static["slt"])
    }

    # 4. Pack into the Aurora Batch object -- UNNORMALIZED!
    batch = Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=Metadata(
            lat=lats,
            lon=lons,
            time=(datetime(2020, 6, 1, 12, 0),), # Start time of the storm
            atmos_levels=(500, 925), # Explicitly define your two pressure levels
        ),
    ).to(device)

    batch = batch.normalise(model.surf_stats)

    # Run the model autoregressively for your 8 timesteps
    with torch.inference_mode():
        # We move predictions to CPU immediately to prevent GPU memory overflow during rollout
        batch_preds = [pred.to("cpu") for pred in rollout(model, batch, steps=timesteps)]
        
    batch_preds = [pred.unnormalise(model.surf_stats) for pred in batch_preds]
    slp_pred = torch.cat([t.surf_vars['msl'] for t in batch_preds], dim=1).squeeze() # B T H W
    u_pred = torch.cat([t.atmos_vars['u'] for t in batch_preds], dim=1)[:, :, 0, :, :] # get rid of pressure lvl
    v_pred = torch.cat([t.atmos_vars['v'] for t in batch_preds], dim=1)[:, :, 0, :, :]
    t_pred = torch.cat([t.atmos_vars['t'] for t in batch_preds], dim=1)[:, :, 1, :, :]
    q_pred = torch.cat([t.atmos_vars['q'] for t in batch_preds], dim=1)[:, :, 0, :, :]
    
    preds.append(torch.stack([slp_pred, u_pred, v_pred, t_pred, q_pred], dim=2)) # appending: B T V H W
    
    
preds = torch.cat(preds) # N T V H W
print(preds.shape)

# slp from hpa to pa
preds[:, :, slp_channel, :, :] /= 100.0
preds = preds.permute(0, 1, 3, 4, 2) # N T H W V
print(preds.shape)
preds = preds.numpy()

print('Writing out predictions...')
for i, name in enumerate(names):
    os.makedirs(os.path.join(out_path, name), exist_ok=True)
    for t in range(timesteps):
        np.save(os.path.join(out_path, name, f'{t}'), preds[i, t])
print('Prediction writing complete')
