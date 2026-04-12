import torch
from torch.utils.data import TensorDataset, DataLoader
from aurora import AuroraSmallPretrained, Batch, Metadata
from aurora import rollout
import pickle
from huggingface_hub import hf_hub_download

import os 
import numpy as np
from tqdm import tqdm
import pandas as pd

in_path = '/mnt/data/sonia/aurora-data/date/input-natlantic-multivar-fullcontext/test'
out_path = '/mnt/data/sonia/aurora-out/date/raw-natlantic-multivar-fullcontext/test'
timesteps = 8
batch_size = 2

slp_channel = 0
u_channel = 1
v_channel = 2
t_channel = 3
q_channel = 4

#### Load tracks data
trackspath1='/home/sonia/mcms/tracker/1940-2010/era5/out_era5/era5/mcms_era5_1940_2010_tracks.txt'
trackspath2='/home/sonia/mcms/tracker/2010-2024/era5/out_era5/era5/FIXEDmcms_era5_2010_2024_tracks.txt'
joinyear = 2010 # overlap for the track data
start_year = 1940 #inclusive
stop_year = 2024 #inclusive

tracks1 = pd.read_csv(trackspath1, sep=' ', header=None, 
        names=['year', 'month', 'day', 'hour', 'total_hrs', 'unk1', 'unk2', 'unk3', 'unk4', 'unk5', 'unk6', 
               'z1', 'z2', 'unk7', 'tid', 'sid'])
# storms that start before the join year (even if they continue into the join year):
sids1 = tracks1[(tracks1['sid']==tracks1['tid']) & (tracks1['year']<joinyear)]['sid'].unique()
tracks1 = tracks1[tracks1['sid'].isin(sids1)]

tracks2 = pd.read_csv(trackspath2, sep=' ', header=None, 
        names=['year', 'month', 'day', 'hour', 'total_hrs', 'unk1', 'unk2', 'unk3', 'unk4', 'unk5', 'unk6', 
               'z1', 'z2', 'unk7', 'tid', 'sid'])
# filter out storms that "start" at the beginning of the join year since they probably started before and are 
# included in tracks1
sids2 = tracks2[(tracks2['sid']==tracks2['tid']) & \
        ((tracks2['year']>=joinyear) | (tracks2['month']>1) | (tracks2['day']>1) | (tracks2['hour']>0))]['sid'].unique()
tracks2 = tracks2[tracks2['sid'].isin(sids2)]

tracks = pd.concat([tracks1, tracks2], ignore_index=True)
tracks = tracks[tracks['year']>=start_year]
tracks = tracks.sort_values(by=['year', 'month', 'day', 'hour'])

# conversions from the MCMS lat/lon system, as described in Jimmy's email:
tracks['lat'] = 90-tracks['unk1'].values/100
tracks['lon'] = tracks['unk2'].values/100

tracks = tracks[['year', 'month', 'day', 'hour', 'tid', 'sid', 'lat', 'lon']]


#### Setup

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
for sid in tqdm(sorted(os.listdir(in_path))):
    prompts.append(np.load(os.path.join(in_path, sid)))
    names.append(sid.split('.')[0])
prompts = np.stack(prompts, axis=0) # N V H W
print(prompts.shape)
print('Prompt loading complete')

# slp from hpa to pa
prompts[:, slp_channel, :, :] *= 100.0

# pressure level configuration 
prompts = np.expand_dims(prompts, 1) # N P V H W (P==pressure level)

prompts_ds = TensorDataset(torch.from_numpy(prompts).float(), names)
prompts_loader = DataLoader(prompts_ds, batch_size=batch_size, shuffle=False)

# Download the official static variables directly from Microsoft's repo
static_path = hf_hub_download(
    repo_id="microsoft/aurora", 
    filename="aurora-0.25-static.pickle" # Fetches the 0.25 degree resolution statics
)
# Load the dictionary
with open(static_path, "rb") as f:
    official_static = pickle.load(f) # lsm, z, slt


for batch_prompts, batch_sids in tqdm(prompts_loader, total=len(prompts_loader)):
    # prompts: BS P V H W
    batch_prompts = batch_prompts.float().cuda()
    # 1. Surface Variables: Shape (Batch, Time, Lat, Lon)
    # You are only using Temperature 2m ("2t")
    surf_vars = {
        "msl": batch_prompts[:, 0, slp_channel, :, :].unsqueeze(1), # add in a time dimension, get rid of pressure dim
        "2t":  batch_prompts[:, 0, t_channel, :, :].unsqueeze(1), # pass 925 as surface
    }

    # 2. Atmospheric Variables: Shape (Batch, Time, Levels, Lat, Lon)
    # Levels index: 0 = 500hPa, 1 = 800hPa
    # If a variable doesn't exist at a certain level, you can pass zeros or NaNs depending on your preprocessing
    atmos_vars = {
        "u": batch_prompts[:, :, u_channel, :, :].unsqueeze(1), # U-wind (primarily 500)
        "v": batch_prompts[:, :, v_channel, :, :].unsqueeze(1), # V-wind (primarily 500)
        # "t": batch_prompts[:, :, t_channel, :, :].unsqueeze(1), # Temperature (primarily 925)
        "q": batch_prompts[:, :, q_channel, :, :].unsqueeze(1), # Humidity (primarily 500)
    }

    # # 3. Static Variables: Shape (Lat, Lon)
    # # Topography/land-sea mask
    static_vars = {
        "z": torch.as_tensor(official_static["z"]),       
        "lsm": torch.as_tensor(official_static["lsm"]),   
        "slt": torch.as_tensor(official_static["slt"])
    }
    
    rows = [tracks[tracks['tid']==int(sid)].to_dict(orient='records')[0] for sid in batch_sids]
    dates = [datetime(row['year'], row['month'], row['day'], row['hour'], 0) for row in rows]

    # 4. Pack into the Aurora Batch object -- UNNORMALIZED!
    batch = Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=Metadata(
            lat=lats,
            lon=lons,
            time=dates,
            atmos_levels=(500,), 
        ),
    ).to(device)

    batch = batch.normalise(model.surf_stats)

    # Run the model autoregressively for your 8 timesteps
    with torch.inference_mode():
        # We move predictions to CPU immediately to prevent GPU memory overflow during rollout
        batch_preds = [pred.to("cpu") for pred in rollout(model, batch, steps=timesteps)]
        
    # batch_preds = [pred.unnormalise(model.surf_stats) for pred in batch_preds]
    slp_pred = torch.cat([t.surf_vars['msl'] for t in batch_preds], dim=1) # B T H W
    u_pred = torch.cat([t.atmos_vars['u'] for t in batch_preds], dim=1)[:, :, 0, :, :] # get rid of pressure lvl
    v_pred = torch.cat([t.atmos_vars['v'] for t in batch_preds], dim=1)[:, :, 0, :, :]
    t_pred = torch.cat([t.surf_vars['2t'] for t in batch_preds], dim=1)
    q_pred = torch.cat([t.atmos_vars['q'] for t in batch_preds], dim=1)[:, :, 0, :, :]
    
    batch_preds = torch.stack([slp_pred, u_pred, v_pred, t_pred, q_pred], dim=2).float() # B T V H W
    batch_preds[:, :, slp_channel, :, :] /= 100.0
    batch_preds = batch_preds.permute(0, 1, 3, 4, 2) # N T H W V
    batch_preds = batch_preds.numpy()
    # print(batch_preds)
    for i, sid in enumerate(batch_sids):
        os.makedirs(os.path.join(out_path, sid), exist_ok=True)
        for t in range(timesteps):
            np.save(os.path.join(out_path, sid, f'{t}'), batch_preds[i, t])
    
    
# preds = torch.cat(preds) # N T V H W
# print(preds.shape)

# # slp from hpa to pa
# preds[:, :, slp_channel, :, :] /= 100.0
# preds = preds.permute(0, 1, 3, 4, 2) # N T H W V
# print(preds.shape)
# preds = preds.numpy()

# print('Writing out predictions...')
# for i, name in enumerate(names):
#     os.makedirs(os.path.join(out_path, name), exist_ok=True)
#     for t in range(timesteps):
#         np.save(os.path.join(out_path, name, f'{t}'), preds[i, t])
# print('Prediction writing complete')
