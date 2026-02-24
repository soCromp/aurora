import torch
from aurora import AuroraSmallPretrained, Batch, Metadata
from aurora import rollout

# Initialize the model and load the weights from Hugging Face
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AuroraSmallPretrained().to(device)
model.load_checkpoint() # This will trigger a one-time download of the weights

from datetime import datetime

# Define your grid (example for 0.25 degree resolution)
lats = torch.linspace(90, -90, 721)
lons = torch.linspace(0, 360, 1440 + 1)[:-1]

## TODO: read in data, convert SLP to pascals from hectopascals

# 1. Surface Variables: Shape (Batch, Time, Lat, Lon)
# You are only using Temperature 2m ("2t")
surf_vars = {
    "msl": torch.randn(1, 1, 721, 1440) # Replace with your actual SLP data in PASCALS
}

# 2. Atmospheric Variables: Shape (Batch, Time, Levels, Lat, Lon)
# Levels index: 0 = 500hPa, 1 = 800hPa
# If a variable doesn't exist at a certain level, you can pass zeros or NaNs depending on your preprocessing
atmos_vars = {
    "u": torch.randn(1, 1, 2, 721, 1440), # U-wind (primarily 500)
    "v": torch.randn(1, 1, 2, 721, 1440), # V-wind (primarily 500)
    "t": torch.randn(1, 1, 2, 721, 1440), # Temperature (primarily 925)
    "q": torch.randn(1, 1, 2, 721, 1440), # Humidity (primarily 500)
}

# 3. Static Variables: Shape (Lat, Lon)
# Topography/land-sea mask
static_vars = {
    "z": torch.randn(721, 1440),   # Surface geopotential (from your topo.nc)
    # "lsm": torch.randn(721, 1440), # Land-sea mask
    # "slt": torch.randn(721, 1440)  # Soil type (optional, can be mocked if missing)
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
    preds = [pred.to("cpu") for pred in rollout(model, batch, steps=8)]
    
preds = [pred.unnormalise(model.surf_stats) for pred in preds]
print(preds)
