from pathlib import Path
import numpy as np, pandas as pd

rng=np.random.default_rng(42)
N=10000
users=np.array([f"u_{i:05d}" for i in range(2500)])
devices=np.array([f"d_{i:04d}" for i in range(1800)])
ips=np.array([f"ip_{i:04d}" for i in range(2200)])
merchants=np.array([f"m_{i:04d}" for i in range(300)])
t=pd.date_range("2025-01-01", periods=N, freq="15min")
df=pd.DataFrame({"transaction_id":[f"tx_{i:06d}" for i in range(N)],"timestamp":t,"user_id":rng.choice(users,N),"device_id":rng.choice(devices,N),"ip_id":rng.choice(ips,N),"merchant_id":rng.choice(merchants,N),"amount":np.round(rng.lognormal(5,1,N),2),"label":0})

# Synthetic coordinated rings: shared device/IP, burst timing, repeated merchants.
# Amounts are mixed across three bands rather than one fixed range, so
# the model has to learn identity-sharing as the fraud signal instead
# of keying off a single narrow amount band.
RING_AMOUNT_PROFILES = [
    (20, 150),      # low-value structuring / card-testing
    (450, 1200),    # mid-range
    (1500, 5000),   # high-value
]

for ring in range(25):
    ix=rng.choice(N, size=18, replace=False); u=f"ring_u_{ring}"; d=f"ring_d_{ring}"; ip=f"ring_ip_{ring}"
    low, high = RING_AMOUNT_PROFILES[ring % len(RING_AMOUNT_PROFILES)]
    df.loc[ix,"user_id"]=[f"{u}_{j}" for j in range(18)]; df.loc[ix,"device_id"]=d; df.loc[ix,"ip_id"]=ip; df.loc[ix,"merchant_id"]=f"ring_m_{ring%8}"; df.loc[ix,"amount"]=np.round(rng.uniform(low,high,len(ix)),2); df.loc[ix,"label"]=1
df.sort_values("timestamp").to_csv(Path("data/transactions.csv"),index=False)
