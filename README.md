# totalenergies_electricity_data

**Common abbreviations:**
- SPP = Settlement Point Prices
    - The locational price at a specific node or hub in the ERCOT grid
- RTM = Real-Time Market
    - ERCOT's market for near-instantaneous electricity dispatch and pricing, updated every 5 minutes
- LMP = locational marginal pricing
    - The price of delivering the next megawatt of electricity to a specific location on hte grid
- DAM = Day-Ahead Market
    - A forward market where participants buy/sell electricity for delivery one day in advance, on an hourly basis
- ASM = Ancillary Services Market
    - Backup reliability services ERCOT procures to ensure grid stability
- CPC = Clearing Prices for Capacity
    - Price paid per MW of ancillary service capacity awarded


**How to understand the data files -- `./data/raw`**:
- Raw electricity price & feature data used for short-term price forecasting for TotalEnergies consulting project
- Any files marked with "gridstatus" have been downloaded using the ERCOT API. All other data files are downloaded off the ERCOT website.
- The year or months that the data goes over have been specified in the filename.

