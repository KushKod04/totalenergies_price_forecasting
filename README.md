# totalenergies_electricity_data
Electricity data used for short-term price forecasting for TotalEnergies consulting project


Common abbreviations:
- spp = settlement point prices
    - the locational price at a specific node or hub in the ERCOT grid
- rtm = real-time market
    - ERCOT's market for near-instantaneous electricity dispatch and pricing, updated every 5 minutes
- lmp = locational marginal pricing
    - the price of delivering the next megawatt of electricity to a specific location on hte grid
- dam = day ahead market
    - a forward market where participants buy/sell electricity for delivery one day in advance, on an hourly basis


**How to understand the data files**:
- Any files marked with "gridstatus" have been downloaded using the ERCOT API. All other data files are downloaded off the ERCOT website.
- The year or months that the data goes over have been specified in the filename.

