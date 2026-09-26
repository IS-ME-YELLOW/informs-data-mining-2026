## 2. Data and Feature Engineering

### 2.1 Data and Prediction Setting

The challenge covers two successive wind-driven storms across 302 counties in four states during March 11–19, 2026. Each county has 216 hourly records; 239 counties form the training set and 63 the test set. Test-county outage observations end at hour 71, while weather covers the full event. For each prediction row $t$ in the remaining 144 hours, outage-derived features remain fixed at this cutoff and the target is $s=t+h$ for $h\in\{1,6,24,48\}$. Training applies the same information boundary. The history includes pre-existing outages and the first storm's onset.

The target is the Outage Severity Index (OSI):

$$
\mathrm{OSI}_{i,s}=\max\!\left(0,\,0.40P_{i,s}+0.35N_{i,s}+0.25D_{i,s}-0.10R_{i,s}\right).
$$

Here, $P$, $N$, $D$, and $R$ represent outage level, new outage growth, six-hour persistence, and restoration, respectively; $N$ and $R$ are normalized outage-count changes smoothed over three centered hours.

### 2.2 Base Feature Representation

Within the observed window, OSI and its lags are reconstructed from the organizer-supplied components, using the supplied $N$ and $R$ values directly. All horizon models share 163 base features describing observed outage state, weather and exposure, temporal position, and county characteristics. Outage features summarize cutoff conditions and historical levels, trends, and variability. Weather features combine county-aggregated NOAA URMA wind and temperature with ERA5 atmospheric context at the origin and target times and over 1h, 6h, 24h, and 48h forward windows. Temporal features locate each row within the day and event, with cyclic encoding for hour and wind direction. Static features describe customer exposure, rurality, population density, utility structure, land cover, and tree canopy.

### 2.3 Spatial and Graph Context

Selected $P$ models add 20 means, maxima, and own-versus-neighbor differences derived from each county's eight nearest neighbors, producing 183 inputs.

The GAT uses a fixed county graph built from NWS county boundaries and coordinates.[^nws] It combines symmetric eight-nearest-neighbor links, shared borders, and self-loops; edge attributes describe distance, bearing, and shared-border status. The node covariates comprise the 163 base features, 32 neighborhood summaries, two coordinates, and seven terrain descriptors derived from USGS 3DEP/NED elevation data.[^dem] These 204 covariates are combined in Section 3 with the horizon-specific tree forecast.

[^nws]: NOAA/National Weather Service. *GIS County Boundaries*, version `c_16ap26`. [Data product](https://www.weather.gov/source/gis/Shapefiles/County/c_16ap26.zip).

[^dem]: U.S. Geological Survey. *3D Elevation Program: National Elevation Dataset, 1 arc-second*. Retrieved through [The National Map Access API](https://tnmaccess.nationalmap.gov/api/v1/products).
