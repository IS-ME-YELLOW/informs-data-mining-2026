## 2. Data and Feature Engineering

### 2.1 Data and Prediction Setting

The challenge covers two successive wind-driven storms across 302 counties in Indiana, Ohio, Pennsylvania, and West Virginia during March 11–19, 2026. Each county has 216 hourly records; 239 counties form the training set and 63 the test set.

Test-county outage observations cover hours 0–71, while weather covers the full event. At each origin $t$ in the remaining 144 hours, we predict severity at $s=t+h$ for $h\in\{1,6,24,48\}$. Training follows the same boundary. The observed history includes pre-existing outages and the first storm's onset, providing each county's initial state.

The target is the Outage Severity Index (OSI):

$$
\mathrm{OSI}_{i,s}=\max\!\left(0,\,0.40P_{i,s}+0.35N_{i,s}+0.25D_{i,s}-0.10R_{i,s}\right).
$$

Here, $P$ is the outage fraction, $D$ its trailing six-hour mean, and $N$ and $R$ are normalized increases and decreases in outage counts smoothed over three centered hours. They represent outage level, growth, persistence, and restoration.

### 2.2 Base Feature Representation

Records are aligned by FIPS code and hour. Historical OSI is reconstructed within the observed window using the supplied components, including $N$ and $R$ directly. Targets beyond the event remain missing. County attributes are linked by FIPS, with state-aware name matching for utility records.

Horizon models share 163 base features covering observed outage state, weather and exposure, temporal position, and county characteristics. Outage features summarize component and OSI values at the cutoff, historical levels and trends, variability, and time since the observed peak. Weather combines NOAA URMA wind and temperature fields aggregated over county polygons with ERA5 atmospheric context sampled at county locations. Features include origin and future-point conditions, inclusive forward-window summaries through 1h, 6h, 24h, and 48h, recent changes, accumulated exposure, and observed-period summaries. Models receive all four window spans; coverage indicators mark boundary truncation.

Temporal features describe hour of day and elapsed time from event onset and the observation cutoff. A four-level code distinguishes March 14, March 15, March 16–17, and March 18–19. Wind direction and hour use sine–cosine encoding. County features include customer count, rurality, population density, utility count, land cover, and tree canopy.

### 2.3 Spatial and Graph Context

Selected tree models add 20 features from each county's eight nearest neighbors. Means, maxima, and county–neighborhood differences summarize initial outage histories and weather, giving 183 inputs with the base features.

The GAT uses a separate spatial representation. Boundaries and coordinates come from NWS GIS County Boundaries, version `c_16ap26`.[^nws] A fixed graph combines symmetric eight-nearest-neighbor links, shared borders, and self-loops. Edges encode normalized distance, sine–cosine bearing, and shared-border status. Non-self neighborhoods provide 32 means and neighbor-minus-own differences for 16 history and weather variables.

Seven terrain descriptors are derived from USGS 3DEP/NED 1 arc-second elevation data within county polygons.[^dem] The 20 tree-model features and 32 GAT neighborhood summaries are distinct representations; Section 3 gives their model-specific use and the complete GAT input composition.

[^nws]: NOAA/National Weather Service. *GIS County Boundaries*, version `c_16ap26`. [Data product](https://www.weather.gov/source/gis/Shapefiles/County/c_16ap26.zip).

[^dem]: U.S. Geological Survey. *3D Elevation Program: National Elevation Dataset, 1 arc-second*. Retrieved through [The National Map Access API](https://tnmaccess.nationalmap.gov/api/v1/products).
