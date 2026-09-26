## 2. Data and Feature Engineering

### 2.1 Data and Prediction Setting

The challenge covers two successive wind-driven storms affecting 302 counties in Indiana, Ohio, Pennsylvania, and West Virginia during March 11–19, 2026. Each county has 216 hourly records, including two pre-event days. A county-level split assigns 239 counties to training and 63 to testing.

Test-county outage observations cover only the first 72 hours, whereas weather is supplied for the full event. Indexing hours from zero, the observation cutoff is hour 71. At each prediction origin $t$ in the remaining 144 hours, we predict severity at $s=t+h$, for $h\in\{1,6,24,48\}$. Training inputs follow the same information boundary. The observed history includes pre-existing outages and the first storm's onset, establishing each county's initial state.

The target is the Outage Severity Index (OSI):

$$
\mathrm{OSI}_{i,s}=\max\!\left(0,\,0.40P_{i,s}+0.35N_{i,s}+0.25D_{i,s}-0.10R_{i,s}\right).
$$

Here, $i$ denotes a county. $P$ is the outage fraction and $D$ its trailing six-hour mean. $N$ and $R$ represent normalized increases and decreases in outage counts, respectively, smoothed with a centered three-hour mean. These components represent outage level, growth, persistence, and restoration.

### 2.2 Data Preparation and Integration

Records are aligned by county FIPS code and hourly timestamp. Targets beyond the event window remain missing. Historical OSI is reconstructed from the supplied components within the observation window, using the provided $N$ and $R$ values directly.

Weather combines NOAA URMA wind and temperature fields at approximately 2.5-km resolution, aggregated over county polygons, with ERA5 atmospheric context at approximately 31-km resolution, sampled at county locations. County attributes are linked through FIPS codes, with state-aware county-name matching for utility records.

### 2.3 Feature Construction

All horizon models share 163 base features in four groups.

| Feature group | Information represented |
|---|---|
| **Observed outage state** | Component and OSI values at the observation cutoff; historical level, variability, and trends; time since the observed peak. |
| **Weather conditions and exposure** | Origin-time and future-point weather; forward-window statistics; weather changes; cumulative exposure; observation-period summaries; selected interactions. |
| **Temporal position** | Hour of day, elapsed time since event onset and observation cutoff, and a fixed calendar-period code. |
| **County characteristics** | Customer count, rural–urban classification, population density, utility count, land cover, and tree canopy. |

Forward weather windows are $W_k(t)=\{t,t+1,\ldots,\min(t+k,215)\}$ for $k\in\{1,6,24,48\}$. Both endpoints are included, giving $k+1$ records in a complete window. Every horizon model receives all four windows and future-point weather at the corresponding offsets. Other features describe changes from 1, 3, and 6 hours earlier, exposure accumulated from March 13 00:00 to $t$, and weather summaries over the initial 72 hours and their final six hours. Coverage indicators and exposure normalization account for truncated windows; future-point weather beyond the record remains missing.

Wind direction and hour of day use sine–cosine encoding. The calendar-period code is a numerical feature assigned by origin date: 0 for March 14, 1 for March 15, 2 for March 16–17, and 3 for March 18–19. It also enters selected weather interactions.

### 2.4 Spatial Context and Graph Representation

Selected component models add 20 features summarizing the eight nearest counties among all 302 counties. Means, maxima, and county–neighborhood differences describe neighbors' initial observed outage histories and supplied weather, yielding 183 inputs.

For the spatial residual model, county boundaries and coordinates come from NWS GIS County Boundaries, version `c_16ap26`.[^nws] A fixed graph over all 302 counties combines symmetric eight-nearest-neighbor connections, shared borders, and self-loops. Edge attributes encode normalized distance, bearing through sine and cosine, and shared-border status. Its non-self neighborhoods also provide 32 summaries: means and neighbor-minus-own differences for 16 history and weather variables.

Seven static elevation, slope, and ruggedness descriptors are derived by aggregating USGS 3DEP/NED 1 arc-second elevation data (approximately 30 m) within the NWS county polygons.[^dem] These geographic inputs support the spatial residual model in Section 3.

[^nws]: NOAA/National Weather Service. *GIS County Boundaries*, version `c_16ap26`. [Data product](https://www.weather.gov/source/gis/Shapefiles/County/c_16ap26.zip).

[^dem]: U.S. Geological Survey. *3D Elevation Program: National Elevation Dataset, 1 arc-second*. Retrieved through [The National Map Access API](https://tnmaccess.nationalmap.gov/api/v1/products).
