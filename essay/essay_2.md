## 2. Data and Feature Engineering

### 2.1 Data and Prediction Setting

The challenge dataset covers two successive wind-driven storm systems affecting 302 counties in Indiana, Ohio, Pennsylvania, and West Virginia during March 11–19, 2026. Each county has 216 hourly records, including a two-day pre-event period. The data are divided geographically into 239 training counties and 63 test counties.

For test counties, outage observations are available for the first 72 hours, while weather information is provided for the full event window. At each hourly prediction origin during the remaining 144 hours, the task is to forecast outage severity 1, 6, 24, and 48 hours ahead. We construct training inputs under the same setting: outage history remains fixed at the initial 72-hour window, while weather features vary with the prediction origin and target time.

The prediction target is the Outage Severity Index (OSI):

$$
\mathrm{OSI}_{i,s}
=
\max\!\left(
0,\,
0.40P_{i,s}+0.35N_{i,s}+0.25D_{i,s}-0.10R_{i,s}
\right),
$$

where *i* denotes a county and *s* the target hour. *P* is the fraction of customers without power, and *D* is its trailing six-hour mean. *N* and *R* represent normalized increases and decreases in outage counts, respectively, smoothed using a centered three-hour average. These components describe outage level, growth, persistence, and restoration, providing the basis for the component forecasting framework in Section 3.

The initial observation window contains both pre-existing outages and the onset of the first storm wave. It therefore provides a meaningful starting state for each county.

### 2.2 Data Preparation

We organize all records by county FIPS code and hourly timestamp, aligning each prediction origin with its corresponding future targets. Training and evaluation use targets within the supplied event window; targets extending beyond the final hour remain missing.

Observed OSI is reconstructed from the supplied components using a consistent formula and four-decimal precision across training and test counties. The supplied *N* and *R* values are used directly. Historical summaries are then calculated from the same initial observation window for every county.

Weather inputs combine NOAA URMA wind and temperature fields at approximately 2.5-km resolution with ERA5 atmospheric context at approximately 31-km resolution. URMA variables are aggregated over county polygons, while ERA5 variables are sampled at county locations. We retain the supplied hourly alignment. External county attributes are linked through FIPS codes, with state-aware county-name matching for utility service records.

### 2.3 Feature Construction

The initial component baseline and ensemble sources described in Section 3 use a common representation of 163 features, organized into four groups.

| Feature group | Main information represented |
|---|---|
| **Observed outage state** | Latest OSI and component values; historical means, maxima, and variability; recent trends; and time since the observed severity peak. |
| **Weather conditions and exposure** | Weather at the prediction origin and target hours; wind, precipitation, and temperature summaries over 1-, 6-, 24-, and 48-hour windows; wind-threshold exceedances, cumulative exposure, and selected interactions. |
| **Temporal position** | Hour of day, elapsed time since storm onset and the observation cutoff, and calendar-based storm-phase indicators. |
| **County characteristics** | Customer count, rural–urban classification, population density, utility count, land-cover composition, and tree-canopy coverage. |

Wind direction and hour of day are encoded using sine and cosine transformations. Weather-window length and availability indicators describe coverage near the end of the dataset, alongside exposure measures normalized by the number of available hours. County characteristics draw on USDA, Census, EIA, and county-level land-cover and canopy data.

We additionally construct 20 spatial-context features using each county’s eight nearest geographic neighbors among all 302 counties. These features summarize neighbors’ observed outage states and supplied weather through means, maxima, and differences between the focal county and its neighborhood. Only the designated component-model extensions in Section 3.2 use this augmented 183-feature representation.

Together, these inputs describe each county’s initial condition, subsequent weather exposure, and local context. Section 3 explains how these information sources are incorporated into the prediction structure.

### 2.4 Spatial Representation

We represent the 302 counties as nodes in a fixed geographic graph. Edges combine symmetric eight-nearest-neighbor connections, shared county borders, and self-loops. Each directed edge carries four attributes: normalized geographic distance, sine and cosine of the bearing, and a shared-border indicator. The topology is shared across hourly graph snapshots.

The graph-model input contains 205 features per county: the tree-ensemble prediction defined in Section 3.5, the 163 base features, two geographic coordinates, seven terrain descriptors, and 32 neighborhood summaries. Terrain descriptors summarize elevation, slope, and ruggedness. The neighborhood summaries are derived from 16 history and weather variables within the base feature set, adding their neighborhood means and neighbor-minus-own differences. These means use the non-self neighbors in the combined graph. This representation brings together the existing forecast, local conditions, and geographic context for spatial residual learning.

<!-- Editorial note: An optional timeline after Section 2.1 could show the 72-hour outage observation window, the 144-hour prediction window, weather availability throughout the 216-hour record, and the four target times for one prediction origin. -->
