## 5. Discussion

**Damage and restoration information.** Development-stage diagnostics found differences in outage magnitude and persistence among counties with broadly similar weather and initial conditions. Infrastructure and vegetation condition, damage reports, and restoration resources available at forecast time could help distinguish these outcomes. Their predictive value should be assessed for both persistent outages and faster-recovering counties, checking whether reduced underprediction in the former introduces overprediction in the latter.

**Learning difficult component values.** The evaluated 1h N/R occurrence–magnitude models improved background predictions but increased errors for larger component values. This motivates examining training weights that balance rare high values with background accuracy. A joint objective combining component supervision with reconstructed OSI error could also account for how component errors reinforce or offset one another. These changes should be assessed through final OSI performance, including any overprediction introduced by greater emphasis on large values.

**Spatial relationships and correction stability.** The GAT varies attention weights across snapshots while retaining a fixed geographic topology. Utility service relationships, electrical connectivity, and weather-dependent exposure patterns offer alternative bases for defining edges. Comparisons should examine whether these relationships improve the distribution of gains across counties alongside pooled accuracy, given the uneven benefits observed for the current spatial module.

**Transfer and operational forecasting.** Data from multiple storms and regions would broaden coverage of initial states, damage, and recovery. Evaluation on entire held-out storms and contiguous regions, with model specifications fixed beforehand, would assess transfer beyond the current event. Event-relative timing features should also be examined as an alternative to the current calendar-period code.

Operational use would require rolling outage observations and weather forecasts available at the actual issue time. In this challenge, a nominal 1h prediction late in the event can still be several days beyond the latest outage observation. Updating observations changes the initial state used by P and the known contribution to D, requiring corresponding changes to features and training. Evaluation should therefore consider both forecast horizon and observation age, together with weather forecast uncertainty. Target-time aggregation must use only predictions available when the forecast is issued.

<!--
Editorial note; not part of the manuscript.

Condensed from essay/essay_5_draft_v2.md only. Source and evidence notes remain in that file. Development-stage diagnostics are not presented as an analysis of all residual errors of the final GAT. Proposed data, objectives, graph relationships, and operational evaluations remain future research directions. No new experimental results are introduced.
-->
