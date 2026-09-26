## 1. Introduction

Successive wind-driven storms can damage infrastructure faster than utilities restore it, compounding outages while crews and mutual-assistance resources are already committed. Forecasts over the next several hours and days can support immediate response, crew positioning, mutual-aid planning, and emergency preparation. Their value depends on anticipating not only outage onset, but also persistence and recovery.

This challenge poses a county-level spatiotemporal forecasting problem with a fixed outage-observation boundary. Later prediction rows do not receive rolling outage updates, so models must project the remaining event from each county's initial state and weather information. The two-wave storm sequence further couples prior damage with later exposure. Moreover, the Outage Severity Index (OSI) combines outage level, growth, persistence, and restoration; these components behave differently, and useful forecast combinations may change with horizon.

We address these challenges with a staged framework that first constructs an OSI forecast from its components, explicitly incorporating observed state where the component definition permits. Short-horizon forecasts combine multiple views of the same target time, whereas longer horizons use ensembles designed to retain high-severity predictions. A graph attention network then models spatial residual structure around the tree forecast. County-separated validation preserves the competition's information boundary throughout model selection and evaluation.


## 5. Conclusion

We developed a multi-horizon OSI forecasting framework for a setting in which outage observations stop well before the event ends. The analysis shows that observed state is most useful when incorporated according to the target definition, and that forecast combination should reflect horizon-specific error patterns. Spatial residual modelling provides additional potential, although its gains are less stable than those of the tree framework.

The evidence is limited to transfer across counties within one storm sequence. Deployment across future events would require evaluation with rolling outage observations and issue-time weather forecasts. Within these boundaries, the framework provides a reproducible basis for forecasting outage severity at operationally relevant horizons.
