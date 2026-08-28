Subject: INFORMS 2026 Data Mining Challenge — Data Clarification Questions

Dear INFORMS Data Mining Challenge Organizers,

Thank you for organizing this challenge. We have carefully read all the provided documentation and conducted a thorough verification of the data against the stated statistics. We found a few discrepancies between the PDF documentation and the actual data files, and would appreciate your clarification on the following points.


1. osi_target_t01h maximum value (variable_descriptions.pdf)

PDF states (variable_descriptions.pdf, page 4):"osi_target_t01h: OSI 1 hour ahead. Range: 0–0.599. Mean: 0.0062."

The Actual data (DM_Train.csv, computed from all 51,385 non-NaN values):max = **0.6505**, mean = 0.0062

For comparison, the other targets' maxima are:
- osi_target_t06h: max = 0.5989 (PDF says 0.599 — matches)
- osi_target_t48h: max = 0.5980 (PDF says 0.598 — matches)

The actual maximum of osi_target_t01h (0.6505) is identical to the maximum of the `osi` column itself (0.6505, which rounds to the PDF-stated 0.65). This makes sense since osi_target_t01h at time t is the OSI at time t+1. The PDF value of 0.599 appears to be inconsistent.

2. osi_delta_t01h range (variable_descriptions.pdf)

PDF states (variable_descriptions.pdf, page 4):"osi_delta_t01h: Change in OSI to t+1h. Range: −0.369 to 0.596."

The Actual data (DM_Train.csv, computed from all 51,385 non-NaN values):min = −0.1757, max = 0.2926

For comparison:
- osi_delta_t06h: min = −0.3692, max = 0.5963 (PDF says −0.369 to 0.596 — matches)
- osi_delta_t48h: min = −0.6401, max = 0.5925 (PDF says −0.640 to 0.593 — matches)

The PDF-stated range for osi_delta_t01h (−0.369 to 0.596) is identical to the range for osi_delta_t06h. Since a 1-hour change should have a narrower range than a 6-hour change (which the actual data confirms), we suspect the PDF may have inadvertently copied the range from osi_delta_t06h. 


3. Pre-event outage county count (osi_methodology.pdf)

PDF states (osi_methodology.pdf, page 2, Data Quality Notes):"Lingering outages from a prior February event are present in 236 of 239 train counties"

The Actual data (DM_Train.csv, March 11–12 pre-event window, in_event_window=False):237 of 239 train counties have outageCount > 0 in at least one pre-event hour.

The two counties with zero outages throughout the entire pre-event window are:
- FIPS 42023 (Cameron County, PA)
- FIPS 42057 (Fulton County, PA)

We are therefore unsure whether the correct count is 236 or 237. 


4. N_t and R_t mutual exclusivity (osi_methodology.pdf)

PDF states (osi_methodology.pdf, page 1):
- N_t = max(0, ΔoutageCount) / customersTracked
- R_t = max(0, −ΔoutageCount) / customersTracked

By this definition, N_t and R_t should be mutually exclusive (when one is positive, the other must be zero, since they are derived from the same signed delta).

The Actual data (DM_Train.csv):In 26.8% of rows (13,826 out of 51,624), both N_t > 0 and R_t > 0 simultaneously.

Example: FIPS 18009 at 3/12/2026 10:00, outageCount goes from 10 (previous hour) to 10 (current hour, Δ=0), yet N_t = 0.000306 and R_t = 0.000306 (both non-zero).

This suggests the stored N_t and R_t may use a gross-flow definition (counting new outages and restorations separately, which can co-occur) rather than the net-change definition stated in the PDF. However we are unsure of the real definition.

We would appreciate it if you could clarify the above issues. Thank you so much for your time.

Best regards,
Team BugNoir
