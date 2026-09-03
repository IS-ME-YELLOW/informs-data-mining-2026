Dear Team BugNoir,

Thank you for your careful verification. You are correct on all four issues. The first three are documentation errors that will not affect your modelling.

The fourth issue on N_t and R_t is more substantive. The data was provided by PowerOutage.com and the actual computation may differ from the formula stated in the documentation. We are following up with the data provider. For now, please use the values stored in DM_Train.csv as your authoritative source rather than recomputing from the formula. The training data reflects the same computation as the test data, so patterns learned from it will generalize correctly.


Thanks