# Regression references

These CSVs contain numerical reference results for the analyses reported in the
manuscript and S1 Appendix. The tests normalize model and condition names before
comparing numerical values.

They are **test fixtures**, not inputs to `reproduce.py` or the notebook. All
analysis outputs are regenerated from `Data/`; existing `Results/` files are
not needed to start the complete workflow.

The score-stability reference contains both implicit and explicit conditions.
The regression test selects the 105 combinations of model, trait, and condition
reported in Table H of S1 Appendix: self-assessment and the four implicit conditions.

Update references only after independently checking an intentional change to
the data or methods; do not replace them merely to make a failing test pass.
