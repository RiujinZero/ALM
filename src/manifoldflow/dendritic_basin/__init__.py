"""Dendritic Basin Learning: local-plasticity, recurrent-basin architecture
with no backpropagation, no autograd, no frozen pretrained encoders.

See docs/Dendritic_Basin_Learning_Design_v0.1.md for the design and
docs/ManifoldFlow_ALM_Journal.md for why this replaced the MF-ALM-TS local
learner as the active line of work (M1/M2's local token head could not
solve temporal XOR -- see the journal's XOR probe entries).

Deliberately pure NumPy: no torch import anywhere in this package. That is
a hard guarantee against accidentally routing a gradient through anything
here (design doc section 3.1), not just a style choice.
"""
