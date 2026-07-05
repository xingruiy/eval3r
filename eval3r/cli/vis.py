"""Visualization command group (intentionally not registered).

There is no ``e3r vis`` command. Visual debug artifacts — error-colored point
clouds and distance histograms — are produced by :mod:`eval3r.reports.plots`
into a run directory's ``debug/`` folder when the protocol's reporting spec
requests them (task 016). An interactive viewer command remains out of scope.
"""
