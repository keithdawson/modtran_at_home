"""Streamlit web GUI for mwir_atmos.

DEV-BETA owned (CONTRACT.md "Wave 2 interfaces"). Launch with
``streamlit run mwir_atmos/webgui/app.py`` (see README.md "GUI launch
command"); nothing in this subpackage is imported by the CLI or the physics
core, and this package is only ever the *entry point* into
:mod:`mwir_atmos.api` -- :mod:`mwir_atmos.webgui.app` builds a
:class:`~mwir_atmos.types.Scenario` from widget state and calls
:func:`mwir_atmos.api.run_scenario`, exactly like :mod:`mwir_atmos.cli` does.
"""
