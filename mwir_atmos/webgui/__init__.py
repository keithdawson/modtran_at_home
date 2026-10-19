"""Streamlit web GUI for mwir_atmos.

Launch with ``streamlit run mwir_atmos/webgui/app.py`` (see README.md "GUI
launch command"). Nothing in this subpackage is imported by the CLI or the
physics core: :mod:`mwir_atmos.webgui.app` builds a
:class:`~mwir_atmos.types.Scenario` from widget state and calls
:func:`mwir_atmos.api.run_scenario`, exactly like :mod:`mwir_atmos.cli` does.
See :mod:`mwir_atmos.webgui.app` for the layout of the package.
"""
