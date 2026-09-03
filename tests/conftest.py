"""pytest bootstrap for the harbor plugin test suite.

Puts the tests/ dir on sys.path so test modules in subdirs can
`from _pluginmeta import ...`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
