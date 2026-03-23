# tests/conftest.py
import pytest

def pytest_addoption(parser):
    parser.addoption("--debug-mode", action="store", default="false", help="run tests in debug mode")

