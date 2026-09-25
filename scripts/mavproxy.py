#!/usr/bin/env python3
"""Space-safe launcher for the project-local MAVProxy installation."""
import runpy


if __name__ == "__main__":
    runpy.run_module("MAVProxy.mavproxy", run_name="__main__")
