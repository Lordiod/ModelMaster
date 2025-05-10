#!/usr/bin/env python3
"""
ModelMaster - Main Application Entry Point

This is the main entry point for the ModelMaster application.
It initializes the GUI and starts the application.
"""

from src.view.gui import ModelMasterGUI


def main():
    """Main function to start the application."""
    app = ModelMasterGUI()
    app.run()


if __name__ == "__main__":
    main()
