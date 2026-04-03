"""
Build entry point for creating deployment artifacts.

Usage: python -m llama_deploy.appserver.build
"""

from llama_agents.appserver.bootstrap import run_build

if __name__ == "__main__":
    run_build()
