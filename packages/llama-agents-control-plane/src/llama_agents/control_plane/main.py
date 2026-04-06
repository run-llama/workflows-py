import argparse
import asyncio
import logging
import subprocess
import sys
from typing import Literal

import uvicorn
from dotenv import load_dotenv

from .log_config import get_uvicorn_log_config, setup_logging

logger = logging.getLogger(__name__)


def run_server_subprocess(
    app_path: str,
    host: str,
    port: int,
    name: str,
    log_level: str,
    access_log: bool,
) -> subprocess.Popen:
    """Run a server with uvicorn in a subprocess with reload"""
    logger.info(f"Starting {name} server on {host}:{port} with reload")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        app_path,
        "--host",
        host,
        "--port",
        str(port),
        "--reload",
        "--log-level",
        log_level.lower(),
    ]
    if not access_log:
        cmd.append("--no-access-log")

    return subprocess.Popen(cmd)


async def run_server_async(
    app_path: str,
    host: str,
    port: int,
    name: str,
    log_level: str,
    access_log: bool,
    log_format: Literal["standard", "json"] = "standard",
) -> None:
    """Run a server with uvicorn async"""
    logger.info(f"Starting {name} server on {host}:{port}")

    config = uvicorn.Config(
        app=app_path,
        host=host,
        port=port,
        log_level=log_level.lower(),
        access_log=access_log,
        reload=False,
        log_config=get_uvicorn_log_config(log_level),
    )
    server = uvicorn.Server(config)
    await asyncio.create_task(server.serve())


async def run_servers(
    host: str = "0.0.0.0",
    manage_api_port: int = 8000,
    build_api_port: int = 8001,
    reload: bool = False,
    log_level: str = "info",
    log_format: Literal["standard", "json"] = "standard",
    access_log: bool = True,
) -> None:
    """Run both main API and build API servers concurrently"""
    setup_logging(log_level, log_format)
    load_dotenv()

    reload_str = " with auto-reload" if reload else ""
    logger.info(
        f"Running servers on {host}:{manage_api_port} and {host}:{build_api_port}{reload_str}"
    )

    if reload:
        # Start both servers as subprocesses with reload
        cp = run_server_subprocess(
            app_path="llama_agents.control_plane.manage_api.manage_app:app",
            host=host,
            port=manage_api_port,
            name="Control Plane",
            log_level=log_level,
            access_log=access_log,
        )
        ba = run_server_subprocess(
            app_path="llama_agents.control_plane.build_api.build_app:build_app",
            host=host,
            port=build_api_port,
            name="Build API",
            log_level=log_level,
            access_log=access_log,
        )
        # Wait for both processes
        cp.wait()
        ba.wait()
    else:
        # Start both servers as async tasks
        cp_task = run_server_async(
            app_path="llama_agents.control_plane.manage_api.manage_app:app",
            host=host,
            port=manage_api_port,
            name="Control Plane",
            log_level=log_level,
            access_log=access_log,
            log_format=log_format,
        )
        ba_task = run_server_async(
            app_path="llama_agents.control_plane.build_api.build_app:build_app",
            host=host,
            port=build_api_port,
            name="Build API",
            log_level=log_level,
            access_log=access_log,
            log_format=log_format,
        )
        await asyncio.gather(cp_task, ba_task)


def main() -> None:
    """Main entry point for running both servers"""

    parser = argparse.ArgumentParser(description="Run the control plane")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the servers on"
    )
    parser.add_argument(
        "--manage-api-port",
        type=int,
        default=8000,
        help="Port to run the manage API on",
    )
    parser.add_argument(
        "--build-api-port", type=int, default=8001, help="Port to run the build API on"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload the servers on code changes (for development)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Log level to run the servers at",
    )
    parser.add_argument(
        "--log-format",
        type=str,
        default="standard",
        help="Log format to use (standard or json)",
    )
    parser.add_argument(
        "--no-access-log",
        action="store_true",
        help="Disable access log for both servers",
    )
    args = parser.parse_args()

    try:
        asyncio.run(
            run_servers(
                host=args.host,
                manage_api_port=args.manage_api_port,
                build_api_port=args.build_api_port,
                reload=args.reload,
                log_level=args.log_level,
                log_format=args.log_format,
                access_log=not args.no_access_log,
            )
        )
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, exiting...")


if __name__ == "__main__":
    main()
