"""GitHub App callback server for handling OAuth flows"""

import asyncio
import logging
import webbrowser
from textwrap import dedent
from typing import Any, Dict, cast

from aiohttp.web_app import Application
from aiohttp.web_request import Request
from aiohttp.web_response import Response
from aiohttp.web_runner import AppRunner, TCPSite

logger = logging.getLogger(__name__)


class GitHubCallbackServer:
    """Local HTTP server to handle GitHub App installation callbacks"""

    def __init__(self, port: int = 41010):
        self.port = port
        self.callback_data: Dict[str, Any] = {}
        self.callback_received = asyncio.Event()
        self.app: Application | None = None
        self.runner: AppRunner | None = None
        self.site: TCPSite | None = None

    async def start_and_wait(self, timeout: float = 300) -> Dict[str, Any]:
        """Start the server and wait for a callback with timeout"""
        await self._start_server()

        try:
            # Wait for callback with timeout
            await asyncio.wait_for(self.callback_received.wait(), timeout=timeout)
            logger.debug(f"processing callback: {self.callback_data}")
            return self.callback_data
        except asyncio.TimeoutError:
            raise TimeoutError(f"GitHub callback timed out after {timeout} seconds")
        finally:
            await self.stop()

    async def _start_server(self) -> None:
        """Start the aiohttp server"""
        self.app = Application()
        self.app.router.add_get("/", self._handle_callback)
        self.app.router.add_get("/{path:.*}", self._handle_callback)

        self.runner = AppRunner(self.app, logger=None)  # Suppress server logs
        await self.runner.setup()

        self.site = TCPSite(self.runner, "localhost", self.port)
        await self.site.start()

        logger.debug(f"GitHub callback server started on port {self.port}")

    async def _handle_callback(self, request: Request) -> Response:
        """Handle the GitHub callback"""
        # Capture query parameters
        query_params: dict[str, str] = dict(cast(Any, request.query))
        self.callback_data.update(query_params)

        # Signal that callback was received
        logger.debug(f"GitHub callback received: {query_params}")
        self.callback_received.set()

        # Return success page
        html_response = self._get_success_html()
        return Response(text=html_response, content_type="text/html")

    async def stop(self) -> None:
        """Stop the server and cleanup"""
        if self.site:
            await self.site.stop()
            self.site = None
        if self.runner:
            await self.runner.cleanup()
            self.runner = None
        self.app = None
        self.callback_received.clear()
        logger.debug("GitHub callback server stopped")

    def _get_success_html(self) -> str:
        """Get the HTML for the success page"""
        return dedent("""
        <!DOCTYPE html>
        <html>
            <meta charset="UTF-8">
            <head>
                <title>llamactl - Authentication Complete</title>
                <style>
                    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

                    * {
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }

                    body {
                        font-family: 'JetBrains Mono', 'Courier New', monospace;
                        background: #1a0d26;
                        color: #e4e4e7;
                        min-height: 100vh;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        line-height: 1.6;
                    }

                    .terminal {
                        background: #0f0a17;
                        border: 2px solid #7c3aed;
                        border-radius: 0;
                        max-width: 600px;
                        width: 90vw;
                        padding: 0;
                        box-shadow: 0 0 20px rgba(124, 58, 237, 0.3);
                    }

                    .terminal-header {
                        background: #7c3aed;
                        color: #ffffff;
                        padding: 12px 20px;
                        font-weight: bold;
                        font-size: 14px;
                        border-bottom: 2px solid #6d28d9;
                    }

                    .terminal-body {
                        padding: 30px;
                    }

                    .prompt {
                        color: #10b981;
                        font-weight: bold;
                    }

                    .success-icon {
                        color: #10b981;
                        font-size: 24px;
                        margin-right: 8px;
                    }

                    .highlight {
                        color: #a78bfa;
                        font-weight: bold;
                    }

                    .instruction {
                        background: #2d1b69;
                        border: 1px solid #7c3aed;
                        padding: 16px;
                        margin: 20px 0;
                        border-radius: 4px;
                    }

                    .blink {
                        animation: blink 1s infinite;
                    }

                    @keyframes blink {
                        0%, 50% { opacity: 1; }
                        51%, 100% { opacity: 0; }
                    }
                </style>
            </head>
            <body>
                <div class="terminal">
                    <div class="terminal-header">
                        llamactl@github-auth-server
                    </div>
                    <div class="terminal-body">
                        <div><span class="prompt">$</span> GitHub App installation successful</div>
                        <div class="instruction">
                            <div><strong>Next Steps:</strong></div>
                            <div>1. Close this browser</div>
                            <div>2. Return to your terminal</div>
                            <div>3. Continue with llamactl<span class="blink">_</span></div>
                        </div>
                    </div>
                </div>
            </body>
        </html>
        """).strip()


async def main() -> None:
    """Main function to demo the callback server"""
    logging.basicConfig(level=logging.INFO)

    server = GitHubCallbackServer(port=41010)

    # Start server and open browser
    logger.debug(f"Starting GitHub callback server on http://localhost:{server.port}")
    logger.debug("Opening browser to show success page...")

    # Open browser to success page to see the styling
    webbrowser.open(f"http://localhost:{server.port}")

    try:
        # Wait for callback (or just keep server running)
        logger.debug("Server running... Press Ctrl+C to stop")
        callback_data = await server.start_and_wait(timeout=3600)  # 1 hour timeout
        logger.debug(f"Received callback data: {callback_data}")
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
