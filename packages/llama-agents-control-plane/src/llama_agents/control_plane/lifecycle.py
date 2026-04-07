import asyncio

# Global shutdown event used to signal long-running operations to stop
shutdown_event = asyncio.Event()
