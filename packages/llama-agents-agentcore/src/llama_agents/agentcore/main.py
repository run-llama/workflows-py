import os
from argparse import ArgumentParser

from .entrypoint import AGENTCORE_HOST, AGENTCORE_PORT, app

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--run",
        help="Run the application in the target environment",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--local",
        help="Run in local mode with in-memory store (no AWS required)",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    if args.local:
        os.environ["LLAMA_AGENTCORE_LOCAL"] = "1"

    if args.run or args.local:
        print(f"Starting app on {AGENTCORE_HOST}:{AGENTCORE_PORT}")  # noqa
        if args.local:
            print(  # noqa
                "Local mode: using in-memory store, no AWS credentials needed.\n"
                "Send requests to POST http://localhost:8080/invocations"
            )
        app.run(port=AGENTCORE_PORT, host=AGENTCORE_HOST)
