"""Run the API server: ``python -m app``.

Uvicorn defaults to the ProactorEventLoop on Windows, which psycopg's async
mode cannot use. This entry point passes a selector loop factory so the same
command works on every platform.
"""

import uvicorn

from app.config import get_settings
from app.runtime import configure_event_loop


def main() -> None:
    configure_event_loop()
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        loop="app.runtime:selector_loop_factory",
    )


if __name__ == "__main__":
    main()
