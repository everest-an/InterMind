import uvicorn

from latentcore.config import get_settings


def main():
    settings = get_settings()
    uvicorn.run(
        "latentcore.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
