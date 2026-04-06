from code_review_env.server.app import app, main as _main

__all__ = ["app", "main"]


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
