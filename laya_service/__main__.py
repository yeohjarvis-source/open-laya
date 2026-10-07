from __future__ import annotations


def main() -> None:
    import uvicorn

    uvicorn.run("laya_service.app:create_app", factory=True, host="0.0.0.0", port=6767)


if __name__ == "__main__":
    main()

