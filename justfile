setup:
    uv sync

run:
    uv run ./src/main.py

clean:
    rm -rf .venv
    rm -rf __pycache__
