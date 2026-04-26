from pathlib import Path

def get_project_root() -> Path:
    root = Path.cwd().resolve()

    # Look for multiple markers (more robust)
    markers = ["configs", "src", ".git"]

    while True:
        if any((root / m).exists() for m in markers):
            return root

        if root.parent == root:
            raise RuntimeError("Project root not found.")

        root = root.parent


def resolve_path(relative_path: str) -> Path:
    return get_project_root() / relative_path