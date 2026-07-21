"""Load prompt templates from the templates/ directory."""

from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def load_template(name: str, template_dir: str | Path | None = None, **kwargs) -> str:
    """Read a template file and optionally format it with kwargs.

    Parameters
    ----------
    name : str
        Filename inside the template directory (e.g. ``"curator_system.txt"``).
    template_dir : str or Path or None
        Override the default template directory.
    **kwargs
        Values to substitute via ``str.format()``.  When empty the raw
        template text is returned.
    """
    base = Path(template_dir) if template_dir else _TEMPLATE_DIR
    text = (base / name).read_text(encoding="utf-8")
    return text.format(**kwargs) if kwargs else text
