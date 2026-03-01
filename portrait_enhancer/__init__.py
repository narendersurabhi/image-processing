"""Portrait Enhancer package."""

__version__ = "0.1.0"


def create_app():
    from portrait_enhancer.ui.app import PortraitEnhancerV2

    return PortraitEnhancerV2()


def run_app():
    from portrait_enhancer.ui.app import run_app as _run_app

    return _run_app()


def create_qt_app():
    from portrait_enhancer.ui_qt.main_window import PortraitEnhancerQtWindow

    return PortraitEnhancerQtWindow()


def run_qt_app():
    try:
        from portrait_enhancer.ui_qt.main_window import run_qt_app as _run_qt_app
    except ImportError as exc:
        raise SystemExit(
            "PySide6 is not installed. Install it with `pip install .[qt]` or `pip install -r requirements/optional-qt.txt`."
        ) from exc

    return _run_qt_app()


__all__ = ["__version__", "create_app", "run_app", "create_qt_app", "run_qt_app"]
