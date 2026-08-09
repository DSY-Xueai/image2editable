from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

from PIL import Image


class RendererUnavailable(RuntimeError):
    pass


class PowerPointRenderer:
    def __init__(
        self,
        dispatch_factory: Callable[[str], object] | None,
        *,
        co_initialize: Callable[[], None] | None = None,
        co_uninitialize: Callable[[], None] | None = None,
    ) -> None:
        self._dispatch_factory = dispatch_factory
        self._co_initialize = co_initialize
        self._co_uninitialize = co_uninitialize

    @classmethod
    def discover(cls) -> "PowerPointRenderer":
        if sys.platform != "win32":
            return cls(None)
        try:
            import pythoncom
            from win32com.client import DispatchEx
        except ImportError:
            return cls(None)
        return cls(
            DispatchEx,
            co_initialize=pythoncom.CoInitialize,
            co_uninitialize=pythoncom.CoUninitialize,
        )

    def available(self) -> bool:
        return self._dispatch_factory is not None

    def identity(self) -> dict:
        return {"renderer": "powerpoint", "available": self.available()}

    def render_page(
        self,
        pptx_path: str | Path,
        page_number: int,
        output_path: str | Path,
        *,
        width: int,
        height: int,
    ) -> dict:
        if not self.available():
            raise RendererUnavailable("PowerPoint renderer is unavailable")
        for value, label in (
            (page_number, "page_number"),
            (width, "width"),
            (height, "height"),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be a positive integer")

        source = Path(pptx_path).resolve()
        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            output.unlink()
        application = None
        presentation = None
        initialized = False
        try:
            if self._co_initialize is not None:
                self._co_initialize()
                initialized = True
            try:
                application = self._dispatch_factory("PowerPoint.Application")
            except Exception as error:
                raise RendererUnavailable(
                    "Microsoft PowerPoint could not be started"
                ) from error
            presentation = application.Presentations.Open(
                str(source),
                ReadOnly=True,
                Untitled=False,
                WithWindow=False,
            )
            slide = presentation.Slides(page_number)
            slide.Export(str(output), "PNG", width, height)
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError("PowerPoint renderer did not produce an image")
            with Image.open(output) as image:
                if image.size != (width, height):
                    raise RuntimeError("PowerPoint render dimensions are invalid")
            result = {
                "renderer": "powerpoint",
                "version": str(application.Version),
                "width": width,
                "height": height,
                "path": str(output),
            }
        finally:
            active_error = sys.exc_info()[0] is not None
            cleanup_error = None
            for action in (
                presentation.Close if presentation is not None else None,
                application.Quit if application is not None else None,
                self._co_uninitialize if initialized else None,
            ):
                if action is None:
                    continue
                try:
                    action()
                except Exception as error:
                    if cleanup_error is None:
                        cleanup_error = error
            if cleanup_error is not None and not active_error:
                raise cleanup_error
        return result
