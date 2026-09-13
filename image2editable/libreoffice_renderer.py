"""Render actual PPTX pages with a local LibreOffice installation."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import pypdfium2 as pdfium
from PIL import Image
import psutil

from image2editable.powerpoint_renderer import RendererUnavailable


class LibreOfficeRenderer:
    def __init__(self, executable: str | None) -> None:
        self.executable = executable

    @classmethod
    def discover(cls):
        configured = os.environ.get("IMAGE2EDITABLE_LIBREOFFICE")
        if configured:
            path = Path(configured)
            return cls(str(path) if path.is_absolute() and path.is_file() else None)
        executable = shutil.which("soffice.com") or shutil.which("soffice")
        if executable is None:
            candidates = [Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")]
            for name in ("ProgramFiles", "ProgramFiles(x86)"):
                if os.environ.get(name):
                    candidates.append(Path(os.environ[name]) / "LibreOffice/program/soffice.com")
            executable = next((str(path) for path in candidates if path.is_file()), None)
        return cls(executable)

    def available(self) -> bool:
        return self.executable is not None

    def identity(self) -> dict:
        return {"renderer": "libreoffice", "available": self.available()}

    def render_page(self, pptx_path, page_number, output_path, *, width, height) -> dict:
        if not self.available():
            raise RendererUnavailable("LibreOffice renderer is unavailable")
        if any(type(value) is not int or value <= 0 for value in (page_number, width, height)):
            raise ValueError("Page number and render dimensions must be positive integers")
        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        # Office creates deeply nested extension registries; a Run-relative
        # profile can exceed Windows path limits during cleanup.
        with tempfile.TemporaryDirectory(prefix="office-render-") as temporary:
            root = Path(temporary)
            source = root / "input.pptx"
            shutil.copyfile(pptx_path, source)
            export_filter = "pdf:impress_pdf_Export:" + json.dumps({
                "PageRange": {"type": "string", "value": str(page_number)},
            })
            command = [
                self.executable, "-env:UserInstallation=" + (root / "profile").as_uri(),
                "--headless", "--norestore", "--convert-to", export_filter,
                "--outdir", str(root), str(source),
            ]
            _run_office(command, root / "render.log")
            pdf = root / "input.pdf"
            if not pdf.is_file():
                raise RuntimeError("LibreOffice did not produce the requested page")
            with pdfium.PdfDocument(pdf) as document:
                if len(document) != 1:
                    raise RuntimeError("LibreOffice page selection is invalid")
                page = document[0]
                try:
                    bitmap = page.render(scale=max(width / page.get_width(), height / page.get_height()))
                    try:
                        with bitmap.to_pil().convert("RGB") as image:
                            if image.size == (width, height):
                                image.save(output)
                            else:
                                with image.resize((width, height), Image.Resampling.LANCZOS) as sized:
                                    sized.save(output)
                    finally:
                        bitmap.close()
                finally:
                    page.close()
        return {"renderer": "libreoffice", "width": width, "height": height, "path": str(output)}


def _run_office(command: list[str], log_path: Path) -> None:
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        previous, last_active = None, time.monotonic()
        tracked = {}
        try:
            while True:
                running = process.poll() is None
                activity = {}
                try:
                    parent = psutil.Process(process.pid)
                    tracked.update({child.pid: child for child in [parent, *parent.children(recursive=True)]})
                except psutil.Error:
                    pass
                for pid, child in list(tracked.items()):
                    try:
                        if not child.is_running() or child.status() == psutil.STATUS_ZOMBIE:
                            del tracked[pid]
                            continue
                        cpu = child.cpu_times()
                        counters = (cpu.user, cpu.system)
                        if hasattr(child, "io_counters"):
                            io = child.io_counters()
                            counters += (io.read_bytes, io.write_bytes)
                        activity[pid] = counters
                    except psutil.Error:
                        del tracked[pid]
                if not running and not tracked:
                    break
                if activity != previous:
                    previous, last_active = activity, time.monotonic()
                elif time.monotonic() - last_active >= 300:
                    raise RuntimeError("LibreOffice renderer is inactive")
                time.sleep(.25)
            if process.returncode:
                raise RuntimeError(f"LibreOffice render exited with code {process.returncode}")
        finally:
            if process.poll() is None or tracked:
                for child in tracked.values():
                    try:
                        child.kill()
                    except psutil.Error:
                        pass
                try:
                    for child in psutil.Process(process.pid).children(recursive=True):
                        try:
                            child.kill()
                        except psutil.Error:
                            pass
                except psutil.Error:
                    pass
                if process.poll() is None:
                    process.kill()
                    process.wait()
