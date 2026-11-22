"""Zero-dependency HTTP server (`ThreadingHTTPServer`) for the Odwołania2JSON local web GUI."""
from __future__ import annotations

import base64
import json
import mimetypes
import re
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .service import WebWorkspaceService

STATIC_DIR = Path(__file__).resolve().parent / "static"


class WebServer:
    """Serves the local Odwołania2JSON web workbench and REST API on `host:port`."""

    def __init__(
        self,
        service: WebWorkspaceService,
        *,
        host: str | None = None,
        port: int | None = None,
    ):
        web_cfg = service.config.web or {}
        self.service = service
        self.host = host or str(web_cfg.get("host", "127.0.0.1"))
        self.port = int(web_cfg.get("port", 8000) if port is None else port)

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        service = self.service

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

            def _send_bytes(self, body: bytes, content_type: str, status: int = 200, *, extra_headers: dict[str, str] | None = None) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                for k, v in (extra_headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _send_json(self, payload: Any, status: int = 200) -> None:
                raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self._send_bytes(raw, "application/json; charset=utf-8", status=status)

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                return json.loads(raw.decode("utf-8"))

            def _read_raw_body(self) -> bytes:
                length = int(self.headers.get("Content-Length", "0"))
                return self.rfile.read(length) if length > 0 else b""

            def _serve_static(self, rel: str) -> None:
                clean = rel.lstrip("/") or "index.html"
                target = (STATIC_DIR / clean).resolve()
                if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
                    self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                    return
                ctype, _ = mimetypes.guess_type(str(target))
                if ctype and ctype.startswith("text/"):
                    ctype = f"{ctype}; charset=utf-8"
                elif target.suffix == ".js":
                    ctype = "application/javascript; charset=utf-8"
                self._send_bytes(target.read_bytes(), ctype or "application/octet-stream")

            def do_GET(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlsplit(self.path)
                path = parsed.path
                try:
                    if path == "/api/info":
                        self._send_json(service.info())
                        return
                    if path == "/api/docs":
                        self._send_json({"documents": service.list_documents()})
                        return
                    if path == "/api/docs/export.zip":
                        data = service.export_all_zip()
                        self._send_bytes(
                            data,
                            "application/zip",
                            extra_headers={"Content-Disposition": 'attachment; filename="odwolania_output.zip"'},
                        )
                        return
                    m_page = re.fullmatch(r"/api/docs/([a-f0-9]+)/page/(\d+)\.png", path)
                    if m_page:
                        png = service.get_page_png(m_page.group(1), int(m_page.group(2)))
                        self._send_bytes(png, "image/png")
                        return
                    m_json = re.fullmatch(r"/api/docs/([a-f0-9]+)/json", path)
                    if m_json:
                        doc = service.get_document(m_json.group(1))
                        fname = Path(doc["filename"]).with_suffix(".json").name
                        raw = (json.dumps(doc["data"] or {}, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
                        self._send_bytes(
                            raw,
                            "application/json; charset=utf-8",
                            extra_headers={"Content-Disposition": f'attachment; filename="{fname}"'},
                        )
                        return
                    m_doc = re.fullmatch(r"/api/docs/([a-f0-9]+)", path)
                    if m_doc:
                        self._send_json(service.get_document(m_doc.group(1)))
                        return
                    if path == "/api/types":
                        self._send_json(service.list_types())
                        return
                    m_opt = re.fullmatch(r"/api/types/optimize/([a-f0-9]+)", path)
                    if m_opt:
                        self._send_json(service.get_optimization_job(m_opt.group(1)))
                        return
                    if path == "/api/settings":
                        self._send_json(service.get_settings())
                        return
                    if not path.startswith("/api/"):
                        self._serve_static(path)
                        return
                    self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                except KeyError as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                except Exception as exc:
                    self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

            def do_POST(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlsplit(self.path)
                path = parsed.path
                qs = urllib.parse.parse_qs(parsed.query)
                try:
                    if path == "/api/docs/upload":
                        filename = urllib.parse.unquote(self.headers.get("X-Filename") or (qs.get("filename") or ["document.pdf"])[0])
                        rel_path = urllib.parse.unquote(self.headers.get("X-Relpath") or filename)
                        raw = self._read_raw_body()
                        if not raw:
                            self._send_json({"error": "empty upload body"}, status=HTTPStatus.BAD_REQUEST)
                            return
                        doc = service.upload_pdf(filename, raw, rel_path=rel_path)
                        self._send_json(doc, status=HTTPStatus.CREATED)
                        return
                    if path == "/api/docs/load-input":
                        added = service.load_from_input_dir()
                        self._send_json({"added": added, "documents": service.list_documents()})
                        return
                    m_analyze = re.fullmatch(r"/api/docs/([a-f0-9]+)/analyze", path)
                    if m_analyze:
                        body = self._read_json()
                        detail = service.analyze_document(m_analyze.group(1), force_ocr=bool(body.get("force_ocr", False)))
                        self._send_json(detail)
                        return
                    if path == "/api/types/save":
                        body = self._read_json()
                        result = service.save_type_config(body)
                        self._send_json(result, status=200 if result.get("ok") else HTTPStatus.BAD_REQUEST)
                        return
                    if path == "/api/types/infer":
                        body = self._read_json()
                        payload = base64.b64decode(body["content_base64"])
                        res = service.infer_type_from_upload(
                            body["filename"],
                            payload,
                            type_id=body["type_id"],
                            name=body.get("name"),
                            description=body.get("description", ""),
                            priority=int(body.get("priority", 10)),
                        )
                        self._send_json(res)
                        return
                    if path == "/api/types/optimize":
                        body = self._read_json()
                        job = service.start_type_optimization(
                            type_id=str(body.get("type_id") or "").strip(),
                            data_dir=str(body.get("data_dir") or ""),
                            samples=body.get("samples") or None,
                            max_metric_calls=int(body.get("max_metric_calls") or 20),
                            apply=bool(body.get("apply", False)),
                        )
                        self._send_json(job, status=HTTPStatus.ACCEPTED)
                        return
                    if path == "/api/settings":
                        body = self._read_json()
                        self._send_json(service.save_settings(body))
                        return
                    self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                except (ValueError, KeyError) as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                except Exception as exc:
                    self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

            def do_DELETE(self) -> None:  # noqa: N802
                parsed = urllib.parse.urlsplit(self.path)
                path = parsed.path
                try:
                    m_doc = re.fullmatch(r"/api/docs/([a-f0-9]+)", path)
                    if m_doc:
                        service.delete_document(m_doc.group(1))
                        self._send_json({"ok": True})
                        return
                    m_type = re.fullmatch(r"/api/types/([a-z][a-z0-9_]{1,48})", path)
                    if m_type:
                        self._send_json(service.delete_type_config(m_type.group(1)))
                        return
                    self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                except FileNotFoundError as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                except Exception as exc:
                    self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        return Handler

    def serve_forever(self, *, open_browser: bool = True) -> None:
        server = ThreadingHTTPServer((self.host, self.port), self._make_handler())
        url = f"http://{self.host}:{self.port}"
        print(f"Odwołania2JSON Web GUI running at {url}  (Ctrl+C to stop)")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down web server...")
        finally:
            server.server_close()
