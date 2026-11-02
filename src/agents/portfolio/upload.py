"""Local-only CSV/XLSX upload interface, sharing the CLI analysis pipeline."""
from __future__ import annotations

import argparse
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from agents.portfolio.holdings import MAX_FILE_BYTES, load_holdings
from agents.portfolio.report import build_report
from rag.store import local_collection
from rag.ingest import DEFAULT_PDF_DIR

PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Finnie · Portfolio analysis</title><style>
body{font:17px/1.6 system-ui;background:#f7f8fa;color:#16283b;max-width:1000px;margin:48px auto;padding:0 24px}
section{background:white;padding:28px;border:1px solid #dbe2e8;border-radius:12px;margin:20px 0}
button{background:#175c59;color:white;border:0;border-radius:7px;padding:12px 20px;cursor:pointer;font:inherit}
button:disabled{opacity:.5}input{font:inherit;margin:16px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 ui-monospace,monospace}
a{color:#175c59}label{display:block}#status{min-height:28px}h1{line-height:1.2}</style>
<h1>Understand your portfolio</h1><p>Upload holdings to calculate allocation and concentration, with financial context from your local PDF library.</p>
<section><h2>Current holdings</h2><p>CSV or XLSX, up to 10 MB. Required: <code>ticker</code>, <code>market_value_usd</code>, and <code>asset_type</code> or <code>asset_class</code>. USD, long-only portfolios.</p>
<p>Optional: sector, sector_source, portfolio, as_of_date (YYYY-MM-DD), age, risk_tolerance, goal. Use asset_type <code>Stock</code> or <code>US Equity ETF</code>, for example. For non-US stocks, supply an explicit asset_class.</p>
<p><a href="/template.csv" download="holdings_template.csv">Download CSV template</a>. Paste values into Excel; formulas are not accepted. Missing valuation dates are shown as unknown.</p>
<input id="file" type="file" accept=".csv,.xlsx" aria-label="Holdings file"><label><input id="rag" type="checkbox" checked> Include financial knowledge from the PDF library</label>
<label><input id="llm" type="checkbox" checked> Generate AI explanation (sends the report, goal and selected PDF excerpts to OpenAI)</label>
<button id="run">Analyze holdings</button><p id="status" role="status"></p></section>
<div id="reports"></div><script>
const token='__TOKEN__';
document.querySelector('#run').onclick=async()=>{
 const file=document.querySelector('#file').files[0], status=document.querySelector('#status'), button=document.querySelector('#run');
 if(!file){status.textContent='Choose a CSV or XLSX file first.';return}
 if(file.size>10*1024*1024){status.textContent='File exceeds 10 MB.';return}
 button.disabled=true;status.textContent='Validating holdings, calculating metrics and retrieving knowledge…';document.querySelector('#reports').replaceChildren();
 try{
 const response=await fetch('/analyze?filename='+encodeURIComponent(file.name)+'&rag='+document.querySelector('#rag').checked+'&llm='+document.querySelector('#llm').checked,{method:'POST',headers:{'X-Upload-Token':token},body:file});
 const data=await response.json();if(!response.ok)throw Error(data.error);
 for(const report of data.reports){
 const section=document.createElement('section'), title=document.createElement('h2');title.textContent=report.analysis.portfolio.name;section.append(title);
 for(const [label,content,type] of [['Download report',report.markdown,'text/markdown'],['Download metrics JSON',JSON.stringify(report,null,2),'application/json']]){
 const a=document.createElement('a');a.textContent=label;a.style.marginRight='24px';a.href=URL.createObjectURL(new Blob([content],{type}));a.download=label.includes('JSON')?'portfolio.json':'portfolio.md';section.append(a)}
 const pre=document.createElement('pre');pre.textContent=report.markdown;section.append(pre);
 for(const knowledge of report.knowledge){for(const chunk of knowledge.retrieval.chunks){const a=document.createElement('a');a.href='/source?name='+encodeURIComponent(chunk.source)+'#page='+chunk.page;a.target='_blank';a.rel='noopener';a.textContent=chunk.source+' · page '+chunk.page;a.style.display='block';section.append(a)}}
 document.querySelector('#reports').append(section)}
 status.textContent='Analysis complete. Values come entirely from your upload.';
 }catch(error){status.textContent=error.message}finally{button.disabled=false}
};</script></html>'''

TEMPLATE = 'ticker,asset_type,market_value_usd,sector,sector_source,portfolio,as_of_date\nAAPL,Stock,6000,Information Technology,User supplied,My portfolio,\nBND,Bond ETF,4000,Not applicable,Non-equity instrument,My portfolio,\n'


def make_handler(token: str):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, content, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send(200, PAGE.replace("__TOKEN__", token).encode(), "text/html; charset=utf-8")
            elif parsed.path == "/template.csv":
                self.send(200, TEMPLATE.encode(), "text/csv")
            elif parsed.path == "/source":
                name = parse_qs(parsed.query).get("name", [""])[0]
                path = DEFAULT_PDF_DIR / name
                if path.suffix.lower() != ".pdf" or path.resolve().parent != DEFAULT_PDF_DIR.resolve() or not path.is_file():
                    self.send(404, b"PDF not found", "text/plain")
                else:
                    self.send(200, path.read_bytes(), "application/pdf")
            else:
                self.send(404, b"Not found", "text/plain")

        def do_POST(self):
            if self.headers.get("X-Upload-Token") != token:
                self.send(403, b'{"error":"Invalid upload token"}', "application/json")
                return
            try:
                parsed = urlparse(self.path)
                if parsed.path != "/analyze":
                    raise ValueError("Unknown upload endpoint")
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_FILE_BYTES:
                    raise ValueError("File must be nonempty and at most 10 MB")
                query = parse_qs(parsed.query)
                portfolios = load_holdings(self.rfile.read(length), query.get("filename", [""])[0])
                warning = None
                collection = None
                if query.get("rag", ["true"])[0] == "true":
                    try:
                        collection = local_collection()
                    except Exception:
                        warning = "Local RAG index unavailable; metrics were calculated successfully."
                results = []
                for portfolio in portfolios:
                    report = build_report(portfolio, collection, use_llm=query.get("llm", ["true"])[0] == "true")
                    if warning:
                        from dataclasses import replace
                        report = replace(report, warnings=(warning,))
                    results.append({**report.to_dict(), "markdown": report.to_markdown()})
                self.send(200, json.dumps({"reports": results}, allow_nan=False).encode(), "application/json")
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                self.send(400, json.dumps({"error": str(exc)}).encode(), "application/json")
            except Exception:
                self.send(500, b'{"error":"Unable to analyze file. Check the format and try again."}', "application/json")

        def log_message(self, format, *args):
            pass  # Do not log uploaded filenames or portfolio content.
    return Handler


def main():
    parser = argparse.ArgumentParser(description="Local portfolio upload page")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(secrets.token_urlsafe(32)))
    print(f"Portfolio upload: http://127.0.0.1:{args.port} (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
