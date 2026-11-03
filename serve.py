#!/usr/bin/env python3
"""Local web server for the BRIGHT viewer (standard library only).

    python build.py      # once, and after new evaluation runs (or use the page's "Rescan runs" button)
    python serve.py      # then open http://127.0.0.1:8765

Serves the low-res proxies from the cache and renders full-resolution layers on demand straight from the
dataset. Binds to localhost only, since it serves your local copy of the dataset.
"""
import argparse
import json
import os
import re
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import build

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
CTYPES = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
          '.json': 'application/json', '.jpg': 'image/jpeg', '.png': 'image/png', '.svg': 'image/svg+xml'}
NAME = r'[A-Za-z0-9_\-]+'
RUN = r'[\w.\-]+'


class State:
    def __init__(self, args):
        self.args = args
        meta_path = os.path.join(args.cache, 'meta.json')
        if not os.path.isfile(meta_path):
            raise SystemExit(f'no cache at {args.cache} - run "python build.py" first')
        with open(meta_path, 'rb') as f:
            self.meta_bytes = f.read()
        self.meta = json.loads(self.meta_bytes)
        self.tiles = set(self.meta['tiles']['name'])
        self.dataset = args.dataset or self.meta.get('dataset') or build.DEFAULT_DATASET   # default: what build.py used
        if not os.path.isdir(os.path.join(self.dataset, 'target')):
            print(f'warning: no dataset at {self.dataset} - full-resolution views will fail (pass --dataset)')
        self.lock = threading.Lock()
        self.rescan = {'running': False, 'msg': ''}
        self.load_runs()

    def load_runs(self):
        runs, root = {}, os.path.join(self.args.cache, 'runs')
        if os.path.isdir(root):
            for d in sorted(os.listdir(root)):
                r = build.load_json(os.path.join(root, d, 'run.json'), None)
                if r:
                    runs[d] = r
        self.runs = runs

    def run_list(self):
        return [{k: r.get(k) for k in ('name', 'pred_dir', 'n', 'built', 'summary')} for r in self.runs.values()]

    def do_rescan(self):
        logs = []
        try:
            built = build.build_runs(self.dataset, self.args.cache, self.args.results, self.args.run, self.meta,
                                     self.args.workers, log=logs.append)
            self.load_runs()
            msg = f'new/updated runs: {", ".join(built)}' if built else 'no new evaluation folders found'
        except Exception as e:   # report to the page instead of dying silently
            traceback.print_exc()
            msg = f'rescan failed: {e}'
        with self.lock:
            self.rescan = {'running': False, 'msg': msg, 'log': logs[-20:]}


class Handler(BaseHTTPRequestHandler):
    server_version = 'BRIGHTViewer/1.0'

    def log_message(self, fmt, *args):   # keep the terminal quiet; errors are printed explicitly
        pass

    def send(self, code, ctype, body, cache='no-cache'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', cache)
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj, code=200):
        self.send(code, 'application/json', json.dumps(obj, separators=(',', ':')).encode())

    def file(self, path, cache='no-cache'):
        if not path or not os.path.isfile(path):
            return self.json({'error': 'not found'}, 404)
        with open(path, 'rb') as f:
            body = f.read()
        self.send(200, CTYPES.get(os.path.splitext(path)[1], 'application/octet-stream'), body, cache)

    def do_GET(self):
        st = self.server.st
        path = unquote(urlparse(self.path).path)
        try:
            if path in ('/', '/index.html'):
                return self.file(os.path.join(STATIC, 'index.html'))
            m = re.fullmatch(r'/static/([\w.\-]+)', path)
            if m:
                return self.file(os.path.join(STATIC, m[1]))
            if path == '/api/meta':
                return self.send(200, 'application/json', st.meta_bytes)
            if path == '/api/runs':
                return self.json(st.run_list())
            if path == '/api/rescan':
                with st.lock:
                    return self.json(st.rescan)
            m = re.fullmatch(rf'/api/run/({RUN})', path)
            if m:
                return self.json(st.runs[m[1]]) if m[1] in st.runs else self.json({'error': 'unknown run'}, 404)
            m = re.fullmatch(rf'/p/(64|256)/(opt|sar|gt)/({NAME})\.(jpg|png)', path)
            if m:
                return self.file(os.path.join(st.args.cache, f'p{m[1]}', m[2], f'{m[3]}.{m[4]}'), 'max-age=86400')
            m = re.fullmatch(r'/atlas/(opt|sar|gt)/(scene_\d+)\.png', path)
            if m:
                return self.file(os.path.join(st.args.cache, 'atlas', m[1], f'{m[2]}.png'), 'max-age=86400')
            m = re.fullmatch(rf'/runs/({RUN})/p(64|256)/(pred|err)/({NAME})\.png', path)
            if m and m[1] in st.runs:
                return self.file(os.path.join(st.args.cache, 'runs', m[1], f'p{m[2]}', m[3], f'{m[4]}.png'), 'max-age=86400')
            m = re.fullmatch(rf'/runs/({RUN})/atlas/(pred|err)/(scene_\d+)\.png', path)
            if m and m[1] in st.runs:
                return self.file(os.path.join(st.args.cache, 'runs', m[1], 'atlas', m[2], f'{m[3]}.png'), 'max-age=86400')
            m = re.fullmatch(rf'/full/(opt|sar|gt)/({NAME})', path)
            if m and m[2] in st.tiles:
                ctype, body = build.render_full(st.dataset, m[1], m[2])
                return self.send(200, ctype, body, 'max-age=86400')
            m = re.fullmatch(rf'/full/(pred|err)/({RUN})/({NAME})', path)
            if m and m[3] in st.tiles and m[2] in st.runs:
                ctype, body = build.render_full(st.dataset, m[1], m[3], st.runs[m[2]]['pred_dir'])
                return self.send(200, ctype, body, 'max-age=3600')
            return self.json({'error': 'not found'}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except FileNotFoundError:   # e.g. a tile that has no prediction in this run
            self.json({'error': 'not found'}, 404)
        except Exception:
            traceback.print_exc()
            try:
                self.json({'error': 'server error'}, 500)
            except OSError:
                pass

    def do_POST(self):
        st = self.server.st
        if urlparse(self.path).path != '/api/rescan':
            return self.json({'error': 'not found'}, 404)
        with st.lock:
            if not st.rescan['running']:
                st.rescan = {'running': True, 'msg': 'scanning for new evaluation folders ...'}
                threading.Thread(target=st.do_rescan, daemon=True).start()
            return self.json(st.rescan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--dataset', default=None, help='default: the dataset folder build.py used')
    ap.add_argument('--results', default=build.DEFAULT_RESULTS)
    ap.add_argument('--cache', default=build.DEFAULT_CACHE)
    ap.add_argument('--run', action='append', default=[], metavar='NAME=DIR')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    httpd.st = State(args)
    print(f'BRIGHT viewer: http://{args.host}:{args.port}  ({len(httpd.st.tiles)} tiles, {len(httpd.st.runs)} runs)  Ctrl+C to stop', flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
