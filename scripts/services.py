"""Linux local development supervisor. Never terminate a process by port number."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / '.kykeon' / 'run'
LOGS = ROOT / '.kykeon' / 'logs'
STATE = RUN / 'services.json'


def identity(pid: int) -> str | None:
    try:
        # Fields after the closing comm parenthesis begin with field 3.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def owned() -> int | None:
    try:
        state = json.loads(STATE.read_text())
        pid = state['pid']
        if not isinstance(pid, int) or pid <= 1 or identity(pid) != state['started']:
            return None
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if str(Path(__file__).resolve()).encode() not in args or b'supervise' not in args:
            return None
        return pid
    except (OSError, ValueError, KeyError, TypeError):
        return None


def stop_children(children: list[subprocess.Popen]) -> None:
    for child in children:
        if child.poll() is None:
            child.terminate()  # Uvicorn cancels/reaps model workers on shutdown.
    deadline = time.monotonic() + 25
    for child in children:
        try:
            child.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def supervise() -> int:
    children: list[subprocess.Popen] = []
    handles = []
    stopping = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.set())
    try:
        state = {'pid': os.getpid(), 'started': identity(os.getpid())}
        temporary = STATE.with_suffix('.tmp')
        temporary.write_text(json.dumps(state))
        temporary.replace(STATE)
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT / 'backend') + os.pathsep + env.get('PYTHONPATH', '')
        env['KYKEON_HOST'] = '127.0.0.1'
        env['KYKEON_PORT'] = '8765'
        env.setdefault('KYKEON_MODEL_ROOTS', str(ROOT / '.kykeon/models'))
        env.setdefault('KYKEON_ARTIFACT_ROOT', str(ROOT / '.kykeon/artifacts'))
        commands = [
            ('backend', [str(ROOT / '.venv/bin/python'), '-m', 'kykeon_backend']),
            ('frontend', ['node', str(ROOT / 'node_modules/vite/bin/vite.js'), '--host', '127.0.0.1', '--port', '5173', '--strictPort']),
        ]
        for name, command in commands:
            handle = (LOGS / f'{name}.log').open('ab')
            handles.append(handle)
            children.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                             stderr=subprocess.STDOUT, start_new_session=True))
        while not stopping.wait(0.25):
            if any(child.poll() is not None for child in children):
                return 1
        return 0
    finally:
        stop_children(children)
        for handle in handles:
            handle.close()
        STATE.unlink(missing_ok=True)


def ready(port: int, path: str) -> bool:
    try:
        # Ignore HTTP proxy environment variables for loopback checks.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f'http://127.0.0.1:{port}{path}', timeout=1) as response:
            return response.status == 200
    except OSError:
        return False


def start() -> None:
    if owned():
        print('Kykeon is already managed by make start. UI: http://127.0.0.1:5173/')
        return
    if not (ROOT / 'node_modules/vite/bin/vite.js').is_file():
        raise RuntimeError('Frontend dependencies are missing. Run make install.')
    subprocess.run([sys.executable, '-c', 'import fastapi, uvicorn, torch, transformers, safetensors'], check=True)
    if os.environ.get('KYKEON_PORT', '8765') != '8765':
        raise RuntimeError('These scripts use backend port 8765 to match the Vite proxy. Unset KYKEON_PORT.')
    for port in (5173, 8765):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(('127.0.0.1', port))
            except OSError as exc:
                raise RuntimeError(f'Port {port} is occupied. Stop its existing server first; no unrelated process was stopped.') from exc
    with (LOGS / 'supervisor.log').open('ab') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'supervise'],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    try:
        for _ in range(120):
            if process.poll() is not None:
                raise RuntimeError(f'Startup failed. See {LOGS}.')
            if ready(8765, '/docs') and ready(5173, '/'):
                print(f'Kykeon started: http://127.0.0.1:5173/\nBackend: http://127.0.0.1:8765/docs\nLogs: {LOGS}\nStop: make stop')
                return
            time.sleep(0.5)
        raise RuntimeError(f'Startup timed out. See {LOGS}.')
    except BaseException:
        process.terminate()
        process.wait(timeout=35)
        raise


def stop() -> None:
    pid = owned()
    if pid is None:
        STATE.unlink(missing_ok=True)
        print('No services started by make start are running. Other processes were left alone.')
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(140):
        if owned() != pid:
            print('Kykeon stopped.')
            return
        time.sleep(0.25)
    raise RuntimeError(f'Shutdown did not finish. Inspect {LOGS}; no unrelated process was stopped.')


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    command = sys.argv[1] if len(sys.argv) == 2 else ''
    if command == 'supervise':
        return supervise()
    if command not in {'start', 'stop'}:
        raise RuntimeError('Usage: services.py start|stop')
    with (RUN / 'control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        (start if command == 'start' else stop)()
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.SubprocessError, OSError) as exc:
        print(f'Kykeon: {exc}', file=sys.stderr)
        sys.exit(1)
