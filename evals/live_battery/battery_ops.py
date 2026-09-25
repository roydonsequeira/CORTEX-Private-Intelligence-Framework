"""Tests 40-44 of the plan plus API/security checks that need more than one chat turn.

- 40: Ollama down -> clean error event, server stays up (starts a 2nd server on :8012)
- 41: two chats at the same moment -> both finish, no crossed answers
- 42: POST /chat/message -> JSON answer + session id
- 43: 9**9**9 in one chat does not block another chat
- 44: `cortex doctor` passes, and names Ollama when it is unreachable
- S1-S6: CORS, Host check (DNS rebinding), debug tool endpoint off, tools listed, validation
"""

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
from harness import BASE, WORKSPACE, run_turn

RESULTS: list[tuple[str, bool, str]] = []


def cortex_command(*args: str) -> list[str]:
    """The installed `cortex` CLI, or the same entry point via this Python."""
    exe = shutil.which("cortex")
    if exe:
        return [exe, *args]
    return [
        sys.executable,
        "-c",
        "import sys; from cortex.cli import main; sys.exit(main(sys.argv[1:]))",
        *args,
    ]


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)


def test40_ollama_down() -> None:
    data = Path(tempfile.mkdtemp(prefix="cortex-eval-40-"))
    env = {
        **os.environ,
        "CORTEX_OLLAMA_BASE_URL": "http://127.0.0.1:9",  # nothing listens here
        "CORTEX_WARMUP_ON_STARTUP": "false",
        "CORTEX_DB_PATH": str(data / "cortex.db"),
        "CORTEX_CHROMA_PATH": str(data / "chroma"),
        "CORTEX_ALLOWED_ROOT": str(WORKSPACE),
    }
    with open(data / "server.log", "w") as log:
        proc = subprocess.Popen(
            cortex_command("serve", "--port", "8012"), env=env, stdout=log, stderr=subprocess.STDOUT
        )
    try:
        with httpx.Client(timeout=120) as client:
            for _ in range(60):
                try:
                    client.get("http://127.0.0.1:8012/health")
                    break
                except httpx.HTTPError:
                    time.sleep(1)
            health = client.get("http://127.0.0.1:8012/health").json()
            with client.stream(
                "POST", "http://127.0.0.1:8012/chat/stream", json={"message": "hi"}
            ) as response:
                body = response.read().decode()
            errors = [
                line
                for line in body.splitlines()
                if '"type": "error"' in line or '"type":"error"' in line
            ]
            alive = client.get("http://127.0.0.1:8012/health").status_code == 200
        ok = (
            bool(errors)
            and "ollama" in errors[0].lower()
            and alive
            and health.get("ollama") is False
        )
        record(
            "40 ollama down",
            ok,
            f"health.ollama={health.get('ollama')} error={errors[0][:160] if errors else None} alive={alive}",
        )
    finally:
        proc.terminate()
        proc.wait(10)


def test41_concurrency() -> None:
    out: dict[str, dict] = {}

    def go(key: str, prompt: str) -> None:
        with httpx.Client(timeout=600) as client:
            out[key] = run_turn(client, prompt, None)

    threads = [
        threading.Thread(
            target=go, args=("fib", "Use Python to compute the 20th Fibonacci number")
        ),
        threading.Thread(
            target=go, args=("primes", "Write Python code to find the primes below 50 and run it")
        ),
    ]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    fib, primes = out["fib"]["answer"], out["primes"]["answer"]
    ok = (
        "6765" in fib.replace(",", "")
        and "47" in primes
        and "6765" not in primes
        and "47" not in fib
    )
    record(
        "41 two chats at once",
        ok,
        f"{round(time.monotonic() - started, 1)}s fib={fib[:60]!r} primes={primes[:60]!r}",
    )


def test42_message_api() -> None:
    with httpx.Client(timeout=120) as client:
        response = client.post(f"{BASE}/chat/message", json={"message": "What is 2+2?"})
        data = response.json()
    ok = (
        response.status_code == 200
        and "4" in (data.get("final_answer") or "")
        and bool(data.get("session_id"))
    )
    record(
        "42 POST /chat/message",
        ok,
        f"status={response.status_code} answer={data.get('final_answer')!r}",
    )


def test43_big_power_does_not_block() -> None:
    out: dict[str, dict] = {}

    def big() -> None:
        with httpx.Client(timeout=600) as client:
            out["big"] = run_turn(client, "Use Python to compute 9**9**9", None)

    thread = threading.Thread(target=big)
    thread.start()
    time.sleep(4)
    with httpx.Client(timeout=120) as client:
        hi = run_turn(client, "hi", None)
    thread.join()
    ok = hi["seconds"] < 20 and bool(hi["answer"]) and out["big"]["done"]
    record("43 9**9**9 + hi", ok, f"hi took {hi['seconds']}s; big took {out['big']['seconds']}s")


def test44_doctor() -> None:
    env = {**os.environ, "CORTEX_API_PORT": "8013", "PYTHONIOENCODING": "utf-8"}
    good = subprocess.run(
        cortex_command("doctor"), env=env, capture_output=True, text=True, encoding="utf-8"
    )
    bad = subprocess.run(
        cortex_command("doctor"),
        env={**env, "CORTEX_OLLAMA_BASE_URL": "http://127.0.0.1:9"},
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    good_ok = good.returncode == 0 and "[FAIL]" not in good.stdout
    bad_ok = bad.returncode != 0 and "[FAIL] Ollama" in bad.stdout
    record(
        "44 cortex doctor",
        good_ok and bad_ok,
        f"good rc={good.returncode}; unreachable Ollama rc={bad.returncode}",
    )


def test_api_security() -> None:
    with httpx.Client(timeout=30) as client:
        evil = client.options(
            f"{BASE}/chat/message",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
        record(
            "S1 CORS blocks a remote origin",
            "access-control-allow-origin" not in evil.headers,
            f"status={evil.status_code}",
        )
        ui = client.options(
            f"{BASE}/chat/message",
            headers={"Origin": "http://localhost:3001", "Access-Control-Request-Method": "POST"},
        )
        allowed = ui.headers.get("access-control-allow-origin")
        record(
            "S2 CORS allows the UI on another local port",
            allowed == "http://localhost:3001",
            f"acao={allowed}",
        )
        rebind = client.get(f"{BASE}/health", headers={"Host": "attacker.example"})
        record(
            "S3 Host check blocks DNS rebinding",
            rebind.status_code == 400,
            f"status={rebind.status_code}",
        )
        tool = client.post(f"{BASE}/tools/python_exec/execute", json={"code": "print(1)"})
        record(
            "S4 debug tool endpoint is off",
            tool.status_code in (403, 404),
            f"status={tool.status_code}",
        )
        names = sorted(t["name"] for t in client.get(f"{BASE}/tools").json())
        record("S5 GET /tools lists the tools", len(names) >= 5, str(names))
        bad_session = client.post(
            f"{BASE}/chat/message", json={"message": "hi", "session_id": "not-a-uuid"}
        )
        record(
            "S6 invalid session_id is rejected",
            bad_session.status_code == 422,
            f"status={bad_session.status_code}",
        )


if __name__ == "__main__":
    for test in (
        test42_message_api,
        test_api_security,
        test41_concurrency,
        test43_big_power_does_not_block,
        test44_doctor,
        test40_ollama_down,
    ):
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - report and continue
            record(test.__name__, False, f"crashed: {type(exc).__name__}: {exc}")
    print(
        f"\nbattery_ops: {sum(ok for _, ok, _ in RESULTS)} passed, {sum(not ok for _, ok, _ in RESULTS)} failed"
    )
