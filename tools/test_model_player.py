#!/usr/bin/env python3
"""Verify the canonical `/bin/liars-dice-player` model path and failure modes.

Tests:
1. `valid`: 1 model seat (`COWORLD_LLM_ENABLED=true`, `COWORLD_LLM_MODEL=anthropic/claude-haiku-4.5`)
   routed to a local OpenAI-compatible HTTP endpoint + 3 explicit baselines (`bayes`, `pressure`, `bayes`).
   Verifies real HTTP calls, exit 0 for game and all players, valid `results.json` and `replay.json`.
2. `fenced`: Same seat mix, with the model replying inside a single ```json ... ``` Markdown fence.
   Verifies canonical JSON reader accepts a single presentation fence without inventing actions.
3. `missing_endpoint`: `COWORLD_LLM_ENABLED=true` with NO endpoint/credentials.
   Verifies explicit policy failure: game exits != 0, NO `results.json`, `player_failure.json` written,
   trajectory `status == "failed"` with `participant_outcomes == {}`.
4. `invalid`: Model replies with prose instead of JSON.
   Verifies explicit policy failure without scores and preserved raw response.
5. `refusal`: Model returns a refusal (`content_filter`).
   Verifies explicit policy failure without scores and preserved raw response.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MOCK_SERVER = REPO_ROOT / "tools" / "ci" / "mock_llm_server.py"


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def wait_tcp(port: int, proc: subprocess.Popen, log_path: Path, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sock = socket.socket()
        ok = sock.connect_ex(("127.0.0.1", port)) == 0
        sock.close()
        if ok:
            return
        if proc.poll() is not None:
            raise RuntimeError(f"process exited early ({proc.returncode}):\n{log_path.read_text(errors='replace')}")
        time.sleep(0.05)
    raise TimeoutError(f"port {port} did not open within {timeout}s")


def run_case(game_bin: str, player_bin: str, case_name: str, root: Path, revision: str) -> dict:
    folder = root / case_name
    folder.mkdir(parents=True, exist_ok=True)
    port = free_port()
    mock_proc = None
    mock_url = ""
    mock_log = folder / "mock_calls.jsonl"

    if case_name != "missing_endpoint":
        port_file = folder / "mock.port"
        mock_proc = subprocess.Popen(
            [
                sys.executable,
                str(MOCK_SERVER),
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--port-file",
                str(port_file),
                "--log-file",
                str(mock_log),
                "--mode",
                case_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 10.0
        while not port_file.exists() or not port_file.read_text().strip():
            if mock_proc.poll() is not None:
                raise RuntimeError(f"mock server died: {mock_proc.stdout.read() if mock_proc.stdout else ''}")
            assert time.monotonic() < deadline
            time.sleep(0.05)
        mock_port = int(port_file.read_text().strip())
        mock_url = f"http://127.0.0.1:{mock_port}"

    config = {
        "players": [
            {"name": "ModelSeat"},
            {"name": "BayesSeat1"},
            {"name": "PressureSeat"},
            {"name": "BayesSeat2"},
        ],
        "tokens": ["tok-0", "tok-1", "tok-2", "tok-3"],
        "num_agents": 4,
        "seed": 11,
        "mode": "dice",
        "handSize": 5,
        "deals": 3,
        "maxBidsPerDeal": 12,
        "talk": True,
        "turnDelayMs": 0,
        "llmTimeoutSeconds": 10,
        "player_connect_timeout_seconds": 15,
    }
    if case_name == "valid":
        config["maxOutputTokens"] = 731
    config_path = folder / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Clean environment so no host LLM credentials leak into either game or players.
    base_env = {
        k: v
        for k, v in os.environ.items()
        if not any(
            k.startswith(prefix)
            for prefix in ("COWORLD_LLM", "ANTHROPIC_", "OPENAI_", "OPENROUTER_", "AWS_BEARER_", "AWS_ENDPOINT_URL_BEDROCK")
        )
    }

    game_env = dict(base_env)
    game_env.update(
        COGAME_HOST="127.0.0.1",
        COGAME_PORT=str(port),
        COGAME_CONFIG_URI=config_path.as_uri(),
        COGAME_RESULTS_URI=(folder / "results.json").as_uri(),
        COGAME_SAVE_REPLAY_URI=(folder / "replay.json").as_uri(),
        COGAME_SAVE_TRAJECTORY_URI=(folder / "trajectory.jsonl").as_uri(),
        COGAME_PLAYER_FAILURE_URI=(folder / "player_failure.json").as_uri(),
        COWORLD_EPISODE_ID=f"model-player-{case_name}",
        COWORLD_GAME_VERSION="test-local",
        COWORLD_SOURCE_REVISION=revision,
    )

    game_log = folder / "game.log"
    players: list[tuple[subprocess.Popen, Path]] = []
    try:
        with game_log.open("w") as gfh:
            game_proc = subprocess.Popen([game_bin], env=game_env, stdout=gfh, stderr=subprocess.STDOUT)
        wait_tcp(port, game_proc, game_log)

        seat_envs = [
            {
                "COWORLD_LLM_ENABLED": "true",
                "COWORLD_LLM_MODEL": "anthropic/claude-haiku-4.5",
                "COWORLD_LLM_TEMPERATURE": "0",
                **({"COWORLD_LLM_ENDPOINT": mock_url} if mock_url else {}),
            },
            {"PLAYER_SCRIPTED": "bayes"},
            {"PLAYER_SCRIPTED": "pressure"},
            {"PLAYER_SCRIPTED": "bayes"},
        ]
        for slot, extra in enumerate(seat_envs):
            p_env = dict(base_env)
            p_env["COWORLD_PLAYER_WS_URL"] = f"ws://127.0.0.1:{port}/player?slot={slot}&token=tok-{slot}"
            p_env.update(extra)
            plog = folder / f"player-{slot}.log"
            with plog.open("w") as pfh:
                p = subprocess.Popen([player_bin], env=p_env, stdout=pfh, stderr=subprocess.STDOUT)
            players.append((p, plog))

        game_rc = game_proc.wait(timeout=60)
        player_rcs = [p.wait(timeout=15) for p, _ in players]
    finally:
        for p, _ in players:
            if p.poll() is None:
                p.terminate()
        if mock_proc is not None and mock_proc.poll() is None:
            mock_proc.terminate()
            mock_proc.wait(timeout=5)

    calls = (
        [json.loads(line) for line in mock_log.read_text(encoding="utf-8").splitlines() if line.strip()]
        if mock_log.exists()
        else []
    )
    assert all(call["max_tokens"] == config.get("maxOutputTokens", 900) for call in calls)
    traj_events = [
        json.loads(line)
        for line in (folder / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expect_fail = case_name in ("missing_endpoint", "invalid", "refusal")
    if expect_fail:
        assert game_rc != 0, f"{case_name}: expected non-zero game exit, got 0\n{game_log.read_text()}"
        assert not (folder / "results.json").exists(), f"{case_name}: results.json must not exist on failure"
        assert (folder / "player_failure.json").exists(), f"{case_name}: player_failure.json missing"
        failure = json.loads((folder / "player_failure.json").read_text(encoding="utf-8"))
        assert set(failure) == {"failed_policy_index", "message"}, failure
        assert failure["failed_policy_index"] == 0, failure
        assert failure["message"].startswith("seat 0: ") and len(failure["message"]) <= 2000, failure
        assert traj_events[-1]["status"] == "failed", f"{case_name}: expected failed trajectory"
        assert traj_events[-1]["participant_outcomes"] == {}, f"{case_name}: expected empty participant_outcomes"
        if case_name in ("invalid", "refusal"):
            assert len(calls) == 1, f"{case_name}: expected exactly 1 HTTP call (no retries), got {len(calls)}"
            # Raw response must be preserved in trajectory attempts.
            failed_decision = traj_events[-2]
            assert failed_decision["attempts"], f"{case_name}: expected preserved attempt evidence"
            assert failed_decision["attempts"][0]["response"], f"{case_name}: expected preserved raw response"
    else:
        assert game_rc == 0, f"{case_name}: game exited {game_rc}\n{game_log.read_text()}"
        assert all(rc == 0 for rc in player_rcs), f"{case_name}: player exits {player_rcs}"
        assert (folder / "results.json").exists(), f"{case_name}: results.json missing"
        assert (folder / "replay.json").exists(), f"{case_name}: replay.json missing"
        assert not (folder / "player_failure.json").exists(), f"{case_name}: unexpected player_failure.json"
        assert len(calls) >= 1, f"{case_name}: expected >= 1 HTTP call from model seat, got 0"
        assert all(c["model"] == "anthropic/claude-haiku-4.5" for c in calls), f"unexpected models: {calls}"
        assert traj_events[-1]["status"] == "completed", f"{case_name}: trajectory did not complete"

    return {
        "case": case_name,
        "game_exit": game_rc,
        "player_exits": player_rcs,
        "http_calls": len(calls),
        "results_written": (folder / "results.json").exists(),
        "player_failure_written": (folder / "player_failure.json").exists(),
    }


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(f"usage: {sys.argv[0]} <liars-dice-bin> <liars-dice-player-bin> [revision]")
    game_bin = sys.argv[1]
    player_bin = sys.argv[2]
    revision = sys.argv[3] if len(sys.argv) > 3 else "local-test"
    with tempfile.TemporaryDirectory(prefix="liars-dice-model-test-") as tmp:
        root = Path(tmp)
        summary = [
            run_case(game_bin, player_bin, case, root, revision)
            for case in ("valid", "fenced", "missing_endpoint", "invalid", "refusal")
        ]
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
