"""Certify the release manifest with its declared model seats connected to the test server."""

import json
import sys
from pathlib import Path

from coworld.certifier import certify_coworld
from coworld.runner.runner import run_coworld_episode
from coworld.upload import cache_certified_manifest


def certify(manifest: Path, endpoint: str, receipt: Path) -> None:
    def run_episode(job, artifacts, timeout_seconds):
        model_seats = {
            slot: {"COWORLD_LLM_ENDPOINT": endpoint}
            for slot, player in enumerate(job.players)
            if player.env.get("COWORLD_LLM_ENABLED") == "true"
        }
        run_coworld_episode(
            job,
            artifacts,
            timeout_seconds=timeout_seconds,
            player_secret_env=model_seats,
            verify_replay=False,
            validate_results=False,
            require_websocket_pong=True,
        )

    # The certifier performs and records its result and replay checks after the episode.
    result = certify_coworld(manifest, episode_runner=run_episode, timeout_seconds=120)
    cache_certified_manifest(manifest)
    steps = [step.model_dump(exclude_none=True) for step in result.step_results]
    receipt.write_text(json.dumps({"ok": True, "steps": steps}, indent=2) + "\n")
    for step in result.step_results:
        print(step.id, step.status, step.feedback or "", flush=True)


if __name__ == "__main__":
    certify(Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3]))
