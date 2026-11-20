"""
benchlib_variants.py -- the engine and model registries, the command builders and the variants.

Split out of benchlib.py, which re-exports every name defined here. A variant is one way of running
a model ({label, engine, cmd, ppl, env[, modified]}); every benchmark script builds its variants with
`pz_variant` / `ref_variant`, so command construction exists once.
"""

import json
import os
import pathlib
import sys
from collections.abc import Sequence
from typing import Any, NotRequired, Protocol, TypedDict

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from bench_sweep_seq import PROMPT  # reused, not copied


class GenCmd(Protocol):
    """Builds the generation command of a variant for one model, or None when it cannot run it."""

    def __call__(
        self,
        key: str,
        info: dict[str, Any],
        path: pathlib.Path,
        tokens: int,
        threads: int,
        fixed: bool,
        prompt: str = ...,
    ) -> list[str] | None:
        """Call."""


class PplCmd(Protocol):
    """Builds the perplexity command of a variant for one model, or None when it cannot run it."""

    def __call__(
        self,
        key: str,
        info: dict[str, Any],
        path: pathlib.Path,
        text: str | pathlib.Path,
        ctx: int,
        chunks: int,
        threads: int,
        extra: Sequence[str] = ...,
    ) -> list[str] | None:
        """Call."""


class Variant(TypedDict):
    """One way of running a model (see pz_variant / ref_variant)."""

    label: str
    engine: str
    cmd: GenCmd
    ppl: PplCmd
    env: dict[str, str] | None
    modified: NotRequired[bool]


# ── registries ──────────────────────────────────────────────────────────────────────────────
def load_engines() -> dict[str, dict[str, Any]]:
    """Engines listed in engines.json, without the "_" comment keys."""
    d = json.loads((HERE / "engines.json").read_text())
    return {k: v for k, v in d.items() if not k.startswith("_")}


def load_models() -> dict[str, dict[str, Any]]:
    """Models listed in models.json, without the "_" comment keys."""
    d = json.loads((HERE / "models.json").read_text())
    return {k: v for k, v in d.items() if not k.startswith("_")}


def default_engine_dir() -> pathlib.Path:
    """Directory holding the reference engines: $PZ_BENCH_ENGINE_DIR or ~/ext."""
    return pathlib.Path(os.environ.get("PZ_BENCH_ENGINE_DIR") or pathlib.Path.home() / "ext")


def engine_binary(key: str, engine_dir: str | pathlib.Path) -> pathlib.Path | None:
    """Path of the named engine's binary under `engine_dir`, or None if it is not built."""
    e = load_engines()[key]
    p = pathlib.Path(engine_dir) / e["dir"] / e["binary"]
    return p if p.exists() else None


def model_path(key: str, models_dir: str | pathlib.Path = "models") -> pathlib.Path | None:
    """Path of the named model file under `models_dir`, or None if it is not downloaded."""
    p = pathlib.Path(models_dir) / load_models()[key]["file"]
    return p if p.exists() else None


def pz_cmd(
    pz: str | pathlib.Path,
    model: str | pathlib.Path,
    tokens: int,
    threads: int,
    extra: Sequence[str] = (),
    fixed_length: bool = True,
    prompt: str = PROMPT,
    flags: Sequence[str] = (),
) -> list[str]:
    """Command line (list of arguments) that runs pz on a model with the given token count, threads and extra flags."""
    cmd = (
        [
            str(pz),
            "--model",
            str(model),
            "--prompt",
            prompt,
            "--max-tokens",
            str(tokens),
            "--temperature",
            "0",
            "--threads",
            str(threads),
            "--no-update-check",
        ]
        + list(flags)
        + list(extra)
    )
    if fixed_length:
        cmd += ["--raw-prompt", "--ignore-eos"]
    return cmd


def ref_cmd(
    binary: str | pathlib.Path,
    runner: str,
    model: str | pathlib.Path,
    tokens: int,
    threads: int,
    extra: Sequence[str] = (),
    fixed_length: bool = True,
    prompt: str = PROMPT,
) -> list[str]:
    """llama.cpp family. runner 'ref' = llama-completion (-no-cnv), 'cli' = llama-cli (ik_llama.cpp)."""
    cmd = [
        str(binary),
        "-m",
        str(model),
        "-p",
        prompt,
        "-n",
        str(tokens),
        "--temp",
        "0",
        "-t",
        str(threads),
    ]
    if runner == "ref":
        cmd.append("-no-cnv")
    cmd += list(extra)
    if fixed_length:
        cmd.append("--ignore-eos")
    return cmd


def ppl_cmd_pz(
    pz: str | pathlib.Path,
    model: str | pathlib.Path,
    text: str | pathlib.Path,
    ctx: int,
    chunks: int,
    threads: int,
    extra: Sequence[str] = (),
) -> list[str]:
    """pz perplexity run (llama-perplexity's method, built into pz)."""
    return [
        str(pz),
        "--model",
        str(model),
        "--perplexity",
        str(text),
        "--ppl-ctx",
        str(ctx),
        "--ppl-chunks",
        str(chunks),
        "--threads",
        str(threads),
        "--no-update-check",
    ] + list(extra)


def ppl_cmd_ref(
    binary: str | pathlib.Path,
    model: str | pathlib.Path,
    text: str | pathlib.Path,
    ctx: int,
    chunks: int,
    threads: int,
    extra: Sequence[str] = (),
) -> list[str]:
    """llama-perplexity of the llama.cpp family."""
    return [
        str(binary),
        "-m",
        str(model),
        "-f",
        str(text),
        "-c",
        str(ctx),
        "--chunks",
        str(chunks),
        "-t",
        str(threads),
    ] + list(extra)


def engine_ppl_binary(key: str, engine_dir: str | pathlib.Path) -> pathlib.Path | None:
    """Path of the named engine's perplexity binary under `engine_dir`, or None when it has none or is not built."""
    spec = load_engines()[key].get("perplexity")
    p = pathlib.Path(engine_dir) / load_engines()[key]["dir"] / spec["binary"] if spec else None
    return p if p and p.exists() else None


# A variant is one way of running a model: {label, engine, cmd, ppl, env[, modified]}.
#   cmd(key, info, path, tokens, threads, fixed_length, prompt=PROMPT) -> argv | None   (generation)
#   ppl(key, info, path, text, ctx, chunks, threads, extra=())        -> argv | None   (gate perplexity)
# None = the variant cannot run that model. Every script builds its variants with these two factories,
# so command construction exists once.
def pz_variant(
    label: str,
    pz: str | pathlib.Path,
    flags: Sequence[str] = (),
    env: dict[str, str] | None = None,
) -> Variant:
    """Variant description for pz: its label, how to build the generation and perplexity commands, and the environment."""

    def cmd(
        key: str,
        info: dict[str, Any],
        path: pathlib.Path,
        tokens: int,
        threads: int,
        fixed: bool,
        prompt: str = PROMPT,
    ) -> list[str] | None:
        """Generation command for this pz variant, or None when the model has no pz engine."""
        return (
            pz_cmd(pz, path, tokens, threads, flags, fixed, prompt)
            if "pz" in info["engines"]
            else None
        )

    def ppl(
        key: str,
        info: dict[str, Any],
        path: pathlib.Path,
        text: str | pathlib.Path,
        ctx: int,
        chunks: int,
        threads: int,
        extra: Sequence[str] = (),
    ) -> list[str] | None:
        """Perplexity command for this pz variant, or None when the model has no pz engine."""
        return (
            ppl_cmd_pz(pz, path, text, ctx, chunks, threads, list(flags) + list(extra))
            if "pz" in info["engines"]
            else None
        )

    return {"label": label, "engine": "pz", "cmd": cmd, "ppl": ppl, "env": env}


def ref_variant(
    label: str,
    engine: str,
    binary: str | pathlib.Path,
    ppl_binary: str | pathlib.Path | None = None,
    env: dict[str, str] | None = None,
) -> Variant:
    """Variant description for a reference engine: label, command builders for generation and perplexity, and the environment."""
    e = load_engines()[engine]

    def cmd(
        key: str,
        info: dict[str, Any],
        path: pathlib.Path,
        tokens: int,
        threads: int,
        fixed: bool,
        prompt: str = PROMPT,
    ) -> list[str] | None:
        """Generation command for this reference engine, or None when the model does not list it."""
        if engine not in info["engines"]:
            return None
        return ref_cmd(
            binary,
            e["runner"],
            path,
            tokens,
            threads,
            info.get("extra", {}).get(engine, []),
            fixed,
            prompt,
        )

    def ppl(
        key: str,
        info: dict[str, Any],
        path: pathlib.Path,
        text: str | pathlib.Path,
        ctx: int,
        chunks: int,
        threads: int,
        extra: Sequence[str] = (),
    ) -> list[str] | None:
        """Perplexity command for this reference engine, or None when it is not listed or has no perplexity binary."""
        if engine not in info["engines"] or not ppl_binary:
            return None
        return ppl_cmd_ref(ppl_binary, path, text, ctx, chunks, threads, extra)

    return {
        "label": label,
        "engine": engine,
        "cmd": cmd,
        "ppl": ppl,
        "env": env,
        "modified": bool(e.get("modified")),
    }
