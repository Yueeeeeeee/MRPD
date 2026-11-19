#!/usr/bin/env python3
"""Prepare the text-to-SQL domain: download Spider and BIRD mini-dev, lay every database
out as <out>/databases/<db_id>/<db_id>.sqlite, and write the prompt files in the jsonl
shape the training and eval scripts read:

    {"prompt": "<schema + question>", "label": "<gold SQL>",
     "metadata": {"db_id": ..., "source_name": "sql", "dataset": ..., "evidence": ..., ...}}

    python3 scripts/data/prepare_sql.py --out /workspace/data/sql              # Spider + BIRD mini-dev
    python3 scripts/data/prepare_sql.py --out /workspace/data/sql --no-bird    # Spider only

Writes
    <out>/databases/<db_id>/<db_id>.sqlite   symlinks into the extracted archives (--copy-db copies)
    <out>/train.jsonl        Spider train, gold queries executed and filtered (errors, timeouts,
                             empty results and duplicate questions dropped), then NUM_TRAIN
                             sampled with a fixed seed
    <out>/spider_dev.jsonl   Spider dev, all 1034 questions
    <out>/spider_test.jsonl  Spider test, 2147 questions on 40 unseen databases, when the archive
                             carries the 2024 test release (test.json + test_database/)
    <out>/bird_minidev.jsonl BIRD mini-dev, all 500 questions, the evidence as a hint (eval only)
    <out>/stats.json         counts of every step above

Sources: HAL-9001/spider-databases (spider_data.zip, a re-host of the Yale archive with the
SQLite files, checksummed), birdsql/bird_mini_dev (questions) and the BIRD dev archive
(databases). --spider-zip / --bird-dev-zip skip the downloads for archives already on disk.
The prompt's schema text is sql_tools.build_schema_text: CREATE TABLE statements plus the
first three rows of each table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train" / "src"))
import sql_tools  # noqa: E402

SPIDER_REPO = "HAL-9001/spider-databases"
SPIDER_FILE = "spider_data.zip"
SPIDER_SHA256 = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
BIRD_MINIDEV_REPO = "birdsql/bird_mini_dev"
BIRD_MINIDEV_FILE = "data/mini_dev_sqlite-00000-of-00001.json"
BIRD_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"


def log(message: str) -> None:
    print(f"[prepare_sql] {message}", flush=True)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hf_download(repo: str, filename: str, local_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    log(f"downloading {repo}/{filename} -> {local_dir}")
    return Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", local_dir=str(local_dir)))


def url_download(url: str, dest: Path) -> Path:
    if dest.exists():
        log(f"already downloaded: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done, last = 0, time.time()
        for chunk in iter(lambda: resp.read(1 << 20), b""):
            out.write(chunk)
            done += len(chunk)
            if time.time() - last > 10:
                log(f"  {done / 1e6:.0f} / {total / 1e6:.0f} MB")
                last = time.time()
    tmp.replace(dest)
    return dest


def extract_zip(zip_path: Path, dest: Path) -> Path:
    """Extract, then extract any zip found inside (BIRD nests its databases)."""
    marker = dest / ".extracted"
    if marker.exists():
        log(f"already extracted: {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    log(f"extracting {zip_path} -> {dest}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    for inner in sorted(dest.rglob("*.zip")):
        if inner.resolve() == zip_path.resolve():
            continue
        log(f"extracting nested {inner}")
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(inner.parent)
    marker.touch()
    return dest


def find_sqlite_dbs(root: Path) -> dict[str, Path]:
    """db_id -> path for every <db_id>/<db_id>.sqlite under root (macOS metadata skipped)."""
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*.sqlite")):
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        db_id = path.stem
        if path.parent.name != db_id:
            continue
        found.setdefault(db_id, path)
    return found


def link_databases(db_map: dict[str, Path], databases_dir: Path, copy: bool) -> None:
    for db_id, src in db_map.items():
        target_dir = databases_dir / db_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{db_id}.sqlite"
        if target.is_symlink() or target.exists():
            target.unlink()
        if copy:
            shutil.copy2(src, target)
        else:
            target.symlink_to(src.resolve())


def find_file(root: Path, name: str) -> Path:
    matches = [p for p in root.rglob(name) if "__MACOSX" not in p.parts]
    if not matches:
        raise FileNotFoundError(f"{name} not found under {root}")
    return sorted(matches, key=lambda p: len(p.parts))[0]


class SchemaCache:
    def __init__(self, databases_dir: Path):
        self.databases_dir = databases_dir
        self.cache: dict[str, str] = {}

    def get(self, db_id: str) -> str:
        if db_id not in self.cache:
            path = sql_tools.resolve_db_path(db_id, str(self.databases_dir))
            self.cache[db_id] = sql_tools.build_schema_text(path)
        return self.cache[db_id]


def make_row(schemas: SchemaCache, dataset: str, db_id: str, question: str, sql: str, evidence: str = "", **extra) -> dict:
    metadata = {"db_id": db_id, "source_name": "sql", "dataset": dataset, "evidence": evidence or ""}
    metadata.update({k: v for k, v in extra.items() if v is not None})
    return {"prompt": sql_tools.build_user_prompt(schemas.get(db_id), question, evidence), "label": sql.strip(), "metadata": metadata}


def filter_train_rows(rows: list[dict], databases_dir: Path, timeout: float) -> tuple[list[dict], dict]:
    """Keep rows whose gold query runs and returns at least one row; drop duplicate questions."""
    stats = {"input": len(rows), "gold_error": 0, "gold_empty": 0, "duplicate": 0, "kept": 0}
    kept, seen = [], set()
    for i, row in enumerate(rows, 1):
        key = (row["metadata"]["db_id"], " ".join(row["prompt"].split("Question:")[-1].split()).lower())
        if key in seen:
            stats["duplicate"] += 1
            continue
        result, _, error = sql_tools.run_query(row["metadata"]["db_id"], row["label"], timeout, db_root=str(databases_dir))
        if result is None:
            stats["gold_error"] += 1
            continue
        if len(result) == 0:
            stats["gold_empty"] += 1
            continue
        seen.add(key)
        kept.append(row)
        if i % 1000 == 0:
            log(f"  filtered {i}/{len(rows)}: kept {len(kept)}")
    stats["kept"] = len(kept)
    return kept, stats


def load_json_or_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else list(data.values())
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"wrote {len(rows)} rows -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory (becomes SQL_DB_ROOT's parent)")
    ap.add_argument("--num-train", type=int, default=6400, help="training prompts to keep after filtering (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bird", dest="bird", action="store_false",
                    help="skip BIRD mini-dev (prepared by default; its dev archive is a 350 MB download)")
    ap.set_defaults(bird=True)
    ap.add_argument("--spider-zip", default=None, help="spider_data.zip already on disk (skips the download)")
    ap.add_argument("--spider-dir", default=None, help="an already extracted Spider directory (skips download and extraction)")
    ap.add_argument("--bird-dev-zip", default=None, help="BIRD dev.zip already on disk")
    ap.add_argument("--bird-minidev-json", default=None, help="mini_dev_sqlite json already on disk")
    ap.add_argument("--copy-db", action="store_true", help="copy the database files instead of symlinking")
    ap.add_argument("--gold-timeout", type=float, default=30.0, help="seconds per gold query when filtering")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    downloads = out / "downloads"
    databases_dir = out / "databases"
    out.mkdir(parents=True, exist_ok=True)
    stats: dict = {}

    # ---- Spider ----------------------------------------------------------------
    if args.spider_dir:
        spider_root = Path(args.spider_dir).resolve()
    else:
        zip_path = Path(args.spider_zip).resolve() if args.spider_zip else hf_download(SPIDER_REPO, SPIDER_FILE, downloads)
        digest = sha256_of(zip_path)
        if digest != SPIDER_SHA256:
            log(f"WARNING: {zip_path} sha256 {digest} != expected {SPIDER_SHA256}")
        spider_root = extract_zip(zip_path, downloads / "spider")
    train_json = find_file(spider_root, "train_spider.json")
    dev_json = find_file(spider_root, "dev.json")
    spider_dir = train_json.parent
    spider_dbs = find_sqlite_dbs(spider_dir / "database")
    if not spider_dbs:
        raise SystemExit(f"no <db_id>/<db_id>.sqlite under {spider_dir / 'database'}")
    log(f"Spider: {len(spider_dbs)} databases under {spider_dir / 'database'}")
    link_databases(spider_dbs, databases_dir, args.copy_db)
    schemas = SchemaCache(databases_dir)

    with train_json.open() as fh:
        train_items = json.load(fh)
    with dev_json.open() as fh:
        dev_items = json.load(fh)
    train_rows = [make_row(schemas, "spider_train", it["db_id"], it["question"], it["query"]) for it in train_items]
    dev_rows = [
        make_row(schemas, "spider_dev", it["db_id"], it["question"], it["query"], question_id=i)
        for i, it in enumerate(dev_items)
    ]
    log(f"Spider train {len(train_rows)} rows, dev {len(dev_rows)} rows; filtering train gold queries")
    kept, filter_stats = filter_train_rows(train_rows, databases_dir, args.gold_timeout)
    stats["spider_train_filter"] = filter_stats
    rng = random.Random(args.seed)
    rng.shuffle(kept)
    if args.num_train and len(kept) > args.num_train:
        kept = kept[: args.num_train]
    stats["train_rows"] = len(kept)
    write_jsonl(kept, out / "train.jsonl")
    write_jsonl(dev_rows, out / "spider_dev.jsonl")
    stats["spider_dev_rows"] = len(dev_rows)

    # The 2024 release ships the test split and its databases; older archives do not.
    test_json, test_db_dir = spider_dir / "test.json", spider_dir / "test_database"
    if test_json.is_file() and test_db_dir.is_dir():
        # test_database/ is a superset: the train/dev databases again plus the unseen
        # test ones. Link only the new ones; a repeated db_id must be the same file.
        test_dbs = find_sqlite_dbs(test_db_dir)
        new_dbs = {}
        for db_id, path in test_dbs.items():
            if db_id in spider_dbs:
                if path.stat().st_size != spider_dbs[db_id].stat().st_size:
                    raise SystemExit(f"test_database/{db_id} differs from database/{db_id}; refusing to guess which to use")
                continue
            new_dbs[db_id] = path
        link_databases(new_dbs, databases_dir, args.copy_db)
        with test_json.open() as fh:
            test_items = json.load(fh)
        test_rows = [
            make_row(schemas, "spider_test", it["db_id"], it["question"], it["query"], question_id=i)
            for i, it in enumerate(test_items)
        ]
        write_jsonl(test_rows, out / "spider_test.jsonl")
        stats["spider_test_rows"] = len(test_rows)
        log(f"Spider test: {len(new_dbs)} unseen databases ({len(test_dbs) - len(new_dbs)} shared with train/dev), {len(test_rows)} questions")
    else:
        log("Spider test split not in this archive; no spider_test.jsonl")

    # ---- BIRD mini-dev ----------------------------------------------------------
    if args.bird:
        minidev_json = (
            Path(args.bird_minidev_json).resolve()
            if args.bird_minidev_json
            else hf_download(BIRD_MINIDEV_REPO, BIRD_MINIDEV_FILE, downloads)
        )
        if args.bird_dev_zip:
            dev_zip = Path(args.bird_dev_zip).resolve()
        else:
            try:
                dev_zip = url_download(BIRD_DEV_URL, downloads / "bird_dev.zip")
            except (OSError, urllib.error.URLError) as exc:
                raise SystemExit(
                    f"could not download {BIRD_DEV_URL} ({exc}). Download it elsewhere and pass "
                    "--bird-dev-zip /path/dev.zip, or run with --no-bird for Spider only."
                ) from exc
        bird_root = extract_zip(dev_zip, downloads / "bird_dev")
        bird_dbs = find_sqlite_dbs(bird_root)
        if not bird_dbs:
            raise SystemExit(f"no <db_id>/<db_id>.sqlite under {bird_root}")
        # BIRD and Spider both have a database called formula_1, with different
        # contents, so BIRD's live under a bird__ prefix in the shared directory and
        # in the rows' db_id; the original id is kept as bird_db_id.
        bird_dbs = {f"bird__{db_id}": path for db_id, path in bird_dbs.items()}
        log(f"BIRD dev: {len(bird_dbs)} databases (linked as bird__<db_id>)")
        link_databases(bird_dbs, databases_dir, args.copy_db)
        minidev_items = load_json_or_jsonl(minidev_json)
        bird_rows = [
            make_row(
                schemas, "bird_minidev", f"bird__{it['db_id']}", it["question"], it["SQL"], it.get("evidence", ""),
                bird_db_id=it["db_id"], question_id=it.get("question_id"), difficulty=it.get("difficulty"),
            )
            for it in minidev_items
        ]
        write_jsonl(bird_rows, out / "bird_minidev.jsonl")
        stats["bird_minidev_rows"] = len(bird_rows)

    stats["databases"] = len(find_sqlite_dbs(databases_dir)) if args.copy_db else len(list(databases_dir.iterdir()))
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    log(f"done: {json.dumps(stats)}")
    log(f"SQL_DB_ROOT={databases_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
