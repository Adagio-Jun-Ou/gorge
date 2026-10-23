"""gorge command-line interface."""

from __future__ import annotations

import argparse
import os
import sys
import time

from . import __version__
from .phasing import Phaser, dim_for_capacity, load_weights, parse_capacity
from .query import probe as run_probe
from .report import print_ingest_summary, print_probe
from .scan import iter_bundles

BANNER = "GORGE: eat a drive, phase it into weights"


def cmd_ingest(args) -> int:
    if args.drop_cache and args.no_bake:
        print("error: --drop-cache and --no-bake together would build nothing",
              file=sys.stderr)
        return 2
    params = parse_capacity(args.capacity)
    dim = dim_for_capacity(params)
    ph = Phaser(
        dim=dim,
        keep_cache=not args.drop_cache,
        bake=not args.no_bake,
        dedup=not args.no_dedup,
        salt=args.salt,
        fold_batch=args.fold_batch,
        chunk=args.chunk,
    )
    print(f"ingesting {args.path}")
    print(f"  dim={dim}  params~{params}  chunk={args.chunk}B  "
          f"cache={'on' if not args.drop_cache else 'off'}  "
          f"bake={'on' if not args.no_bake else 'off'}")
    t0 = time.time()
    seen = 0
    try:
        for b in iter_bundles(args.path, chunk=args.chunk, max_files=args.max_files):
            ph.add_bundle(b.path, b.offset, b.data, b.type_id, b.file_size)
            seen += 1
            if seen % args.progress == 0:
                dt = max(1e-9, time.time() - t0)
                print(f"  {ph.n_files} files  {ph.bytes_seen / 1e6:.1f} MB  "
                      f"{ph.engrams} engrams  {ph.bytes_seen / 1e6 / dt:.1f} MB/s",
                      flush=True)
    except (FileNotFoundError, PermissionError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    pw = ph.finalize()
    pw.save(args.out)
    print()
    print_ingest_summary(pw, elapsed=time.time() - t0, out_path=args.out)
    return 0


def cmd_probe(args) -> int:
    pw = load_weights(args.model)
    if args.text is not None:
        sample = args.text.encode("utf-8")
        desc = f"text: {args.text!r}"
    elif args.sample:
        with open(args.sample, "rb") as f:
            sample = f.read(args.max_bytes)
        desc = f"file: {args.sample} ({len(sample)} bytes)"
    else:
        print("provide --text or --sample", file=sys.stderr)
        return 2
    res = run_probe(pw, sample, top=args.top, beta=args.beta,
                    resonate_steps=args.steps)
    print_probe(pw, res, desc)
    return 0


def cmd_stats(args) -> int:
    pw = load_weights(args.model)
    print_ingest_summary(pw, out_path=args.model)
    return 0


def cmd_demo(args) -> int:
    from . import demo
    return demo.run(capacity=args.capacity, chunk=args.chunk)


def cmd_ui(args) -> int:
    """Serve the browser demo (web/) on localhost and open it."""
    import functools
    import http.server
    import threading
    import webbrowser

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
    webdir = os.path.join(root, "web")
    if not os.path.isdir(webdir):
        print("web/ not found -- the UI ships with the repo; open web/index.html directly")
        return 2
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=webdir)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"gorge UI live at {url}  (Ctrl+C to stop)")
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


def cmd_memorize(args) -> int:
    from .memorize import memorize_file
    out = args.out or (args.file + ".gwpt")
    res = memorize_file(args.file, out, steps=args.steps, hidden=args.hidden,
                        depth=args.depth, block=args.block, lr=args.lr)
    print(f"memorized {res['n_bytes']} bytes into {out}")
    print(f"  positions : {res['positions']}")
    print(f"  final mse : {res['mse']:.6f}")
    print(f"  byte-exact recall at memorize time: {res['exact_pct'] * 100:.1f}%")
    print(f"restore with: gorge recall {out}")
    return 0


def cmd_recall(args) -> int:
    from .memorize import recall_file
    out = args.out or (args.weights + ".out")
    res = recall_file(args.weights, out)
    print(f"recalled {res['n_bytes']} bytes -> {out}")
    print(f"  sha match with original      : {res['sha_match']}")
    print(f"  byte-exact at memorize time  : {res['exact_pct_stored'] * 100:.1f}%")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gorge", description=BANNER)
    p.add_argument("--version", action="version", version=f"gorge {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="eat a drive/directory/file into weights")
    ing.add_argument("path", help="drive, directory, or file to eat (read-only)")
    ing.add_argument("-o", "--out", default="gorge.gw", help="output model path")
    ing.add_argument("--capacity", default="25M",
                    help="model size in parameters, e.g. 25M (sets dim = sqrt)")
    ing.add_argument("--chunk", type=int, default=65536,
                    help="bytes per engram bundle (rule of thumb: >= 8*dim)")
    ing.add_argument("--max-files", type=int, default=None)
    ing.add_argument("--drop-cache", action="store_true",
                     help="weights-only artifact (no exact phase cache)")
    ing.add_argument("--no-bake", action="store_true",
                     help="skip building W (fast ingest, cache-only)")
    ing.add_argument("--no-dedup", action="store_true")
    ing.add_argument("--salt", type=int, default=0)
    ing.add_argument("--fold-batch", type=int, default=512)
    ing.add_argument("--progress", type=int, default=50000)
    ing.set_defaults(func=cmd_ingest)

    pr = sub.add_parser("probe", help="ask the model what a sample is")
    pr.add_argument("model", help="path to a .gw model")
    pr.add_argument("--text", default=None, help="query by text")
    pr.add_argument("--sample", default=None, help="query by file sample")
    pr.add_argument("--top", type=int, default=5)
    pr.add_argument("--beta", type=float, default=20.0)
    pr.add_argument("--steps", type=int, default=2, help="hopfield resonance steps")
    pr.add_argument("--max-bytes", type=int, default=1_000_000)
    pr.set_defaults(func=cmd_probe)

    st = sub.add_parser("stats", help="model size, load factor, type breakdown")
    st.add_argument("model")
    st.set_defaults(func=cmd_stats)

    dm = sub.add_parser("demo", help="build a synthetic drive, eat it, probe it")
    dm.add_argument("--capacity", default="4M")
    dm.add_argument("--chunk", type=int, default=2048)
    dm.set_defaults(func=cmd_demo)

    ui = sub.add_parser("ui", help="serve the animated browser demo and open it")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--no-open", action="store_true", help="do not launch a browser")
    ui.set_defaults(func=cmd_ui)

    me = sub.add_parser("memorize", help="[torch] store a file literally in weights")
    me.add_argument("file")
    me.add_argument("-o", "--out", default=None)
    me.add_argument("--steps", type=int, default=1500)
    me.add_argument("--hidden", type=int, default=256)
    me.add_argument("--depth", type=int, default=4)
    me.add_argument("--block", type=int, default=64, help="bytes per position")
    me.add_argument("--lr", type=float, default=1e-3)
    me.set_defaults(func=cmd_memorize)

    rc = sub.add_parser("recall", help="[torch] reconstruct bytes from weights")
    rc.add_argument("weights", help=".gwpt weights from gorge memorize")
    rc.add_argument("-o", "--out", default=None)
    rc.set_defaults(func=cmd_recall)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())