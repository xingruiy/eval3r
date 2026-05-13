"""e3r benchmark — dataset-specific benchmark subcommands."""

from __future__ import annotations

import typer

from eval3r.cli.benchmark import dtu
from eval3r.cli.benchmark import eth3d
from eval3r.cli.benchmark import generic
from eval3r.cli.benchmark import replica
from eval3r.cli.benchmark import scannet
from eval3r.cli.benchmark import tanks_temples
from eval3r.cli.benchmark import tum_rgbd

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.command("scannet")(scannet.command)
app.command("tanks-temples")(tanks_temples.command)
app.command("tum-rgbd")(tum_rgbd.command)
app.command("replica")(replica.command)
app.command("dtu")(dtu.command)
app.command("eth3d")(eth3d.command)
app.command("generic")(generic.command)
