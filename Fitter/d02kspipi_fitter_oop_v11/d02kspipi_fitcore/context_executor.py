from __future__ import annotations

def run_code_block(ctx: dict, code: str, label: str) -> None:
    """Execute one preserved fitter block inside the shared fitter context."""
    compiled = compile(code, label, "exec")
    exec(compiled, ctx)
