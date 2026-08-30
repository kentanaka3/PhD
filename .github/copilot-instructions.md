# GitHub Copilot repository instructions

Follow the repository-wide contract in [`../AGENTS.md`](../AGENTS.md) and use
[`../LLM/README.md`](../LLM/README.md) as the LLM-workspace guide.

Before changing files:

1. Run `git status --short` and preserve all existing user work.
2. Map the repository with `bash LLM/scripts/handler.sh navigate --root . --scripts`.
3. Inspect the relevant Makefile, Bash, Python, configuration, and tests.
4. State scope, assumptions, risks, and focused validation.

After changing files, run the smallest meaningful checks. For documentation
or navigation changes, use:

```bash
bash -n LLM/scripts/handler.sh
bash LLM/scripts/handler.sh init --dry-run
bash LLM/scripts/handler.sh navigate --root .
bash LLM/scripts/handler.sh validate --root .
git diff --check
```

Do not install dependencies, download data, run inference, submit jobs, or
change scientific pipeline semantics without explicit approval. Preserve the
boundary between observations, predictions, labels, hypotheses, and validated
scientific results.
