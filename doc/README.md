# Documentation build

From this directory, build the UniTS thesis with:

```bash
make units
```

The target runs `pdflatex`, `bibtex`, and two additional `pdflatex` passes.
The thesis uses `acro` for its acronym declarations and `\printacronyms` for
the linked front-matter list; do not add a second acronym declaration system.