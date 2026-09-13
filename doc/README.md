# PhD Documentation and Publications

The `doc/` directory contains thesis sources, conference publications, figures,
and periodic progress reports. It is a documentation and publication layer,
not the execution or data-storage layer for the OGS pipeline.

## Build the UniTS thesis

From this directory, build the UniTS thesis with:

```bash
make units
```

The target runs `pdflatex`, `bibtex`, and two additional `pdflatex` passes.
The source is [`UNITS/UNITS.tex`](UNITS/UNITS.tex), and the generated PDF is
written to [`UNITS/UNITS.pdf`](UNITS/UNITS.pdf).

## Purpose and evidence boundary

Use this directory for human-authored research documentation and publication
sources. Verify scientific claims against [`../OGS/`](../OGS/) and experiment
records before using narrative reports as technical evidence.

## Directory guide

- [`UNITS/`](UNITS/) contains the UniTS thesis source, bibliography, chapters,
  settings, supplementary material, and build output.
- [`MHPC/`](MHPC/) contains the MHPC thesis source, bibliography, chapters, and
  build output.
- [`MICAI/`](MICAI/) contains the MICAI paper source and PDF.
- [`EduAI/`](EduAI/) contains education-related documentation sources.
- [`imgs/`](imgs/) contains shared figures and generated catalog visual assets.
- [`reports/`](reports/) contains Markdown and historical PDF progress reports;
  see its [`README.md`](reports/README.md) for the index.
- [`Makefile`](Makefile) defines the supported documentation build targets.

Keep credentials, restricted raw data, generated waveforms, and generated
catalog artifacts outside this directory. Generated PDFs may not be
reproducible from Markdown files alone.

## Inputs and outputs

- **Inputs:** reviewed research notes, report material, figures, and
  publication sources.
- **Outputs:** LaTeX/PDF documents and Markdown research records. These
  outputs are documentation artifacts, not validated scientific results by
  themselves.
