# Extra JPEG checks

These scripts are not required to reproduce the training tables in `RESULTS.md`.
They are the later checks on the released NPR checkpoint and the trained JA-NPR weights:
four JPEG arms at quality 95 (4:4:4, 4:2:2, 4:4:0, 4:2:0), the activation-space interaction,
and the same measurement on JA-NPR.

The image files and the large per-image dumps are not in this repo. They can be rebuilt
from GANGen-Detection with `npr_chroma_factorial/jpeg_codec.py` and the run scripts.
The CSVs here are the summaries those runs wrote.

`notes/` is the write-up of those runs. The main paper does not depend on them.
