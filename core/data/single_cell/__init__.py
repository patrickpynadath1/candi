"""Single-cell benchmark provenance and prepared count matrices."""

DATASETS = {
    "pbmc3k": {"genes": 8573, "cells": 2638, "cell_types": 8, "label_key": "cell_type"},
    "dentategyrus": {
        "genes": 17002,
        "cells": 18213,
        "cell_types": 14,
        "label_key": "clusters",
    },
    "pbmc_covid": {
        "genes": 2000,
        "cells": 422220,
        "cell_types": 29,
        "label_key": "cell_type",
    },
}
SOURCE_URL = "https://zenodo.org/records/15031364"
ARCHIVE_URL = SOURCE_URL + "/files/datasets.zip?download=1"
ARCHIVE_MD5 = "c0a6582b05e6ce758811e58d703c0aff"
