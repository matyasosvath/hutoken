#!/usr/bin/bash

set -euo pipefail

if ! command -v foma &> /dev/null; then
    echo "Error: Required dependency foma is not installed." >&2
    exit 1
fi

if ! command -v hfst-xfst &> /dev/null; then
    echo "Error: Required dependency hfst is not installed." >&2
    exit 1
fi

echo "1: Building initial lexicon files."
(
    cd emmorph/mak
    bash mkX.sh xlx
    bash mkxlxrmseg.sh
    bash xlx2lglexc.sh huX
)

echo "2: Compiling FSTs with foma and hfst."
(
    cd emmorph/lexc

    echo "Creating case-sensitive analyzer..."
    foma -e "read lexc huXlg.lexc" \
        -e "eliminate flag St" \
        -e "minimize net" \
        -e "save stack hu_case_sensitive.foma.bin" \
        -e "exit"

    echo "Creating case-normalizing transducer..."
    hfst-xfst -F casenormhuX.xfs
    hfst-invert casenormhu.hfst -o casenormhu_inv.hfst
    hfst-fst2fst -F -b -i casenormhu_inv.hfst -o casenormhu_inv.foma.bin

    echo "Composing analyzer with normalizer..."
    foma -e "load stack casenormhu_inv.foma.bin" \
        -e "load stack hu_case_sensitive.foma.bin" \
        -e "compose" \
        -e "save stack hu.foma.bin" \
        -e "exit"
)

echo "3: Installing final artifact."
(
    cd ../
    mkdir -p bin
    mv ./scripts/emmorph/lexc/hu.foma.bin bin/
)

echo "Successfully created bin/hu.foma.bin."
