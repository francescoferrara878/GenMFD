#!/bin/sh
# Downloads the public resources used in the evaluation into ./data (run from the repository root).
set -e
mkdir -p data && cd data
git clone --depth 1 https://github.com/medianeuroscience/emfd.git            # MFD, MFD 2.0, eMFD (dictionaries/)
git clone --depth 1 https://github.com/oaraque/moral-foundations.git        # MoralStrength (moralstrength/annotations/v1.1/)
git clone --depth 1 https://github.com/franciellevargas/MFTCXplain.git      # MFTCXplain (dataset/mftcxplain_en.json)
echo "Still to be placed by hand in data/:"
echo "  final_mfrc_data.csv   (MFRC, with columns text, annotator, annotation, confidence)"
echo "  MFTC_V4_text.json     (MFTC with tweet texts)"
