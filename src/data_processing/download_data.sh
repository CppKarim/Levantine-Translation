#!/bin/bash
# Change to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { echo "Failed to change directory to $SCRIPT_DIR"; exit 1; }
cd ../
mkdir "data/dataset/UFAL Parallel Corpus of North Levantine 1.0" || exit 1

cd "data/dataset/UFAL Parallel Corpus of North Levantine 1.0"
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5033{/README.md,/ufal-nla-v1.apc,/ufal-nla-v1.arb,/ufal-nla-v1.deu,/ufal-nla-v1.ell,/ufal-nla-v1.eng,/ufal-nla-v1.fra,/ufal-nla-v1.spa,/ufal-nla-v1.arb-eng.ids,/ufal-nla-v1.deu-eng.ids,/ufal-nla-v1.ell-eng.ids,/ufal-nla-v1.eng-fra.ids,/ufal-nla-v1.eng-spa.ids}