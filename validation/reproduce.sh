#!/usr/bin/env bash
# Requires local, licensed Piper assets + optional installed ASR and eSpeak NG.
# Run from repository root: bash validation/reproduce.sh /path/to/model.onnx new-validation
# Output must be a new directory. ASR base may download if not cached.
set -euo pipefail
MODEL=${1:?provide local Piper model}
DEST=${2:?provide new output directory}
if [ -e "$DEST" ]; then echo 'Refusing to mix prior evidence' >&2; exit 4; fi
mkdir -p "$DEST"
export TTS_PIPER_MODEL="$MODEL"
python -m pytest -q > "$DEST/pytest-full.log" 2>&1
python tests/run_eval.py --model "$MODEL" --out "$DEST/piper-repeat" \
  --ids digits_date,mixed_en,zh_emotional_turn --repeat 3 --asr base \
  --lexicon assets/lexicon.example.tsv
for locale in en-US en-GB de ja; do
  python tests/run_eval.py --engine espeak --language "$locale" --out "$DEST/espeak-$locale" --repeat 3
done
python - "$MODEL" "$DEST" <<'PY'
import json, resource, sys, time
from pathlib import Path
sys.path.insert(0, 'scripts')
import tts, qa
model, dest = sys.argv[1:]
dest = Path(dest)
cases = [json.loads(l) for l in Path('tests/cases.jsonl').read_text().splitlines()]
text = next(c['text'] for c in cases if c['id'] == 'zh_long_production')
source = dest / 'long-input.txt'
source.write_text(text)
start = time.perf_counter()
code = tts.main(['synth', '--model', model, '--input', str(source),
                 '--out', str(dest / 'long.wav'), '--report', str(dest / 'long.json')])
measurement = {'synth_exit': code, 'wall_seconds': time.perf_counter() - start,
               'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024)}
if code == 0:
    measurement['qa_exit'] = qa.main([str(dest / 'long.wav'), '--synthesis-report', str(dest / 'long.json'),
                                      '--report', str(dest / 'long-qa.json')])
(dest / 'long-resources.json').write_text(json.dumps(measurement, indent=2))
if code: raise SystemExit(code)
PY
