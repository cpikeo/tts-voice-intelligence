#!/usr/bin/env bash
# Requires local, licensed Piper assets + optional installed ASR and eSpeak NG.
# Run from repository root: bash validation/reproduce.sh /path/to/model.onnx new-validation
# Output must be a new directory. ASR base may download if not cached.
set -euo pipefail
MODEL=${1:?provide local Piper model}
DEST=${2:?provide new output directory}
PRIVATE_DIR=${3:-private}
if [ -e "$DEST" ]; then echo 'Refusing to mix prior evidence' >&2; exit 4; fi
mkdir -p "$DEST" "$PRIVATE_DIR"
chmod 700 "$PRIVATE_DIR"

python -m pytest -q > "$DEST/pytest-no-piper.log" 2>&1
export TTS_PIPER_MODEL="$MODEL"
python -m pytest -q > "$DEST/pytest-full.log" 2>&1

python scripts/tts.py check --model "$MODEL" > "$DEST/piper-check.json"
python scripts/tts.py check --engine xtts > "$DEST/xtts-blocked.json" || true

# Priority 2: Chinese accuracy, mixed EN, digits/dates, proper nouns, polyphones, narrative prosody, long text
python tests/run_eval.py --model "$MODEL" --out "$DEST/piper-repeat" \
  --ids zh_plain,digits_date,mixed_en,proper,polyphone,zh_emotional_turn,zh_long \
  --repeat 3 --asr base --lexicon assets/lexicon.example.tsv > "$DEST/piper-eval.log" 2>&1

# Multilingual eSpeak evaluation (en-US with ASR, plus en-GB, de, ja)
: > "$DEST/espeak-eval.log"
python tests/run_eval.py --engine espeak --language en-US --out "$DEST/espeak-en-us" \
  --repeat 3 --asr base >> "$DEST/espeak-eval.log" 2>&1
for locale in en-GB de ja; do
  python tests/run_eval.py --engine espeak --language "$locale" --out "$DEST/espeak-$locale" \
    --repeat 3 >> "$DEST/espeak-eval.log" 2>&1
done

# Voice direction execution & QA verification
python scripts/tts.py synth --engine espeak --language en-US \
  --text 'This discovery changes everything.' --direction assets/direction.example.json \
  --out "$DEST/direction.wav" --report "$DEST/direction.json" > "$DEST/direction-run.log" 2>&1
python scripts/qa.py "$DEST/direction.wav" --synthesis-report "$DEST/direction.json" \
  --report "$DEST/direction-qa.json" >> "$DEST/direction-run.log" 2>&1

# Long-text production continuity synthesis + signal & ASR QA
python - "$MODEL" "$DEST" <<'PY' > "$DEST/long-run.log" 2>&1
import json, resource, sys, time
from pathlib import Path
sys.path.insert(0, 'scripts')
import tts, qa
model, dest = sys.argv[1:]
dest = Path(dest)
cases = [json.loads(l) for l in Path('tests/cases.jsonl').read_text().splitlines() if l.strip()]
text = next(c['text'] for c in cases if c['id'] == 'zh_long_production')
source = dest / 'long-input.txt'
source.write_text(text, encoding='utf-8')
start = time.perf_counter()
code = tts.main(['synth', '--model', model, '--input', str(source),
                 '--out', str(dest / 'long.wav'), '--report', str(dest / 'long.json')])
measurement = {'synth_exit': code, 'wall_seconds': round(time.perf_counter() - start, 4),
               'peak_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024), 3)}
if code == 0:
    measurement['qa_exit'] = qa.main([str(dest / 'long.wav'), '--synthesis-report', str(dest / 'long.json'),
                                      '--asr', 'base', '--report', str(dest / 'long-qa.json')])
(dest / 'long-resources.json').write_text(json.dumps(measurement, indent=2), encoding='utf-8')
if code: raise SystemExit(code)
PY

# Priority 1: Isolated blind-test package (listen/ vs ratings/ vs private/blind_map.json)
rm -f "$PRIVATE_DIR/listening-blind-map.json"
python scripts/blindtest.py prepare \
  --audio \
    "$DEST/piper-repeat/zh_plain.norm.r0.wav" \
    "$DEST/piper-repeat/digits_date.norm.r0.wav" \
    "$DEST/piper-repeat/digits_date.raw.r0.wav" \
    "$DEST/piper-repeat/mixed_en.norm.r0.wav" \
    "$DEST/piper-repeat/mixed_en.raw.r0.wav" \
    "$DEST/piper-repeat/proper.norm.r0.wav" \
    "$DEST/piper-repeat/polyphone.norm.r0.wav" \
    "$DEST/piper-repeat/zh_emotional_turn.norm.r0.wav" \
    "$DEST/piper-repeat/zh_long.norm.r0.wav" \
  --out "$DEST/listening-unrated" \
  --ratings-dir "$DEST/listening-unrated/ratings" \
  --map-file "$PRIVATE_DIR/listening-blind-map.json" \
  --isolate --seed 17 > "$DEST/listening-prepare.log"
