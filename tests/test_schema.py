"""Schema handling for the full (gated) corpus.

The full dataset differs from the earlier sample in two ways that silently break
naive code, so both are pinned here:

  * it has an `annotationQuality` column - camelCase, where the sample had no
    tier field at all;
  * `NoiseSubCategoryTimeStamp.start/end` are typed **string**, not float32.

The repo is gated, so the exact quality values cannot be read from here. The
mapping is therefore generous and anything unmatched is *reported*, never
guessed - these tests cover both behaviours.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data.prepare import (UNMAPPED_QUALITY, _lookup, _resolve_tier, _to_float,
                              build_record, quality_to_tier)

fails = []


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def ev(start, end, cat="animal_sound", tag=""):
    return {"category": cat, "tag": tag, "start": start, "end": end}


print("-- string timestamp parsing --")
ok(_to_float("1.234") == 1.234, "plain numeric string")
ok(_to_float(1.234) == 1.234, "float passthrough")
ok(_to_float("1,234") == 1.234, "comma decimal separator")
ok(_to_float("  2.5 ") == 2.5, "surrounding whitespace")
ok(_to_float("") is None, "empty string -> None")
ok(_to_float("NA") is None and _to_float("nan") is None, "NA/nan -> None")
ok(_to_float(None) is None, "None -> None")
ok(_to_float("abc") is None, "garbage -> None (must not raise)")

print("\n-- case/separator-insensitive column lookup --")
ok(_lookup({"annotationQuality": "gold"}, ["annotationquality"]) == (True, "gold"),
   "camelCase annotationQuality found")
ok(_lookup({"annotation_quality": "gold"}, ["annotationquality"]) == (True, "gold"),
   "snake_case annotation_quality found")
ok(_lookup({"AnnotationQuality": "gold"}, ["annotationquality"]) == (True, "gold"),
   "PascalCase found")
ok(_lookup({"other": 1}, ["annotationquality"])[0] is False, "absent column reported")

print("\n-- quality value mapping --")
for raw, want in [("gold", "gold"), ("Gold", "gold"), ("GOLD", "gold"),
                  ("silver", "silver"), ("Silver ", "silver"),
                  ("bronze", "bronze"), ("verified", "gold"), ("unverified", "silver"),
                  ("tier1", "gold"), ("tier2", "silver"), ("tier3", "bronze"),
                  ("high", "gold"), ("low", "bronze"),
                  # The real corpus's actual annotationQuality strings - "verified"
                  # is a substring of "unverified", so a naive first-match substring
                  # scan collapses every unverified_timestamps clip into gold. This
                  # is the regression that shipped and needs to stay caught.
                  ("verified_timestamps", "gold"), ("unverified_timestamps", "silver"),
                  ("no_timestamps", "bronze")]:
    ok(quality_to_tier(raw) == want, "%-12r -> %s" % (raw, want))
ok(quality_to_tier("wibble") is None, "unknown value -> None (reported, not guessed)")

print("\n-- tier resolution --")
row_gold = {"annotationQuality": "gold", "NoiseSubCategoryTimeStamp": [ev("0.1", "0.9")]}
ok(_resolve_tier(row_gold, True, set(), "silver") == "gold", "gold with timestamps")

row_sil = {"annotationQuality": "silver", "NoiseSubCategoryTimeStamp": [ev("0.1", "0.9")]}
ok(_resolve_tier(row_sil, True, set(), "silver") == "silver", "silver with timestamps")

# A gold label with no timestamps cannot supply frame supervision.
ok(_resolve_tier({"annotationQuality": "gold"}, False, set(), "silver") == "bronze",
   "gold WITHOUT timestamps is demoted to bronze")

ok(_resolve_tier({"annotationQuality": "bronze"}, False, set(), "silver") == "bronze",
   "bronze stays bronze")

UNMAPPED_QUALITY.clear()
t = _resolve_tier({"annotationQuality": "wibble",
                   "NoiseSubCategoryTimeStamp": [ev("0", "1")]}, True, set(), "silver")
ok(t == "silver", "unmapped quality falls back to the configured default")
ok(dict(UNMAPPED_QUALITY) == {"wibble": 1}, "unmapped quality is recorded for reporting")

print("\n-- end-to-end record from a full-corpus shaped row --")
row = {
    "audio": {"path": "x/CLIP123.wav", "bytes": b""},
    "state": "Bihar", "district": "Patna", "language": "Hindi", "duration": 3.0,
    "annotationQuality": "gold",
    "NoiseCategory": ["animal_sound", "vehicle_traffic"],
    "NoiseSubCategoryTimeStamp": [
        ev("0.500", "1.250", "animal_sound", "<dog barking>"),
        ev("1.900", "2.400", "vehicle_traffic", "<horn>"),
        ev("bad", "2.0", "animal_sound"),          # unparsable -> skipped
        ev("2.5", "2.5", "animal_sound"),          # zero length -> skipped
        ev("2.8", "9.9", "animal_sound"),          # end beyond clip -> clipped
    ],
}
unknown = __import__("collections").Counter()
rec = build_record(row, "CLIP123", 3.0, set(), "silver", True, unknown)
print("  ", rec)
ok(rec["tier"] == "gold", "tier from annotationQuality")
ok(len(rec["events"]) == 3, "2 bad events dropped, 3 kept (got %d)" % len(rec["events"]))
ok(rec["events"][0]["start"] == 0.5 and rec["events"][0]["end"] == 1.25,
   "string timestamps parsed to floats")
ok(rec["events"][1]["cls"] == "vehicle_horn", "<horn> -> vehicle_horn subtype")
ok(rec["events"][2]["end"] == 3.0, "event end clipped to clip duration")
ok(unknown.get("<unparsable timestamp>") == 1, "unparsable timestamp counted")
ok(set(rec["clip_labels"]) == {"animal_sound", "vehicle_traffic"}, "clip labels")

print("\n%d failures" % len(fails))
for f in fails:
    print("  -", f)
sys.exit(1 if fails else 0)
