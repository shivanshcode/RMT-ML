#!/usr/bin/env python3
"""Verify a checkout matches the intended post-overlay state.

    cd remote-shree-sk-random-matrix-ml-rmt-stats-esd-fixed
    python check_repo.py

Exit 0 = correct. Non-zero = something is missing, stale, or misplaced.
"""
import hashlib, os, sys

# path, md5, size, which overlay it should come from
EXPECT = [
 ("rmt/per_matrix.py","9dedabfc1da30da9b8754e71261f3fdd",17688,"files3"),
 ("rmt/config.py","d360874bacba3d3554c8161275e5daf7",9279,"files3"),
 ("rmt/cli.py","5c0c33dbbed0c5777fc0662f4be9d525",5618,"fix_v2 (bulk-mode auto patch)"),
 ("rmt/ensembles.py","dcedd27bd620fcaba5d2a7204a73bbf3",2599,"files3"),
 ("tests/test_nulls.py","cc1d8b0a5209a3fa1eeaa0595cf7d86f",8751,"files3"),
 ("tests/test_controls.py","ea7e7a533af13f8f946009a65d2d4267",8558,"files3"),
 ("tests/test_docs_and_references.py","529c6cbdc16e61206d36e674e39b8a97",8170,"files3"),
 ("tests/test_sigma2_determinism.py","8a6851bd23f627e0843d1ac7995568fc",6459,"files3"),
 ("tests/test_sigma2_lmax_calibration.py","6eae6ebac50151a02cb5defaab11613d",3843,"files3"),
 ("rmt_null_calibration.py","9737c27e57cee6ea9ce39d8323b30895",32834,"files3"),
 ("export_weights_npy.py","ba6709f6462cab9f1748dc025a6b249b",5029,"files3"),
 ("rmt/spacing.py","ccac15d936a5f648c0d6b2899b923cf4",56391,"fix_v2"),
 ("rmt/nulls.py","3b7ba7c435ac02d8445d2892fa55e92d",20456,"fix_v2"),
 ("rmt/controls.py","09ba677eb083520f5604b6f463d317fb",15173,"fix_v2"),
 ("rmt/reference.py","c9036a2df91bdeb2378f17e86e6ec909",6852,"fix_v2"),
 ("tests/test_reference_and_unfolding.py","ad580aa39635b18c05265f9218560f16",8938,"fix_v2"),
 ("tests/test_spacing.py","7616c0eaa332a84410284d5c4059321f",14451,"fix_v2"),
 ("tests/test_bulk_selection.py","b0daff145c5dcaa0cef478795100ed90",9761,"fix_v2"),
 ("tests/test_crosscheck_calibration.py","3af0e5616d30af837e2a107d547341e9",3576,"fix_v2"),
 ("benchmark3.py","4e5616325feb0de487500675ece48c52",8760,"fix_v2"),
 ("run_rmt_calibrated_v4.slurm","ce9f926e2d82a91743ca6029aef8ffd5",13714,"fix_v2 (supersedes run_rmt_calibrated.slurm and _v2.slurm)"),
 ("run_stage2_bands.slurm","eccf3d2682a677127d76b18b18a17d22",4777,"fix_v2"),
]
# Files that must exist but whose content I don't pin (docs / artefacts)
SHOULD_EXIST = ["design.md","resolved-issues.md","outstanding-issues.md",
                "FILE_TREE.md","FILES3_PLACEMENT.md","README_APPLY.md",
                "BENCHMARK_TABLE.md","benchmark_results_3way.csv",
                "benchmark_results.csv","RESOLUTION_REPORT.md",
                "tests/conftest.py","rmt/linalg.py","rmt/mp.py",
                "rmt/porter_thomas.py","benchmark.py"]
# Things that indicate a MISPLACED file
MISPLACED = ["nulls.py","controls.py","spacing.py","reference.py",
             "test_nulls.py","test_spacing.py","test_reference_and_unfolding.py",
             "test_controls.py","test_bulk_selection.py"]

def md5(p):
    h=hashlib.md5()
    with open(p,"rb") as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

miss,stale,ok=[],[],0
for path,want,size,src in EXPECT:
    if not os.path.isfile(path): miss.append((path,src)); continue
    got=md5(path)
    if got!=want: stale.append((path,src,os.path.getsize(path),size))
    else: ok+=1

print(f"content-pinned files: {ok}/{len(EXPECT)} correct\n")
if miss:
    print("MISSING:")
    for p,s in miss: print(f"  {p:44s} (copy from {s})")
    print()
if stale:
    print("WRONG CONTENT (stale or hand-edited):")
    for p,s,g,w in stale: print(f"  {p:44s} {g:>7d} bytes, expected {w:>7d}  (recopy from {s})")
    print()

ne=[p for p in SHOULD_EXIST if not os.path.exists(p)]
if ne:
    print("EXPECTED BUT ABSENT (docs/artefacts/untouched repo files):")
    for p in ne: print(f"  {p}")
    print()

mp=[p for p in MISPLACED if os.path.isfile(p)]
if mp:
    print("MISPLACED AT REPO ROOT (should be in rmt/ or tests/):")
    for p in mp: print(f"  ./{p}")
    print()

def _is_demo_band(p):
    """True only if the file really holds the 400x400 demo bands.

    This used to fire on the mere existence of cal/null_bands.csv, so it warned
    on every run once the real four-shape bands were built. Locate the 'shape'
    column by name (it is column 1 in null_bands.csv but column 2 in
    zscores.csv) and judge on its contents instead.
    """
    try:
        with open(p) as fh:
            header = fh.readline().rstrip("\n").split(",")
            if "shape" not in header:
                return False
            i = header.index("shape")
            shapes = {ln.rstrip("\n").split(",")[i]
                      for ln in fh if ln.strip() and len(ln.split(",")) > i}
    except OSError:
        return False
    return bool(shapes) and shapes <= {"400x400", "512x512"}

bad_cal=[p for p in ("cal/null_bands.csv","cal/zscores.csv")
         if os.path.exists(p) and _is_demo_band(p)]
if bad_cal:
    print("WARNING - stale 400x400 demo bands in the LIVE cal/ directory:")
    for p in bad_cal: print(f"  {p}   -> move to cal_results/ (outstanding-issues O13)")
    print()

fail=bool(miss or stale or mp)
print("RESULT:", "FAIL - see above" if fail else "PASS - tree matches intended state")
if not fail:
    print("\nNow run:")
    print("  python -m rmt --selftest                    # expect 11/11 PASS")
    print('  python -m pytest tests/ -q -m "not torch"   # expect 263 passed, 31 deselected')
sys.exit(1 if fail else 0)
