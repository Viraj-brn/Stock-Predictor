"""Check accuracy of the returns-based model predictions."""
import json

data = json.load(open("data/predictions.json"))

print("=" * 65)
print("RETURNS-BASED MODEL - TEST SET ACCURACY")
print("=" * 65)

# Check last 5 seeded entries
for entry in data[-5:]:
    preds = entry["predictions"]
    acts = entry["actuals"]
    errors = []
    for t in sorted(preds.keys()):
        if preds[t] is not None and acts[t] is not None and acts[t] != 0:
            err = abs(preds[t] - acts[t]) / acts[t] * 100
            errors.append(err)
    avg_err = sum(errors) / len(errors) if errors else 0
    print(f"  {entry['date']}: avg error = {avg_err:.1f}%")

# Overall stats
print()
print("=" * 65)
print("OVERALL STATS (all 313 test entries)")
print("=" * 65)

all_errors = {t: [] for t in ["AAPL", "JNJ", "JPM", "MSFT", "PEP"]}
for entry in data:
    preds = entry["predictions"]
    acts = entry["actuals"]
    for t in all_errors:
        if preds.get(t) and acts.get(t) and acts[t] != 0:
            err = abs(preds[t] - acts[t]) / acts[t] * 100
            all_errors[t].append(err)

print(f"{'Ticker':8s} | {'Avg Err%':>8s} | {'Med Err%':>8s} | {'Max Err%':>8s}")
print("-" * 48)
import statistics
total_errors = []
for t in sorted(all_errors.keys()):
    errs = all_errors[t]
    avg = sum(errs) / len(errs) if errs else 0
    med = statistics.median(errs) if errs else 0
    mx = max(errs) if errs else 0
    print(f"{t:8s} | {avg:>7.1f}% | {med:>7.1f}% | {mx:>7.1f}%")
    total_errors.extend(errs)

overall = sum(total_errors) / len(total_errors)
print(f"\n  Overall average error: {overall:.1f}%")

# Show a sample entry detail
print()
print("=" * 65)
print("SAMPLE ENTRY (last one)")
print("=" * 65)
last = data[-1]
preds = last["predictions"]
acts = last["actuals"]
print(f"Date: {last['date']}")
print(f"{'Ticker':8s} | {'Predicted':>10s} | {'Actual':>10s} | {'Error%':>8s}")
print("-" * 45)
for t in sorted(preds.keys()):
    if preds[t] and acts[t] and acts[t] != 0:
        err = abs(preds[t] - acts[t]) / acts[t] * 100
        print(f"{t:8s} | ${preds[t]:>9.2f} | ${acts[t]:>9.2f} | {err:>6.1f}%")
