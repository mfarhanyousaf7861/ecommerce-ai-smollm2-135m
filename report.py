"""Summarise only existing measurements; never fill missing model results."""
import json
from experiment import RESULTS

lines = ['# Experiment results', '',
         'Results are from a small synthetic English-only dataset. No production claim is justified.', '',
         '| Method | Test examples | Accuracy | Macro F1 | Median request ms |', '|---|---:|---:|---:|---:|']
for name in ['baseline', 'base', 'tuned']:
    p = RESULTS / f'{name}.json'
    if p.exists():
        r = json.loads(p.read_text());
        lines.append(
            f"| {name} | {r['n']} | {r['accuracy']:.1%} | {r['macro_f1']:.3f} | {r['latency_ms_median']:.2f} |")
    else:
        lines.append(f'| {name} | NOT RUN | — | — | — |')
p = RESULTS / 'retrieval.json'
if p.exists():
    r = json.loads(p.read_text());
    lines += ['',
              f"Retrieval: top-1 hit rate {r['hit_at_1']:.1%}; top-3 hit rate {r['hit_at_3']:.1%}, on {r['answerable_n']} answerable questions."]
lines += ['', 'RAG generation: ' + ('answers exist; inspect human-review fields before making quality claims.' if (
            RESULTS / 'rag_answers.json').exists() else 'NOT RUN.'),
          '',
          'Latency figures are comparable only when rerun on the same machine with the same settings. Twelve-candidate SLM scoring is not the same workload as a single classifier prediction.']
(RESULTS / 'SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
