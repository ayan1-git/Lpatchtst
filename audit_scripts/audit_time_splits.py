#!/usr/bin/env python3
import ast
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional


def find_repo_root(start='.'):
    candidates = [start, os.getcwd(), os.path.dirname(os.getcwd())]
    for base in candidates:
        if os.path.exists(os.path.join(base, 'train.py')) and os.path.exists(os.path.join(base, 'data_loader.py')):
            return os.path.abspath(base)
    for base, dirs, files in os.walk(start):
        if 'train.py' in files and 'data_loader.py' in files:
            return os.path.abspath(base)
    raise FileNotFoundError('Could not locate repo root containing train.py and data_loader.py')


def read_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def get_lines(text):
    return text.splitlines()


def grep_patterns(lines: List[str], patterns: List[str]) -> List[Dict[str, Any]]:
    hits = []
    rx = re.compile('|'.join(patterns), re.I)
    for i, line in enumerate(lines, start=1):
        if rx.search(line):
            hits.append({'line_no': i, 'line': line.rstrip()})
    return hits


def ast_calls_info(text: str) -> List[Dict[str, Any]]:
    try:
        tree = ast.parse(text)
    except Exception as e:
        return [{'parse_error': repr(e)}]
    out = []
    class V(ast.NodeVisitor):
        def visit_Call(self, node):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                parts = []
                cur = node.func
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                name = '.'.join(reversed(parts))
            kws = {}
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                try:
                    kws[kw.arg] = ast.unparse(kw.value)
                except Exception:
                    kws[kw.arg] = '<unparse_failed>'
            out.append({'func': name, 'line_no': getattr(node, 'lineno', None), 'keywords': kws})
            self.generic_visit(node)
    V().visit(tree)
    return out


def extract_functions(text: str) -> List[Dict[str, Any]]:
    try:
        tree = ast.parse(text)
    except Exception:
        return []
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append({
                'name': node.name,
                'line_no': node.lineno,
                'end_line_no': getattr(node, 'end_lineno', node.lineno),
            })
    return sorted(funcs, key=lambda x: x['line_no'])


def slice_text(lines, start, end, pad=3):
    a = max(1, start - pad)
    b = min(len(lines), end + pad)
    return '\n'.join(f'{i:04d}: {lines[i-1]}' for i in range(a, b + 1))


def inspect_train_loader_usage(calls):
    suspicious = []
    for c in calls:
        fn = (c.get('func') or '').lower()
        kws = c.get('keywords', {})
        if fn.endswith('dataloader') or fn == 'dataloader' or fn.endswith('data.DataLoader'.lower()):
            if 'shuffle' in kws:
                suspicious.append({'type': 'dataloader_shuffle_kw', 'line_no': c['line_no'], 'value': kws['shuffle']})
            if 'sampler' in kws:
                suspicious.append({'type': 'dataloader_sampler_kw', 'line_no': c['line_no'], 'value': kws['sampler']})
    return suspicious


def inspect_random_ops(calls):
    suspects = []
    random_markers = ['random_split', 'np.random', 'random.shuffle', 'torch.randperm', 'permutation', 'shuffle']
    for c in calls:
        fn = c.get('func') or ''
        low = fn.lower()
        if any(m.lower() in low for m in random_markers):
            suspects.append({'func': fn, 'line_no': c.get('line_no'), 'keywords': c.get('keywords', {})})
    return suspects


def inspect_split_logic(lines, funcs, calls):
    patterns = [
        r'\btrain_ratio\b', r'\bval_ratio\b', r'\btest_ratio\b', r'\bshuffle\b', r'\bsplit\b',
        r'\bwalk\b', r'\bWFV\b', r'\bTimeSeriesSplit\b', r'\brandom_split\b', r'\bsampler\b',
        r'\bORACLE_MAX_HOLD\b', r'\bFORECAST_HORIZON\b', r'\bmax_hold\b', r'\blookback\b',
        r'\bseq_len\b', r'\bwindow\b', r'\bval\b', r'\btest\b'
    ]
    return grep_patterns(lines, patterns)


def boundary_risk_heuristics(text: str, lines: List[str]) -> List[Dict[str, Any]]:
    findings = []
    checks = [
        ('oracle_generation_present', r'generate_targets\s*\('),
        ('split_after_target_generation_risk', r'generate_targets[\s\S]{0,2500}(train|val|test|split)'),
        ('concat_before_split_risk', r'concat|pd\.concat|np\.concatenate'),
        ('iloc_split_present', r'iloc\s*\['),
        ('ratio_split_present', r'train_ratio|val_ratio|test_ratio'),
        ('wfv_present', r'WFV|walk.?forward'),
    ]
    for name, pat in checks:
        findings.append({'check': name, 'present': bool(re.search(pat, text, re.I))})
    for i, line in enumerate(lines, start=1):
        if 'generate_targets' in line or 'ORACLE_MAX_HOLD' in line or 'FORECAST_HORIZON' in line:
            findings.append({'line_no': i, 'context': line.strip()})
    return findings


def make_assessment(train_text, dl_text, train_lines, dl_lines, train_calls, dl_calls):
    issues = []
    notes = []

    loader_usage = inspect_train_loader_usage(train_calls) + inspect_train_loader_usage(dl_calls)
    random_ops = inspect_random_ops(train_calls) + inspect_random_ops(dl_calls)

    if any(x['type'] == 'dataloader_shuffle_kw' and x['value'] not in ('False', 'false') for x in loader_usage):
        issues.append('A DataLoader uses shuffle=True or non-false shuffle, which is risky for time-aware validation.')
    else:
        notes.append('No obvious DataLoader shuffle=True found in static call inspection.')

    if random_ops:
        issues.append('Randomized split/shuffle-related calls detected; inspect whether they affect train/val/test partitioning.')
    else:
        notes.append('No obvious random_split / randperm / permutation call detected in static inspection.')

    if re.search(r'generate_targets\s*\(', dl_text, re.I) and re.search(r'train_ratio|val_ratio|test_ratio|WFV|walk', dl_text, re.I):
        notes.append('data_loader.py appears to contain both target generation and split logic; boundary-crossing must be checked carefully.')
    if re.search(r'generate_targets\s*\(', dl_text, re.I) and not re.search(r'ORACLE_MAX_HOLD|max_hold', dl_text, re.I):
        issues.append('Targets may be generated without explicit boundary clipping references near split logic.')

    if re.search(r'WFV_ENABLED', train_text, re.I):
        notes.append('Walk-forward validation flags are referenced in train.py/config path, which is a good sign.')
    else:
        issues.append('No obvious WFV flag usage found in train.py.')

    if re.search(r'train_ratio|val_ratio|test_ratio', train_text, re.I) and re.search(r'np\.random|shuffle|random_split|randperm', train_text, re.I):
        issues.append('Ratio-based split logic combined with randomization suggests non-time-aware splitting risk.')

    return {'issues': issues, 'notes': notes, 'loader_usage': loader_usage, 'random_ops': random_ops}


def main():
    root = find_repo_root('.')
    train_path = os.path.join(root, 'train.py')
    dl_path = os.path.join(root, 'data_loader.py')

    train_text = read_text(train_path)
    dl_text = read_text(dl_path)
    train_lines = get_lines(train_text)
    dl_lines = get_lines(dl_text)

    train_calls = ast_calls_info(train_text)
    dl_calls = ast_calls_info(dl_text)
    train_funcs = extract_functions(train_text)
    dl_funcs = extract_functions(dl_text)

    train_hits = inspect_split_logic(train_lines, train_funcs, train_calls)
    dl_hits = inspect_split_logic(dl_lines, dl_funcs, dl_calls)

    assess = make_assessment(train_text, dl_text, train_lines, dl_lines, train_calls, dl_calls)
    boundary = {
        'train_py': boundary_risk_heuristics(train_text, train_lines),
        'data_loader_py': boundary_risk_heuristics(dl_text, dl_lines),
    }

    interesting_funcs = []
    keywords = ['split', 'fold', 'walk', 'loader', 'dataset', 'target', 'oracle', 'train', 'val', 'test']
    for src_name, funcs, lines in [('train.py', train_funcs, train_lines), ('data_loader.py', dl_funcs, dl_lines)]:
        for f in funcs:
            if any(k in f['name'].lower() for k in keywords):
                interesting_funcs.append({
                    'source': src_name,
                    'name': f['name'],
                    'line_no': f['line_no'],
                    'end_line_no': f['end_line_no'],
                    'snippet': slice_text(lines, f['line_no'], min(f['end_line_no'], f['line_no'] + 20), pad=2)
                })

    report = {
        'repo_root': root,
        'files_checked': [train_path, dl_path],
        'static_findings': assess,
        'boundary_risk_heuristics': boundary,
        'train_hits': train_hits[:250],
        'data_loader_hits': dl_hits[:250],
        'interesting_functions': interesting_funcs[:80],
        'audit_focus': [
            'Whether train/val/test or WFV splits are strictly time-ordered',
            'Whether DataLoader or samplers reintroduce shuffle',
            'Whether oracle target generation can cross split boundaries',
            'Whether ratio-based/randomized splitting exists',
        ],
        'manual_review_questions': [
            'Are targets generated before or after slicing each fold?',
            'If before splitting, are last ORACLE_MAX_HOLD samples removed from each split segment?',
            'Are validation and test windows entirely after training windows?',
            'Does any sampler operate on validation/test loaders?',
            'Does any concat/merge happen across files before splitting that could mix eras/regimes?',
        ]
    }

    os.makedirs('output/time_split_audit', exist_ok=True)
    with open('output/time_split_audit/time_split_audit.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    md = []
    md.append('# Time-Aware Split Audit')
    md.append('')
    md.append(f"Repo root: `{root}`")
    md.append('')
    md.append('## Static assessment')
    for item in assess['issues']:
        md.append(f'- ISSUE: {item}')
    for item in assess['notes']:
        md.append(f'- NOTE: {item}')
    md.append('')
    md.append('## Loader usage')
    for item in assess['loader_usage'][:50]:
        md.append(f"- line {item.get('line_no')}: {item.get('type')} = {item.get('value')}")
    md.append('')
    md.append('## Randomization suspects')
    for item in assess['random_ops'][:50]:
        md.append(f"- line {item.get('line_no')}: {item.get('func')} {item.get('keywords')}")
    md.append('')
    md.append('## Boundary heuristics')
    for src, items in boundary.items():
        md.append(f'### {src}')
        for item in items[:60]:
            if 'check' in item:
                md.append(f"- {item['check']}: {item['present']}")
            else:
                md.append(f"- line {item['line_no']}: {item['context']}")
        md.append('')
    md.append('## Key grep hits from train.py')
    for h in train_hits[:80]:
        md.append(f"- {h['line_no']}: `{h['line']}`")
    md.append('')
    md.append('## Key grep hits from data_loader.py')
    for h in dl_hits[:80]:
        md.append(f"- {h['line_no']}: `{h['line']}`")
    md.append('')
    md.append('## Interesting function snippets')
    for item in interesting_funcs[:20]:
        md.append(f"### {item['source']}::{item['name']} ({item['line_no']}-{item['end_line_no']})")
        md.append('```python')
        md.append(item['snippet'])
        md.append('```')
        md.append('')

    with open('output/time_split_audit/README.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print('Wrote output/time_split_audit/time_split_audit.json and README.md')


if __name__ == '__main__':
    main()