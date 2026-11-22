"""The notices of other people's work in the Web build: python3 tools/notices.py > licenses.txt
The crates linked into the wasm (cargo tree: the normal dependencies for wasm32-unknown-unknown) and the routines of
Aphros in the interface shader, each licence text once, with the crates that carry it. The licences ask that their
notices go with every copy, and a public Web build is one. Trunk runs this after a build (Trunk.toml)."""
import collections, json, pathlib, re, subprocess, sys

root = pathlib.Path(__file__).resolve().parent.parent
tree = subprocess.run(['cargo', 'tree', '--target', 'wasm32-unknown-unknown', '-e', 'normal', '--prefix', 'none', '--format', '{p}'],
                      cwd=root, capture_output=True, text=True, check=True).stdout
linked = set()
for line in tree.splitlines():
    found = re.match(r'(\S+) v(\S+)', line)
    if found: linked.add(found.groups())
metadata = json.loads(subprocess.run(['cargo', 'metadata', '--format-version', '1'], cwd=root, capture_output=True, text=True, check=True).stdout)
NAMES = re.compile(r'(licen[sc]e|copying|copyright|notice|unlicense|ofl|ufl)', re.I)
texts = collections.defaultdict(list)   # a licence text -> the crates that carry it
bare = []                               # crates whose package holds no licence text
for package in sorted(metadata['packages'], key=lambda p: (p['name'], p['version'])):
    if (package['name'], package['version']) not in linked or package['name'] == 'sand-picture': continue
    label = f"{package['name']} {package['version']} ({package.get('license') or 'no licence declared'})"
    if package.get('repository'): label += f" {package['repository']}"
    folder = pathlib.Path(package['manifest_path']).parent
    files = [f for depth in ('*', '*/*') for f in sorted(folder.glob(depth)) if f.is_file() and NAMES.search(f.name) and f.suffix.lower() in ('', '.txt', '.md')]
    if not files: bare.append(label)
    for file in files:
        text = file.read_text(errors='replace').replace('\r\n', '\n').strip()
        texts[text].append(label)
out = sys.stdout
out.write('Sand Picture: notices of the work of others that this build carries\n\n')
out.write('=' * 78 + '\nAphros (ETH Zurich): routines of the interface shader are transcribed from it.\n\n')
out.write((root / 'src/model/APHROS_LICENSE.txt').read_text().strip() + '\n\n')
for text, crates in sorted(texts.items(), key=lambda item: (-len(item[1]), item[1][0])):
    out.write('=' * 78 + '\n' + '\n'.join(sorted(set(crates))) + '\n\n' + text + '\n\n')
if bare:
    out.write('=' * 78 + '\nCrates whose packages hold no licence text, under the licences they declare:\n' + '\n'.join(bare) + '\n')
