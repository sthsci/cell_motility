#!/usr/bin/env python3
"""Create local review/source archives; never uploads or publishes anything."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'manuscript'
EXCLUDED={'.pyc','.nbc','.nbi','.log','.aux','.out','.toc','.fls','.fdb_latexmk'}
def good(p):return p.is_file() and not any(x in p.parts for x in ('__pycache__','.ipynb_checkpoints')) and p.suffix not in EXCLUDED and p.name!='build_console.log'
def files_under(p):return [x for x in p.rglob('*') if good(x)]
def make(name,files):
 files=sorted(set(files));manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
 with zipfile.ZipFile(OUT/name,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files:z.write(p,p.relative_to(ROOT))
  z.writestr('ARCHIVE_SHA256.json',json.dumps(manifest,indent=2))
 return {'file':name,'files':len(files),'bytes':(OUT/name).stat().st_size,'sha256':hashlib.sha256((OUT/name).read_bytes()).hexdigest()}
paper=files_under(OUT/'nature_communications')+files_under(OUT/'nature_communications_supplement')+[OUT/'README.md',OUT/'main_template.tex',OUT/'supplement_template.tex',OUT/'output_checks.json']
paper+=[ROOT/'research'/n for n in ('journal_requirements.md','paper_evidence.md','paper_references.bib','paper_references.tex')]
source_data=files_under(OUT/'nature_communications/source_data')+[OUT/'nature_communications/figure_manifest.json',OUT/'nature_communications/numeric_claim_ledger.json']
full=paper+[ROOT/'pyproject.toml',ROOT/'README.md']
for path in ['src/motility_inference','src/mantishrimp','scripts','tests','results/synthetic_v1','results/manuscript_v2','notebooks','research']:full+=files_under(ROOT/path)
records=[make('journal_source.zip',paper),make('source_data.zip',source_data),make('reproducibility_bundle.zip',full)]
(OUT/'archives.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
