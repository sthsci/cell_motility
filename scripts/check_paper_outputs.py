#!/usr/bin/env python3
"""Inspect compiled paper artifacts, font embedding, figures and numeric tokens."""
import ast,hashlib,json,re,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'manuscript';PREVIEW=OUT/'inspection';PREVIEW.mkdir(exist_ok=True)
report={}
for name in ['nature_communications','nature_communications_supplement']:
 p=OUT/name;pdf=p/'main.pdf';source=(p/'main.tex').read_text();log=(p/'main.log').read_text()
 assert not re.search(r'@@\w+@@',source)
 warnings=[l for l in log.splitlines() if any(x in l for x in ('Warning','Overfull','Underfull','! LaTeX'))]
 assert not warnings,(name,warnings)
 pages=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True).split('\f')
 pages=[x for x in pages if x.strip()]
 font_text=subprocess.check_output(['pdffonts',str(pdf)],text=True);lines=font_text.splitlines();idx=lines[0].index('emb')
 assert all(line[idx:idx+3]=='yes' for line in lines[2:] if line.strip()),font_text
 assert 'Type 3' not in font_text
 chosen=[0]
 searches=['Results','Methods','Figure legends','Tables'] if name=='nature_communications' else ['Supplementary Table S4','Supplementary Table S6']
 for word in searches:
  candidates=[i for i,text in enumerate(pages) if word in text and i>0]
  if candidates:chosen.append(candidates[-1] if 'Table' in word else candidates[0])
 chosen=sorted(set(chosen))
 for i in chosen:subprocess.run(['pdftoppm','-f',str(i+1),'-l',str(i+1),'-png','-singlefile','-scale-to','1300',str(pdf),str(PREVIEW/f'{name}_{i+1:02d}')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 report[name]={'pages':len(pages),'bytes':pdf.stat().st_size,'latex_warnings':warnings,'all_fonts_embedded':True,'type3_fonts':False,'preview_pages':[i+1 for i in chosen],'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()}
 (p/'fonts.txt').write_text(font_text)
for path in (ROOT/'scripts').glob('*.py'):ast.parse(path.read_text())
figures=json.loads((OUT/'nature_communications/figure_manifest.json').read_text());assert len(figures)==13
for f in figures:
 folder=OUT/('nature_communications_supplement' if f['supplementary'] else 'nature_communications')
 assert all((folder/'figures'/f"{f['figure']}.{ext}").exists() for ext in ['pdf','svg','png'])
 assert (OUT/'nature_communications'/f['source_data']).exists()
root=ET.parse(ROOT/'results/manuscript_v2/core_tests.xml').getroot();suites=list(root) if root.tag=='testsuites' else [root]
report['core_tests']={'tests':sum(int(s.attrib.get('tests',0)) for s in suites),'failures':sum(int(s.attrib.get('failures',0)) for s in suites),'errors':sum(int(s.attrib.get('errors',0)) for s in suites)}
report['figure_count']=len(figures);report['format']=json.loads((OUT/'nature_communications/format_audit.json').read_text())
(OUT/'output_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
