#!/usr/bin/env python3
"""Build self-contained journal-style LaTeX from templates and recorded results."""
from __future__ import annotations
import hashlib,json,re,sys,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from motility_inference.priors import PRIOR_LOW,PRIOR_HIGH
V1=ROOT/'results/synthetic_v1';V2=ROOT/'results/manuscript_v2';M=ROOT/'manuscript/nature_communications';S=ROOT/'manuscript/nature_communications_supplement'
def read(p):return json.loads(Path(p).read_text())
def pc(v,d=1):return f'{100*v:.{d}f}\\%'
def prange(values):return f'{100*min(values):.1f}--{100*max(values):.1f}\\%'
def wilson(a):return '['+pc(a[0])+', '+pc(a[1])+']'
def count(case,j=None):return sum(bool(r['any_detected'] if j is None else r['detected'][j]) for r in case['records'] if r.get('diagnostics',{}).get('converged',True))
def tex(s):
 s=str(s)
 names={'frozen_v1':'F0','fixed_117001':'F1','fixed_117002':'F2','fixed_117003':'F3',
 'conditioned_127001':'C1','conditioned_127002':'C2','conditioned_127003':'C3',
 'marginalized_127001':'M1','marginalized_127002':'M2','marginalized_127003':'M3',
 'varied_baseline_null':'Varied null','varied_null':'Varied null','fixed_null':'Fixed null',
 'null_sd015':'Null 0.15','null_sd060':'Null 0.60','combined_sd015':'Combined 0.15','combined_sd060':'Combined 0.60',
 'reference_combined':'Source combined','reference_mechanics':'Source mechanics','reference_null':'Source null','state_space':'State-space'}
 s=names.get(s,s)
 for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:s=s.replace(a,b)
 return s

def table(title,headers,rows):
 n=len(headers);width=.86/n
 out=[r'\subsection*{'+title+'}',r'\small',r'\begin{longtable}{'+''.join([f'>{{\\raggedright\\arraybackslash}}p{{{width:.3f}\\textwidth}}']*n)+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule',r'\endfirsthead',r'\toprule',' & '.join(headers)+r' \\',r'\midrule',r'\endhead']
 out+=[' & '.join(str(x) for x in row)+r' \\' for row in rows];out +=[r'\bottomrule',r'\end{longtable}',r'\normalsize']
 return '\n'.join(out)


def build():
 matched=read(V2/'sbi_matched_evaluation.json')['models'];noise=read(V2/'sbi_noise_evaluation.json')['models'];paired=read(V2/'sbi_paired_ssm_evaluation.json')['models'];plan=read(V2/'sbi_plan.json')
 ss={n:read(V2/f'ssm_{n}.json') for n in ['calibration','fixed_null','combined','varied_baseline_null']}
 assert all(v['complete'] and v['n_attempted']==v['n_planned'] for v in ss.values()),'Await all planned SSM records'
 cases=matched['frozen_v1']['scenarios'];a=cases['calibration']['aggregate'];c=ss['calibration']['aggregate'];nc=noise['conditioned_127001']['scenarios'];nm=noise['marginalized_127001']['scenarios']
 replacements={
 'NEURAL_COVERAGE_RANGE':prange(a['coverage90']),
 'NEURAL_EFFECT_RMSE':', '.join(f'{x:.3f}' for x in a['rmse'][3:]),
 'SSM_EFFECT_RMSE':', '.join(f'{x:.3f}' for x in c['rmse'][3:]),
 'SSM_COVERAGE_RANGE':prange(c['coverage90']),
 'SSM_CAL_PASSED':str(ss['calibration']['n_passed']),
 'NEURAL_NULL_COUNT':str(count(cases['null'])),'NEURAL_VARIED_NULL_COUNT':str(count(cases['varied_null'])),
 'NEURAL_NULL_WILSON':wilson(cases['null']['aggregate']['any_detection_wilson']),
 'NEURAL_VARIED_NULL_WILSON':wilson(cases['varied_null']['aggregate']['any_detection_wilson']),
 'NEURAL_SOLO_T_COUNT':str(count(cases['decorrelation'],1)),
 'SEED_COVERAGE_RANGES':', '.join(prange(matched[f'fixed_{seed}']['scenarios']['calibration']['aggregate']['coverage90']) for seed in (117001,117002,117003)),
 'NOISE_COND_COUNT':str(count(nc['null_sd060'])),'NOISE_COND_LOW_COUNT':str(count(nc['null_sd015'])),
 'NOISE_MARG_COUNT':str(count(nm['null_sd060'])),
 'NOISE_COND_COVERAGE':prange(nc['calibration_noise']['aggregate']['coverage90']),
 'NOISE_MARG_COVERAGE':prange(nm['calibration_noise']['aggregate']['coverage90']),
 'NOISE_COND_COMBINED_COUNT':str(count(nc['combined_sd060'])),
 'NOISE_MARG_COMBINED_COUNT':str(count(nm['combined_sd060'])),
 'NOISE_COND_SEED_COUNTS':', '.join(str(count(noise[f'conditioned_{seed}']['scenarios']['null_sd060'])) for seed in (127001,127002,127003)),
 'NOISE_MARG_SEED_COUNTS':', '.join(str(count(noise[f'marginalized_{seed}']['scenarios']['null_sd060'])) for seed in (127001,127002,127003)),
 'POSTERIOR_DRAWS':f"{plan['posterior_draws']:,}",
 'SSM_VARIED_TAU_COUNT':str(sum(r['covered90'][1] for r in ss['varied_baseline_null']['records'] if r['diagnostics']['converged'])),
 'SSM_VARIED_TAU_COVERAGE':pc(ss['varied_baseline_null']['aggregate']['coverage90'][1]),
 'SSM_VARIED_TAU_WILSON':wilson(ss['varied_baseline_null']['aggregate']['coverage90_wilson'][1]),
 }
 for j,key in enumerate(('S','TAU','J')):replacements[key+'_RELATIVE_WIDTH']=f"{a['mean_width90'][j+3]/(.9*(PRIOR_HIGH[j+3]-PRIOR_LOW[j+3])):.2f}"
 for sn,label in [('fixed_null','NULL'),('varied_baseline_null','VARIED_NULL'),('combined','COMBINED')]:
  replacements['SSM_'+label+'_N']=str(ss[sn]['n_passed']);replacements['SSM_'+label+'_COUNT']=str(count(ss[sn]));replacements['SSM_'+label+'_WILSON']=wilson(ss[sn]['aggregate']['any_detection_wilson'])
 for j,key in enumerate(('S','T','J')):
  replacements['NEURAL_COMBINED_'+key+'_COUNT']=str(count(cases['combined'],j));replacements['SSM_COMBINED_'+key+'_COUNT']=str(count(ss['combined'],j))
 main=(ROOT/'manuscript/main_template.tex').read_text();supp=(ROOT/'manuscript/supplement_template.tex').read_text()
 # Embed only cited references, in first-citation order, as required for single-file text.
 cited=[]
 for group in re.findall(r'\\cite\w*\{([^}]+)\}',main):
  for key in group.split(','):
   if key not in cited:cited.append(key)
 bib=(ROOT/'research/paper_references.tex').read_text();entries={}
 for match in re.finditer(r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})',bib,re.S):entries[match.group(1).lower()]=match.group(2).strip()
 missing=set(cited)-set(entries);assert not missing,missing
 replacements['REFERENCES']='\\begin{thebibliography}{99}\n'+'\n\n'.join('\\bibitem{'+k+'}\n'+entries[k] for k in cited)+'\n\\end{thebibliography}'
 # Numerical tables include every trained model rather than selecting by outcome.
 tables=[]
 rows=[['Calibration','96',str(ss['calibration']['n_passed']),str(ss['calibration']['n_failed'])],['Fixed null','96',str(ss['fixed_null']['n_passed']),str(ss['fixed_null']['n_failed'])],['Combined','96',str(ss['combined']['n_passed']),str(ss['combined']['n_failed'])],['Varied-baseline null','48',str(ss['varied_baseline_null']['n_passed']),str(ss['varied_baseline_null']['n_failed'])]]
 tables.append(table('Supplementary Table S1 | Expanded sampling counts',['Ensemble','Planned','Accepted','Unresolved'],rows))
 labels=[r'$\mu_S$',r'$\mu_\tau$',r'$\mu_J$',r'$\delta_S$',r'$\delta_\tau$',r'$\delta_J$']
 rows=[]
 for j,label in enumerate(labels):rows.append([label,pc(c['coverage90'][j])+' '+wilson(c['coverage90_wilson'][j]),f"{c['rmse'][j]:.3f}",f"{c['mean_width90'][j]:.3f}"])
 tables.append(table('Supplementary Table S2 | State-space calibration',['Parameter','90\\% coverage [interval]','RMSE','Mean 90\\% width'],rows))
 rows=[]
 for model,record in matched.items():
  a0=record['scenarios']['calibration']['aggregate']
  for j,l in enumerate(labels):rows.append([tex(model),l,pc(a0['coverage90'][j])+' '+wilson(a0['coverage90_wilson'][j]),f"{a0['rmse'][j]:.3f}"])
 tables.append(table('Supplementary Table S3 | Neural calibration by seed',['Network','Parameter','90\\% coverage [interval]','RMSE'],rows))
 rows=[]
 for model,record in matched.items():
  for sc,case in record['scenarios'].items():
   if sc=='calibration':continue
   ag=case['aggregate'];rows.append([tex(model),tex(sc),f"{count(case)}/{ag['n']}",wilson(ag['any_detection_wilson']),'/'.join(str(count(case,j)) for j in range(3))])
 tables.append(table('Supplementary Table S4 | Neural decisions by seed',['Network','Scenario','Any / $n$','95\\% interval','Counts $S/\\tau/J$'],rows))
 rows=[]
 for model,record in noise.items():
  for sc,case in record['scenarios'].items():
   if sc=='calibration_noise':continue
   ag=case['aggregate'];rows.append([tex(model),tex(sc),f"{count(case)}/{ag['n']}",wilson(ag['any_detection_wilson']),'/'.join(str(count(case,j)) for j in range(3))])
 tables.append(table('Supplementary Table S5 | Observation-noise decisions',['Network','Scenario','Any / $n$','95\\% interval','Counts $S/\\tau/J$'],rows))
 rows=[]
 for model,record in noise.items():
  a0=record['scenarios']['calibration_noise']['aggregate']
  for j,l in enumerate(labels):rows.append([tex(model),l,pc(a0['coverage90'][j]),f"{a0['rmse'][j]:.3f}",f"{a0['mean_width90'][j]:.3f}"])
 tables.append(table('Supplementary Table S6 | Calibration across noise',['Network','Parameter','90\\% coverage','RMSE','90\\% width'],rows))
 rows=[]
 for sc in ['fixed_null','combined','varied_baseline_null']:
  case=ss[sc];rows.append(['State-space',tex(sc),f"{count(case)}/{case['n_passed']}",wilson(case['aggregate']['any_detection_wilson']),'/'.join(str(count(case,j)) for j in range(3))])
  for model,record in paired.items():
   case=record['scenarios'][sc];ag=case['aggregate'];rows.append([tex(model),tex(sc),f"{count(case)}/{ag['n']}",wilson(ag['any_detection_wilson']),'/'.join(str(count(case,j)) for j in range(3))])
 tables.append(table('Supplementary Table S7 | Paired movie decisions',['Method','Scenario','Any / $n$','95\\% interval','Counts $S/\\tau/J$'],rows))
 rows=[]
 for sc,record in read(V1/'reference_validation.json').items():
  for method in ['neural','abc','state_space']:
   rr=record[method];rr=[r for r in rr if r.get('diagnostics',{}).get('converged',True)];case={'records':rr};rows.append([tex(sc),tex(method),str(len(rr)),str(count(case)),'/'.join(str(count(case,j)) for j in range(3))])
 tables.append(table('Supplementary Table S8 | Independent source transfer',['Scenario','Method','$n$','Any count','Counts $S/\\tau/J$'],rows))
 replacements['NUMERICAL_TABLES']='\n\n'.join(tables)
 for key,value in replacements.items():main=main.replace('@@'+key+'@@',value);supp=supp.replace('@@'+key+'@@',value)
 assert not re.search(r'@@\w+@@',main+supp),re.findall(r'@@\w+@@',main+supp)
 (M/'main.tex').write_text(main);(S/'main.tex').write_text(supp)
 # Scope-bound provenance: generated values are machine linked; humans remain reviewers.
 facts={k:v for k,v in replacements.items() if k not in ['REFERENCES','NUMERICAL_TABLES']}
 ledger={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'assistant_checked_human_verification_pending','numeric_replacements':facts,'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [V2/'sbi_matched_evaluation.json',V2/'sbi_noise_evaluation.json',V2/'sbi_paired_ssm_evaluation.json',V2/'ssm_plan.json',V2/'sbi_plan.json',V1/'speed.json',V1/'reference_validation.json',V1/'timestep_sensitivity.json']+[V2/f'ssm_{k}.json' for k in ss]},'static_benchmark_facts':'Verified in results/synthetic_v1/REPORT.md and the source files cited by that report. Mathematical identities are derived in Supplementary Information.','bibliography_keys_in_order':cited,'submission_ready':False}
 (M/'numeric_claim_ledger.json').write_text(json.dumps(ledger,indent=2))
 def words(text):
  text=re.sub(r'%[^\n]*','',text) if '\\%' not in text else text
  text=re.sub(r'\\(?:cite\w*|label|ref)\{[^}]*\}','',text);text=re.sub(r'\\[A-Za-z]+\*?(?:\[[^]]*\])?',' ',text);text=re.sub(r'[{}$\\]',' ',text)
  return len(re.findall(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*",text))
 abstract=main.split(r'\begin{abstract}',1)[1].split(r'\end{abstract}',1)[0];body=main.split(r'\linenumbers',1)[1].split(r'\section*{Methods}',1)[0]
 titles=re.findall(r'\\subsection\*\{([^}]+)\}',main);legends=main.split(r'\section*{Figure legends}',1)[1].split(r'\section*{Tables}',1)[0]
 leg_parts=re.split(r'\\textbf\{Figure \d+ \|',legends)[1:]
 audit={'title_words':words(re.search(r'\\title\{([^}]+)\}',main)[1]),'abstract_words':words(abstract),'main_text_words_excluding_methods':words(body),'max_subheading_characters':max(map(len,titles)),'figure_legend_words':[words(x) for x in leg_parts],'main_figures':5,'main_tables':1,'bibliography_count':len(cited),'author_field_blank':r'\author{}' in main,'remaining_numeric_tokens':False,'declared_pending':['human scientific/reference verification','authorship and affiliations','funding','contributions','competing interests','public data/code archival identifiers','distribution license'],'submission_ready':False,'official_requirements_record':'research/journal_requirements.md'}
 (M/'format_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
 assert audit['abstract_words']<=150 and audit['main_text_words_excluding_methods']<=6000 and audit['max_subheading_characters']<=60 and max(audit['figure_legend_words'])<=350
 print('Wrote main and supplementary self-contained LaTeX, numerical ledger and format audit.')
if __name__=='__main__':build()
