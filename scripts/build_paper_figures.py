#!/usr/bin/env python3
"""Publication figures and per-figure CSV source data from recorded results only."""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch
from motility_inference.simulator import load_dataset,proximity
from motility_inference.priors import PRIOR_LOW,PRIOR_HIGH
from motility_inference.evaluation import wilson_interval
V1=ROOT/'results/synthetic_v1';V2=ROOT/'results/manuscript_v2'
MAIN=ROOT/'manuscript/nature_communications';SUPP=ROOT/'manuscript/nature_communications_supplement'
C={'neural':'#0F4D92','state_space':'#007F78','abc':'#B64342','conditioned':'#007F78','marginalized':'#B37900'}
LABELS=[r'$\mu_S$',r'$\mu_\tau$',r'$\mu_J$',r'$\delta_S$',r'$\delta_\tau$',r'$\delta_J$']
METHODS={'neural':'Neural SBI','state_space':'State-space','abc':'Rejection ABC'}
MANIFEST=[]
def read(path):return json.loads(Path(path).read_text())
def panel(ax,letter):ax.text(-.16,1.07,letter,transform=ax.transAxes,fontweight='bold',fontsize=9,va='top')
def save(fig,name,rows,supp=False):
 out=SUPP if supp else MAIN;(out/'figures').mkdir(parents=True,exist_ok=True);(MAIN/'source_data').mkdir(exist_ok=True)
 for ext in ('pdf','svg','png'):fig.savefig(out/'figures'/f'{name}.{ext}',dpi=400,facecolor='white')
 plt.close(fig)
 pd.DataFrame(rows).to_csv(MAIN/'source_data'/f'{name}.csv',index=False)
 MANIFEST.append({'figure':name,'supplementary':supp,'source_data':f'source_data/{name}.csv','width_mm':float(fig.get_figwidth()*25.4),'height_mm':float(fig.get_figheight()*25.4)})
def err(ax,x,p,interval,color,marker='o',label=None):
 lo,hi=interval;ax.errorbar(x,p,yerr=[[max(0,p-lo)],[max(0,hi-p)]],fmt=marker,color=color,ms=4,capsize=2,lw=1,label=label)
def ratio_label(ax):ax.set_ylim(-.04,1.04);ax.set_yticks([0,.25,.5,.75,1])
def fig1():
 d=load_dataset(V1/'examples/combined.npz');cfg=d['config'];g=proximity(d['positions'],d['targets'],cfg['box_size'],cfg['contact_distance'],cfg['interaction_length'])[0]
 cell=int(np.argmax(np.ptp(g,axis=1)));paths=np.load(V1/'examples/combined_paths.npz',allow_pickle=False)
 fig,axes=plt.subplots(2,3,figsize=(7.087,5.15),layout='constrained');rows=[]
 ax=axes[0,0];x=d['positions'][cell]%cfg['box_size'];seg=np.stack([x[:-1],x[1:]],1);valid=np.linalg.norm(x[1:]-x[:-1],axis=1)<cfg['box_size']/2
 lc=LineCollection(seg[valid],array=g[cell,:-1][valid],cmap='viridis',linewidth=1);lc.set_clim(0,1);ax.add_collection(lc)
 for t in d['targets'][:,0]:ax.add_patch(Circle(t,cfg['contact_distance'],facecolor='.92',edgecolor='.65',lw=1))
 ax.plot(x[0,0],x[0,1],'ks',ms=3);ax.set(xlim=(0,180),ylim=(0,180),aspect='equal',xlabel='x (µm)',ylabel='y (µm)',title='Synthetic target encounters')
 fig.colorbar(lc,ax=ax,shrink=.7,ticks=[0,1],label='Proximity')
 for t in range(len(d['times'])):rows.append({'panel':'a','time_min':d['times'][t],'cell':cell,'x_um':x[t,0],'y_um':x[t,1],'gate':g[cell,t]})
 ax=axes[0,1];gap=np.linspace(0,60,150);gate=np.exp(-gap/10)
 ax.plot(gap,np.exp(-.8*gate),color=C['neural'],label=r'$S/S_0$, $\tau/\tau_0$');ax.plot(gap,np.exp(.8*gate),'--',color=C['abc'],label=r'$J/J_0$')
 ax.axhline(1,color='.6',ls=':',lw=1);ax.set(xlabel='Surface gap (µm)',ylabel='Local / far-field value',title='Programmed distance response',ylim=(0,2.5));ax.legend(fontsize=7,loc='upper right')
 for x0,g0 in zip(gap,gate):rows.append({'panel':'b','gap_um':x0,'speed_ratio':np.exp(-.8*g0),'tau_ratio':np.exp(-.8*g0),'jitter_ratio':np.exp(.8*g0)})
 ax=axes[0,2];ax.axis('off');ax.set_title('Two inference routes')
 boxes=[(.5,.88,'Positions + targets'),(.5,.65,'Distance histories'),(.23,.38,'Kalman likelihood\n+ hierarchy'),(.78,.38,'Track summaries\n+ neural posterior'),(.5,.10,'Effect uncertainty')]
 for x0,y0,txt in boxes:ax.text(x0,y0,txt,ha='center',va='center',fontsize=6.8,bbox=dict(boxstyle='round,pad=.4',fc='.96',ec='.5',lw=1),transform=ax.transAxes)
 for a,b in [((.5,.81),(.5,.71)),((.45,.58),(.24,.48)),((.55,.58),(.78,.48)),((.24,.28),(.44,.17)),((.78,.28),(.56,.17))]:ax.add_patch(FancyArrowPatch(a,b,arrowstyle='->',mutation_scale=8,lw=1,transform=ax.transAxes,color='.3'))
 for ax,name,b,e,ylab in zip(axes[1],('speed','tau','jitter'),(0,1,2),(3,4,5),('Active velocity SD (µm/min)','Persistence time (min)','Jitter amplitude (µm/√min)')):
  truth=np.exp(d['theta'][b]+(d['cell_offsets'][cell] if b==0 else 0)+d['theta'][e]*d['true_gate'][cell])
  ax.fill_between(d['times'],paths[name+'_lower'][cell],paths[name+'_upper'][cell],color=C['state_space'],alpha=.22,lw=0)
  ax.plot(d['times'],truth,'k-',label='Generating value');ax.plot(d['times'],paths[name+'_median'][cell],color=C['state_space'],label='Posterior median')
  ax.set(xlabel='Time (min)',ylabel=ylab)
  for t,time in enumerate(d['times']):rows.append({'panel':name,'time_min':time,'truth':truth[t],'median':paths[name+'_median'][cell,t],'lower90':paths[name+'_lower'][cell,t],'upper90':paths[name+'_upper'][cell,t]})
 axes[1,0].legend(fontsize=6.5,loc='upper right')
 for ax,letter in zip(axes.flat,'abcdef'):panel(ax,letter)
 save(fig,'fig1_model',rows)

def expanded():
 sb=read(V2/'sbi_matched_evaluation.json')['models']['frozen_v1']['scenarios'];ss=read(V2/'ssm_calibration.json')
 return sb,ss

def fig2():
 sb,ss=expanded();nr=sb['calibration'];aa=read(V1/'calibration.json')
 data={'neural':nr,'state_space':ss,'abc':{'aggregate':aa['aggregate']['abc'],'records':aa['records']['abc']}}
 fig=plt.figure(figsize=(7.087,5.1),layout='constrained');gs=fig.add_gridspec(2,6);axes=[fig.add_subplot(gs[0,:3]),fig.add_subplot(gs[0,3:])]+[fig.add_subplot(gs[1,2*j:2*j+2]) for j in range(3)];rows=[]
 for method,offset,marker in [('abc',-.18,'^'),('neural',0,'o'),('state_space',.18,'s')]:
  a=data[method]['aggregate']
  for j in range(6):
   lo,hi=a['coverage90_wilson'][j];err(axes[0],j+offset,a['coverage90'][j],(lo,hi),C[method],marker,label=METHODS[method] if j==0 else None)
   width=a['mean_width90'][j]/(.9*(PRIOR_HIGH[j]-PRIOR_LOW[j]));axes[1].plot(j+offset,width,marker,color=C[method],ms=4)
   rows.append({'panel':'ab','method':method,'parameter':j,'n':a['n'],'coverage90':a['coverage90'][j],'coverage90_low':lo,'coverage90_high':hi,'relative_width90':width,'rmse':a['rmse'][j]})
 axes[0].axhline(.9,ls='--',color='.5');axes[0].set(xticks=range(6),xticklabels=LABELS,ylabel='Empirical 90% coverage',ylim=(.60,1.02));axes[0].legend(fontsize=6.8,loc='lower left')
 axes[1].axhline(1,color='.5',ls=':');axes[1].set(xticks=range(6),xticklabels=LABELS,ylabel='Posterior / prior 90% width',ylim=(0,1.05))
 for j,ax in enumerate(axes[2:]):
  for method,marker in [('neural','o'),('state_space','x')]:
   rr=data[method]['records'];rr=[r for r in rr if r.get('diagnostics',{}).get('converged',True)]
   truth=[r['truth'][j+3] for r in rr];med=[r['median'][j+3] for r in rr];ax.scatter(truth,med,s=7 if method=='neural' else 13,alpha=.25 if method=='neural' else .8,c=C[method],marker=marker,lw=.7)
   for t,m in zip(truth,med):rows.append({'panel':str(j+3),'method':method,'truth':t,'median':m})
  lo=PRIOR_LOW[j+3];hi=PRIOR_HIGH[j+3];ax.plot([lo,hi],[lo,hi],'--',color='.4');ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Generating '+LABELS[j+3],ylabel='Posterior median')
 for ax,l in zip(axes,'abcde'):panel(ax,l)
 save(fig,'fig2_calibration',rows)

def fig3():
 sb,_=expanded();fig,axes=plt.subplots(1,3,figsize=(7.087,2.65),layout='constrained');rows=[]
 cases=[('null','Fixed null'),('varied_null','Varied baseline')]
 for k,(sn,label) in enumerate(cases):
  a=sb[sn]['aggregate'];err(axes[0],k-.1,a['any_detection_rate'],a['any_detection_wilson'],C['neural'],label='Neural SBI' if k==0 else None)
  sf='ssm_fixed_null.json' if k==0 else 'ssm_varied_baseline_null.json';s=read(V2/sf);b=s['aggregate'];err(axes[0],k+.1,b['any_detection_rate'],b['any_detection_wilson'],C['state_space'],'s',label='State-space' if k==0 else None)
  for method,a0 in [('neural',a),('state_space',b)]:rows.append({'panel':'a','scenario':sn,'method':method,'n':a0['n'],'rate':a0['any_detection_rate'],'low':a0['any_detection_wilson'][0],'high':a0['any_detection_wilson'][1]})
 axes[0].set(xticks=[0,1],xticklabels=['Fixed\nnull','Varied\nbaseline'],ylabel='Any-effect declaration rate',ylim=(-.01,.25));axes[0].legend(fontsize=6.8)
 ss=read(V2/'ssm_combined.json');nr=sb['combined']
 for method,case,shift,marker in [('neural',nr,-.10,'o'),('state_space',ss,.10,'s')]:
  a=case['aggregate'];records=[r for r in case['records'] if r.get('diagnostics',{}).get('converged',True)]
  for j in range(3):
   n=len(records);k=sum(bool(r['detected'][j]) for r in records);ci=wilson_interval(k,n);err(axes[1],j+shift,k/n,ci,C[method],marker)
   rows.append({'panel':'b','method':method,'effect':j,'n':n,'count':k,'rate':k/n,'low':ci[0],'high':ci[1]})
 axes[1].set(xticks=range(3),xticklabels=[r'Lower $S$',r'Shorter $\tau$',r'Higher $J$'],ylabel='Effect detection, combined');ratio_label(axes[1])
 names=['slowdown','decorrelation','jitter','weak','combined'];mat=np.array([sb[n]['aggregate']['detection_rate'] for n in names]);im=axes[2].imshow(mat,vmin=0,vmax=1,cmap='Blues',aspect='auto')
 for i,n in enumerate(names):
  for j in range(3):axes[2].text(j,i,f'{mat[i,j]:.0%}',ha='center',va='center',fontsize=7,color='white' if mat[i,j]>.6 else 'black');rows.append({'panel':'c','scenario':n,'effect':j,'rate':mat[i,j],'n':sb[n]['aggregate']['n']})
 axes[2].set(xticks=range(3),xticklabels=[r'$\delta_S$',r'$\delta_\tau$',r'$\delta_J$'],yticks=range(5),yticklabels=['Slowdown','Persistence','Jitter','Weak mix','Strong mix'],title='Neural per-effect detection');axes[2].tick_params(length=0)
 for ax,l in zip(axes,'abc'):panel(ax,l)
 save(fig,'fig3_identifiability',rows)

def fig4():
 noise=read(V2/'sbi_noise_evaluation.json')['models'];fig,axes=plt.subplots(2,2,figsize=(7.087,4.9),layout='constrained');rows=[]
 methods=[('frozen_v1','neural','Fixed-noise model','o'),('marginalized_127001','marginalized','Noise marginalized','^'),('conditioned_127001','conditioned','Known-noise model','s')]
 for index,(model,color,label,marker) in enumerate(methods):
  scenarios=noise[model]['scenarios'];shift=(index-1)*.13
  for k,sc in enumerate(['null_sd015','null_sd060']):
   a=scenarios[sc]['aggregate'];err(axes[0,0],k+shift,a['any_detection_rate'],a['any_detection_wilson'],C[color],marker,label=label if k==0 else None)
   rows.append({'panel':'a','model':model,'scenario':sc,'n':a['n'],'rate':a['any_detection_rate'],'low':a['any_detection_wilson'][0],'high':a['any_detection_wilson'][1]})
  a=scenarios['calibration_noise']['aggregate']
  for j in range(6):
   err(axes[0,1],j+shift,a['coverage90'][j],a['coverage90_wilson'][j],C[color],marker)
   rows.append({'panel':'b','model':model,'parameter':j,'n':a['n'],'coverage90':a['coverage90'][j],'low':a['coverage90_wilson'][j][0],'high':a['coverage90_wilson'][j][1]})
  case=scenarios['combined_sd060'];a=case['aggregate']
  for j in range(3):
   n=a['n'];k=sum(bool(r['detected'][j]) for r in case['records']);ci=wilson_interval(k,n);err(axes[1,0],j+shift,k/n,ci,C[color],marker)
   rows.append({'panel':'c','model':model,'effect':j,'n':n,'rate':k/n,'low':ci[0],'high':ci[1]})
  for k,sc in enumerate(['null_sd015','null_sd060']):
   rr=np.array([r['median'][3] for r in scenarios[sc]['records']]);q=np.quantile(rr,[.1,.5,.9]);axes[1,1].errorbar(k+shift,q[1],yerr=[[q[1]-q[0]],[q[2]-q[1]]],fmt=marker,color=C[color],capsize=2,ms=4)
   for r in rr:rows.append({'panel':'d','model':model,'scenario':sc,'delta_speed_median':r})
 axes[0,0].set(xticks=[0,1],xticklabels=['0.15','0.60'],xlabel='Localization SD (µm)',ylabel='Null declaration rate');ratio_label(axes[0,0]);axes[0,0].legend(fontsize=6.8,loc='upper left')
 axes[0,1].axhline(.9,color='.4',ls='--');axes[0,1].set(xticks=range(6),xticklabels=LABELS,ylabel='90% coverage across noise');ratio_label(axes[0,1])
 axes[1,0].set(xticks=range(3),xticklabels=[r'Lower $S$',r'Shorter $\tau$',r'Higher $J$'],ylabel='Combined detection at 0.60 µm');ratio_label(axes[1,0])
 axes[1,1].axhline(0,color='.4',ls='--');axes[1,1].set(xticks=[0,1],xticklabels=['0.15','0.60'],xlabel='Localization SD (µm)',ylabel=r'Posterior median $\delta_S$ under null')
 for ax,l in zip(axes.flat,'abcd'):panel(ax,l)
 save(fig,'fig4_observation_noise',rows)

def fig5():
 ref=read(V1/'reference_validation.json');sc=read(V1/'scenarios.json');ss=read(V1/'state_space_scenarios.json');rows=[]
 fig,axes=plt.subplots(1,3,figsize=(7.087,2.8),layout='constrained')
 for method,shift,marker in [('neural',-.1,'o'),('state_space',.1,'s')]:
  case=ref['reference_combined'];records=case[method];records=[r for r in records if r.get('diagnostics',{}).get('converged',True)]
  for j in range(3):
   n=len(records);k=sum(bool(r['detected'][j]) for r in records);ci=wilson_interval(k,n);err(axes[0],j+shift,k/n,ci,C[method],marker,label=METHODS[method] if j==0 else None)
   rows.append({'panel':'a','method':method,'effect':j,'n':n,'rate':k/n,'low':ci[0],'high':ci[1]})
  for i,name in enumerate(['null','spontaneous_null','mechanical_null']):
   a=sc[name]['aggregate']['neural'] if method=='neural' else ss[name]['aggregate'];err(axes[1],i+shift,a['any_detection_rate'],a['any_detection_wilson'],C[method],marker)
   rows.append({'panel':'b','method':method,'scenario':name,'n':a['n'],'rate':a['any_detection_rate'],'low':a['any_detection_wilson'][0],'high':a['any_detection_wilson'][1]})
 for i,name in enumerate(['null','spontaneous_null','mechanical_null']):
  a=ss[name]['aggregate'];err(axes[2],i,a['coverage90'][3],a['coverage90_wilson'][3],C['state_space'],'s');rows.append({'panel':'c','scenario':name,'n':a['n'],'coverage90':a['coverage90'][3],'low':a['coverage90_wilson'][3][0],'high':a['coverage90_wilson'][3][1]})
 axes[0].set(xticks=range(3),xticklabels=[r'Lower $S$',r'Shorter $\tau$',r'Higher $J$'],ylabel='Detection in supplied ABM');ratio_label(axes[0]);fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncols=2,fontsize=7)
 for ax in axes[1:]:ax.set(xticks=range(3),xticklabels=['Matched\nnull','Temporal\nvariation','Mechanical\nonly']);ratio_label(ax)
 axes[1].set_ylabel('Any intrinsic-effect declaration');axes[2].set_ylabel(r'90% coverage of $\delta_S$ (state-space)');axes[2].axhline(.9,color='.5',ls='--')
 for ax,l in zip(axes,'abc'):panel(ax,l)
 save(fig,'fig5_attribution',rows)

def supplementary():
 sb,_=expanded();rr=sb['calibration']['records'];ranks=np.array([r['rank_fraction'] for r in rr]);rows=[]
 from scipy.stats import binom
 fig,axes=plt.subplots(2,3,figsize=(7.087,4.5),layout='constrained');low,high=binom.interval(.95,len(rr),.1)
 for j,ax in enumerate(axes.flat):
  ax.axhspan(low,high,color='.9');counts,edges,_=ax.hist(ranks[:,j],bins=np.linspace(0,1,11),color=C['neural'],edgecolor='white',lw=1);ax.axhline(len(rr)/10,color='.4',ls='--');ax.set(xlabel='Rank fraction',ylabel='Movies',title=LABELS[j]);panel(ax,'abcdef'[j])
  for a,b,c in zip(edges[:-1],edges[1:],counts):rows.append({'parameter':j,'left':a,'right':b,'count':c,'pointwise_band_low':low,'pointwise_band_high':high})
 save(fig,'figS1_rank_calibration',rows,True)
 data=read(V2/'sbi_matched_evaluation.json')['models'];fig,axes=plt.subplots(1,2,figsize=(7.087,2.8),layout='constrained');rows=[]
 for i,(model,d) in enumerate(data.items()):
  a=d['scenarios']['calibration']['aggregate'];style=['o','s','^','D'][i%4];color=['#0F4D92','#007F78','#B37900','#9A4D8E'][i%4]
  axes[0].plot(range(6),a['coverage90'],marker=style,color=color,label=model.replace('fixed_','seed '));axes[1].plot(range(6),a['rmse'],marker=style,color=color)
  for j in range(6):rows.append({'model':model,'parameter':j,'n':a['n'],'coverage90':a['coverage90'][j],'rmse':a['rmse'][j]})
 axes[0].axhline(.9,color='.5',ls='--');axes[0].set_ylabel('Empirical 90% coverage');axes[0].legend(fontsize=6.8);axes[1].set_ylabel('Parameter RMSE (log units)')
 for ax,l in zip(axes,'ab'):ax.set(xticks=range(6),xticklabels=LABELS);panel(ax,l)
 save(fig,'figS2_training_seeds',rows,True)
 noise=read(V2/'sbi_noise_evaluation.json')['models'];fig,axes=plt.subplots(1,2,figsize=(7.087,2.8),layout='constrained');rows=[]
 for i,(model,d) in enumerate(noise.items()):
  if model=='frozen_v1':continue
  typ='conditioned' if model.startswith('conditioned') else 'marginalized';seed=int(model[-1]);marker=['o','s','^'][seed-1];a=d['scenarios']['calibration_noise']['aggregate']
  axes[0].plot(range(6),a['coverage90'],marker=marker,color=C[typ],alpha=.75,label=model.replace('_12700',' '))
  for k,sc in enumerate(['null_sd015','null_sd060','combined_sd015','combined_sd060']):
   b=d['scenarios'][sc]['aggregate'];axes[1].plot(k+(seed-2)*.10+(-.19 if typ=='conditioned' else .19),b['any_detection_rate'],marker,color=C[typ],ms=3.5);rows.append({'model':model,'scenario':sc,'n':b['n'],'any_rate':b['any_detection_rate']})
  for j in range(6):rows.append({'model':model,'parameter':j,'n':a['n'],'coverage90':a['coverage90'][j]})
 axes[0].axhline(.9,ls='--',color='.5');axes[0].set(xticks=range(6),xticklabels=LABELS,ylabel='90% coverage across noise');fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncols=3,fontsize=6.7)
 axes[1].set(xticks=range(4),xticklabels=['Null\n0.15','Null\n0.60','Combined\n0.15','Combined\n0.60'],ylabel='Any-effect declaration rate');ratio_label(axes[1])
 for ax,l in zip(axes,'ab'):panel(ax,l)
 save(fig,'figS3_noise_training_seeds',rows,True)
 # Parameter traces for the fixed first combined replicate, no selection by fit quality.
 z=np.load(V1/'examples/combined_ssm.npz',allow_pickle=False);samples=z['samples'];fig,axes=plt.subplots(2,3,figsize=(7.087,4.3),layout='constrained');rows=[]
 colors=['#0F4D92','#007F78','#B37900','#9A4D8E'];stride=max(1,samples.shape[1]//500)
 for j,ax in enumerate(axes.flat):
  for c in range(samples.shape[0]):
   idx=np.arange(0,samples.shape[1],stride);ax.plot(idx,samples[c,idx,j],lw=1,alpha=.5,color=colors[c],label=f'Chain {c+1}')
   for it,val in zip(idx,samples[c,idx,j]):rows.append({'parameter':j,'chain':c+1,'iteration':int(it),'value':val,'display_stride':stride})
  ax.set(xlabel='Retained iteration',ylabel=LABELS[j]);panel(ax,'abcdef'[j])
 axes[0,0].legend(fontsize=6.2,ncol=2);save(fig,'figS4_sampler_traces',rows,True)
 # Predictive checks display standardized residuals relative to predictive SD.
 rows=[];fig,axes=plt.subplots(3,1,figsize=(7.087,4.8),layout='constrained')
 for ax,case,l in zip(axes,['null','combined','mechanical_null'],'abc'):
  d=read(V1/'examples'/f'{case}_ppc.json')
  for method in ('neural','state_space'):
   q=d[method];reps=np.array(q['replicate_summaries']);mu=reps.mean(0);sd=reps.std(0);res=(np.array(q['observed'])-mu)/np.maximum(sd,1e-8);ax.plot(range(len(res)),res,'o' if method=='neural' else 'x',color=C[method],ms=3,label=METHODS[method])
   for j,r in enumerate(res):rows.append({'case':case,'method':method,'feature':j,'observed':q['observed'][j],'predictive_mean':mu[j],'predictive_sd':sd[j],'standardized_discrepancy':r,'lower90':q['lower90'][j],'upper90':q['upper90'][j]})
  ax.axhline(0,color='.5',ls='--');ax.set(ylabel='Standardized\ndiscrepancy',title=case.replace('mechanical_null','mechanical_only').replace('_',' '));panel(ax,l)
 axes[0].legend(fontsize=6.8,ncol=2);axes[-1].set_xlabel('Summary feature index (ordered as implementation)');save(fig,'figS6_predictive_checks',rows,True)
 # Diagnostics across newly generated datasets; failure handling is encoded in source records.
 rows=[];fig,axes=plt.subplots(1,3,figsize=(7.087,2.7),layout='constrained')
 for idx,name in enumerate(['calibration','fixed_null','combined','varied_baseline_null']):
  d=read(V2/f'ssm_{name}.json');rr=d['records'];rhat=[r['diagnostics']['max_rhat'] for r in rr];ess=[min(r['diagnostics']['min_ess_bulk'],r['diagnostics']['min_ess_tail']) for r in rr]
  for ax,vals in zip(axes[:2],[rhat,ess]):ax.scatter(np.arange(len(vals))/max(1,len(vals)-1)*.5+idx-.25,vals,s=5,alpha=.6,color=C['state_space'])
  elapsed=[r.get('total_seconds',r.get('elapsed_seconds',float('nan'))) for r in rr];axes[2].scatter(np.arange(len(rr))/max(1,len(rr)-1)*.5+idx-.25,elapsed,s=5,alpha=.6,color=C['state_space'])
  for r,h,e,t in zip(rr,rhat,ess,elapsed):rows.append({'scenario':name,'max_rhat':h,'min_ess':e,'seconds':t,'converged':r['diagnostics']['converged']})
 for ax,l in zip(axes,'abc'):ax.set(xticks=range(4),xticklabels=['Prior','Null','Mix','Varied\nnull']);panel(ax,l)
 axes[0].axhline(1.01,color='.4',ls='--');axes[0].set_ylabel('Maximum rank R-hat');axes[1].axhline(400,color='.4',ls='--');axes[1].set_ylabel('Minimum effective sample size');axes[2].set_ylabel('Fit wall time (s)');save(fig,'figS7_numerical_diagnostics',rows,True)

def mechanistic_supplements():
 d=read(V1/'timestep_sensitivity.json');rows=[];fig,axes=plt.subplots(2,3,figsize=(7.087,4.5),layout='constrained')
 labels=['Mean squared step (µm²)','Mean turning cosine','Overlap fraction']
 for row,case in enumerate(['combined','mechanical_only']):
  for steps,v in d['records'][case].items():
   for j in (3,4):rows.append({'case':case,'metric':d['metric_names'][j],'steps':int(steps),'n':v['n'],'mean':v['mean'][j],'mc_standard_error':v['se'][j]})
  for j,ax in enumerate(axes[row]):
   for i,(steps,v) in enumerate(d['records'][case].items()):
    ax.errorbar(int(steps),v['mean'][j],yerr=1.96*v['se'][j],fmt='o',ms=4,color=C['neural'] if row==0 else C['abc'],capsize=2)
    rows.append({'case':case,'metric':d['metric_names'][j],'steps':int(steps),'n':v['n'],'mean':v['mean'][j],'mc_standard_error':v['se'][j]})
   ax.set(xscale='log',xticks=[1,2,4,8,16],xticklabels=['1','2','4','8','16'],xlabel='Internal steps / frame',ylabel=labels[j],title='Combined response' if row==0 else 'Mechanical only');panel(ax,'abcdef'[row*3+j])
 save(fig,'figS5_timestep',rows,True)
 fig,axes=plt.subplots(1,2,figsize=(7.087,2.8),layout='constrained');rows=[];dt=.25;obs=.15
 colors=['#0F4D92','#B37900','#767676'];labels=['Active OU','Brownian jitter','Localization']
 for i,(name,S,tau,J) in enumerate([('Far',3.2,1.6,.7),('Contact',3.2*np.exp(-.8),1.6*np.exp(-.8),.7*np.exp(.8))]):
  components=[2*S*S*tau*(dt-tau*(-np.expm1(-dt/tau))),J*J*dt,2*obs*obs];bottom=0
  for j,val in enumerate(components):axes[0].bar(i,val,bottom=bottom,color=colors[j],edgecolor='white',lw=1,width=.55,label=labels[j] if i==0 else None);bottom+=val;rows.append({'panel':'a','regime':name,'component':labels[j],'variance_um2':val,'dt_min':dt})
  h=np.geomspace(.02,3,150);var=2*S*S*tau*(h-tau*(-np.expm1(-h/tau)))+J*J*h+2*obs*obs;fraction=J*J*h/var
  axes[1].semilogx(h,fraction,'-' if i==0 else '--',color=C['neural'] if i==0 else C['abc'],label=name)
  for h0,f0 in zip(h,fraction):rows.append({'panel':'b','regime':name,'dt_min':h0,'brownian_fraction':f0})
 axes[0].set(xticks=[0,1],xticklabels=['Far','Contact'],ylabel='Per-axis displacement variance (µm²)');fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncols=3,fontsize=7)
 axes[1].axvline(.25,ls=':',color='.5');axes[1].set(xlabel='Observation interval (min)',ylabel='Brownian fraction of variance',ylim=(0,1));axes[1].legend(fontsize=7)
 for ax,l in zip(axes,'ab'):panel(ax,l)
 save(fig,'figS8_local_variance',rows,True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--available',action='store_true');args=p.parse_args()
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':7.5,'axes.titlesize':8,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':1.,'lines.linewidth':1.,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
 fig1();fig5();mechanistic_supplements()
 if not args.available:fig2();fig3();fig4();supplementary()
 (MAIN/'figure_manifest.json').write_text(json.dumps(MANIFEST,indent=2))
 print('Rendered',len(MANIFEST),'figures with CSV source data')
if __name__=='__main__':main()
