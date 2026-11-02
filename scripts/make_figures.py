"""Draw all revised figures from frozen predictions and revision statistics."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

W=Path(__file__).resolve().parents[1]; F=W/'figures';F.mkdir(exist_ok=True)
S=json.loads((W/'results/revision_summary.json').read_text())
CORE=json.loads((W/'results/core_test_predictions.json').read_text())
AUDIT=json.loads((W/'results/coverage_audit_predictions.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12.5,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
BLUE,RED,TEAL,ORANGE='#235789','#b33a3a','#15796b','#bd6a12'
def save(fig,name):
    fig.savefig(F/(name+'.png'),dpi=1200);plt.close(fig)

fig,ax=plt.subplots(figsize=(8,6.5));ax.axis('off')
nodes=[(.05,.77,.9,.13,'CASI released rows  37,500\nLabel filtering  554 rows removed'),
(.05,.56,.9,.13,'Exact context isolation within abbreviation\nRemove 742 surplus rows from nonconflicting groups\nand 12 rows in 6 conflicting groups'),
(.05,.35,.43,.13,'Core benchmark\n207 classes  35,966 contexts'),
(.52,.35,.43,.13,'Candidate coverage audit\n141 rare classes  226 contexts'),
(.05,.04,.43,.24,'Training 25,192\nValidation 5,387   Test 5,387\nFit selectors on training\nFit temperature and gate\non validation (90% coverage)'),
(.52,.04,.43,.24,'Apply frozen gate\nCore accuracy and calibration\nAudit acceptance with\ncorrect candidate absent')]
for x,y,w,h,txt in nodes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',facecolor='#edf3f8',edgecolor=BLUE,linewidth=1.2))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=12.5)
for a,b in [((.5,.77),(.5,.70)),((.28,.56),(.28,.49)),((.73,.56),(.73,.49)),((.28,.35),(.28,.29)),((.73,.35),(.73,.29))]:
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':BLUE})
save(fig,'Figure_1')

fig,axes=plt.subplots(1,2,figsize=(8,4.9))
fig.subplots_adjust(left=.15,right=.98,top=.91,bottom=.31,wspace=.65)
parts=S['leakage_cluster_sensitivity']['reference'];labels=['All test\nrows','Exposed\nrows','Unexposed\nrows'];keys=['all','exposed','unexposed']
y=[parts[k]['difference_pp'] for k in keys];ci=np.array([parts[k]['cluster_ci_pp'] for k in keys])
axes[0].errorbar(y,range(3),xerr=[np.array(y)-ci[:,0],ci[:,1]-np.array(y)],fmt='o',capsize=4,color=BLUE)
axes[0].axvline(0,color='gray',lw=.8);axes[0].set_yticks(range(3),labels);axes[0].invert_yaxis();axes[0].set(xlim=(-.25,4.0),xlabel='Exposed minus purged\naccuracy (percentage points)');axes[0].set_title('a  Exact duplicate comparison',loc='left',fontsize=12.5)
for rs,label,color,ls in [([r for r in CORE if r['correct']],'Correct core (n = 5,162)',TEAL,'-'),([r for r in CORE if not r['correct']],'Incorrect core (n = 225)',ORANGE,'--'),(AUDIT,'Candidate absent (n = 226)',RED,'-')]:
    x=np.sort([r['scaled_confidence'] for r in rs]);axes[1].step(x,100*np.arange(1,len(x)+1)/len(x),where='post',label=label,color=color,ls=ls)
axes[1].axvline(S['threshold'],color='gray',ls=':',label='Frozen validation gate')
axes[1].set(xlim=(0,1.01),ylim=(0,101),xlabel='Calibrated confidence',ylabel='Cumulative proportion (%)');axes[1].set_title('b  Confidence by population',loc='left',fontsize=12.5)
handles,names=axes[1].get_legend_handles_labels();fig.legend(handles,names,loc='lower center',ncol=2,fontsize=12.5,frameon=False,bbox_to_anchor=(.53,-.015))
save(fig,'Figure_2')

fig,ax=plt.subplots(figsize=(6.5,4.5),layout='constrained');rs=sorted(CORE,key=lambda r:r['scaled_confidence'],reverse=True)
c=np.arange(1,len(rs)+1);risk=100*(1-np.cumsum([r['correct'] for r in rs])/c)
ax.plot(100*c/len(rs),risk,color=BLUE);g=S['gate']['all_core'];x=100*g['pass_rate'];y=100*(1-sum(r['correct'] for r in rs if r['passes'])/g['passing'])
ax.scatter([x],[y],color=RED,zorder=5);ax.axvline(x,color=RED,ls='--',lw=1);ax.annotate(f'{x:.2f}% coverage\n{y:.2f}% risk',xy=(x,y),xytext=(57,2.5),arrowprops={'arrowstyle':'->','color':RED},color=RED)
ax.set(xlabel='Retained core test items in percent',ylabel='Selective error in percent',xlim=(0,101),ylim=(0,6));save(fig,'Figure_3')

fig,axes=plt.subplots(1,2,figsize=(8,4.6),layout='constrained')
for ax,key,title in zip(axes,['all','multiple_candidates'],['a  All core test items  n = 5,387','b  Multiple candidates  n = 5,109']):
    ax.plot([0,1],[0,1],color='gray',ls=':',lw=1)
    for scale,color,label,marker in [('raw',ORANGE,'Before scaling','s'),('scaled',BLUE,'After scaling','o')]:
        data=[b for b in S['calibration'][key][scale]['bins'] if b['n']]
        x=[b['confidence'] for b in data];y=[b['accuracy'] for b in data]
        ax.plot(x,y,color=color,marker=marker,markersize=5,label=label)
    ax.set(xlim=(0,1.12),ylim=(0,1.10),xlabel='Mean confidence in bin',ylabel='Observed accuracy');ax.set_title(title,loc='left',fontsize=12.5);ax.legend(fontsize=12.5,loc='lower right')
save(fig,'Figure_4')

fig,ax=plt.subplots(figsize=(8,4.4),layout='constrained')
for i,s in enumerate(['2','3','4','5+']):
    p=S['calibration']['pool_'+s]
    ax.bar(i-.17,p['raw']['ece'],width=.33,color=ORANGE,label='Before scaling' if i==0 else None)
    ax.bar(i+.17,p['scaled']['ece'],width=.33,color=BLUE,label='After scaling' if i==0 else None)
ax.set_xticks(range(4),['2','3','4','At least 5']);ax.set(xlabel='Candidate pool size',ylabel='Expected calibration error');ax.legend();save(fig,'Figure_S1')
print('Wrote four manuscript figures and one supplementary figure')
