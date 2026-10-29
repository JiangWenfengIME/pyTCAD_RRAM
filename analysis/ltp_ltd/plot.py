"""Plot genuine independent-read data, with no pulse-zero point or interpolation."""
import argparse,csv,json,warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from experiments.ltp_ltd.run import ROOT,settings
from experiments.ltp_ltd.common import digest

def plot_curves(conditions,output,direction):
 s=settings(direction);output=Path(output);output.mkdir(parents=True,exist_ok=True)
 try:
  import scienceplots
  plt.style.use(['science','ieee']);style='science+ieee'
 except ImportError:
  styles=Path(__file__).with_name('styles')
  plt.style.use([str(styles/'science.mplstyle'),str(styles/'ieee.mplstyle')]);style='science+ieee (bundled SciencePlots 2.2.2 styles)'
 plt.rcParams.update({'text.usetex':False,'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':9,'axes.labelsize':10,'xtick.labelsize':8,'ytick.labelsize':8,'xtick.direction':'in','ytick.direction':'in'})
 fig,ax=plt.subplots(figsize=(3.5,3));sources=[];data=[]
 for folder in conditions:
  folder=Path(folder);source=folder/'values.csv';v=json.loads((folder/'config.json').read_text())['hold_voltage_V']
  with source.open() as f:rows=list(csv.DictReader(f))
  if not rows:raise ValueError(f'No data: {source}')
  for n,r in enumerate(rows,1):
   g=float(r['conductance_S']);rv=float(r['read_voltage_V'])
   if int(r['pulse'])!=n or rv!=s['read_voltage_V'] or not np.isfinite(g) or not np.isclose(g,float(r['read_current_A'])/rv,rtol=1e-12,atol=0):raise ValueError('Invalid read data')
  ax.plot([int(r['pulse']) for r in rows],[float(r['conductance_S'])*1e6 for r in rows],color=s['colors'][str(float(v))],label=f'{v} V',linestyle='-',marker=None,linewidth=2,alpha=.8)
  sources.append(dict(path=str(source),sha256=digest(source),voltage_V=v,pulses=len(rows)))
  data.extend(dict(hold_voltage_V=v,**r) for r in rows)
 ax.set_xlabel('Pulse number');ax.set_ylabel('Read conductance (μS)');ax.set_xlim(0,500)
 ax.set_ylim(0,max(float(r['conductance_S']) for r in data)*1.06e6)
 ax.xaxis.set_minor_locator(NullLocator());ax.yaxis.set_minor_locator(NullLocator());ax.tick_params(which='major',width=1.2);ax.grid(False)
 for spine in ax.spines.values():spine.set_linewidth(1.2)
 ax.legend(loc='best',ncol=2,fontsize=7,frameon=False,handlelength=1.5,labelspacing=.2)
 fig.tight_layout();stem=f'read_{direction}_full500'
 for ext in ['png','tiff','pdf']:fig.savefig(output/(stem+'.'+ext),dpi=300,bbox_inches='tight')
 plt.close(fig)
 with (output/(stem+'.csv')).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 (output/(stem+'.json')).write_text(json.dumps(dict(sources=sources,colors=s['colors'],style=style,initial_point_included=False,processing='conductance_S * 1e6 only',excluded_voltage_V=s['excluded_voltage_V']),indent=2))
 return output/(stem+'.png')

def main(direction=None):
 p=argparse.ArgumentParser(description=__doc__)
 if direction is None:p.add_argument('direction',choices=['LTP','LTD'])
 p.add_argument('--input',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();direction=direction or a.direction;s=settings(direction)
 root=a.input or ROOT/'reference_results/ltp_ltd'/direction
 output=a.output or ROOT/'results_ltp_ltd/reference_figures'/direction
 plot_curves([root/f'{v:g}V' for v in s['voltages_V']],output,direction)
if __name__=='__main__':main()
