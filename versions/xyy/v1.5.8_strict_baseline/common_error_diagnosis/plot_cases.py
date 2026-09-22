"""Static research figures using bundled ReportLab and Sharp; all writes local."""
from pathlib import Path
import os,subprocess
import pandas as pd
from reportlab.graphics.shapes import Drawing,String,Rect,Line
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor,white

OUT=Path(__file__).resolve().parent
NODE='/home/jacklo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
SHARP='/home/jacklo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp'
COLORS=['#152c42','#d37519','#3679bf','#b04c91']

def panel(d,x,y,w,h,g,cols,labels,title):
    plot=LinePlot();plot.x=x;plot.y=y;plot.width=w;plot.height=h
    values=[[(float(a),float(b)) for a,b in zip(g.target_hour,g[c]) if pd.notna(b)] for c in cols]
    plot.data=values
    xx=[a for data in values for a,b in data];yy=[b for data in values for a,b in data]
    plot.xValueAxis.valueMin=min(xx);plot.xValueAxis.valueMax=max(xx)
    plot.xValueAxis.valueSteps=[v for v in [48,60,71,72,96,120,144,168,192,215] if min(xx)<=v<=max(xx)]
    low=min(0,min(yy));high=max(yy);plot.yValueAxis.valueMin=low;plot.yValueAxis.valueMax=high+max(high*.07,.001)
    plot.yValueAxis.labelTextFormat=lambda z:f'{z:.3f}' if high<1 else f'{z:.1f}'
    plot.yValueAxis.visibleGrid=True;plot.yValueAxis.gridStrokeColor=HexColor('#e3e8ec')
    for ax in [plot.xValueAxis,plot.yValueAxis]:ax.labels.fontName='Helvetica';ax.labels.fontSize=8;ax.strokeColor=HexColor('#98a7b3')
    for i in range(len(values)):
        plot.lines[i].strokeColor=HexColor(COLORS[i]);plot.lines[i].strokeWidth=1.7
        if i%2:plot.lines[i].strokeDashArray=[4,2]
    d.add(plot);d.add(String(x,y+h+32,title,fontName='Helvetica-Bold',fontSize=12,fillColor=HexColor('#152c42')))
    for i,label in enumerate(labels):
        left=x+i*210
        d.add(Line(left,y+h+16,left+20,y+h+16,strokeColor=HexColor(COLORS[i]),strokeWidth=2))
        d.add(String(left+26,y+h+13,label,fontName='Helvetica',fontSize=9,fillColor=HexColor('#334155')))

def save(d,path,env):
    assert path.resolve().is_relative_to(OUT)
    renderSVG.drawToFile(d,str(path))
    js='const sharp=require(process.argv[1]); sharp(process.argv[2],{density:105}).png().toFile(process.argv[3]).catch(e=>{console.error(e);process.exit(1)});'
    subprocess.run([NODE,'-e',js,SHARP,str(path),str(path.with_suffix('.png'))],check=True,env=env)

def main():
    runtime=OUT/'figures_runtime';cache=runtime/'font-cache';cache.mkdir(parents=True,exist_ok=True)
    (runtime/'fonts.conf').write_text(f'<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd"><fontconfig><dir>/usr/share/fonts</dir><dir>/usr/local/share/fonts</dir><cachedir>{cache}</cachedir><include ignore_missing="yes">/etc/fonts/conf.d</include></fontconfig>')
    env=dict(os.environ,FONTCONFIG_FILE=str(runtime/'fonts.conf'))
    cases=pd.read_csv(OUT/'figures/plot_cases.csv',dtype={'fipsCode':str})
    data=pd.read_csv(OUT/'figures/trajectory_data.csv',dtype={'fipsCode':str},float_precision='round_trip')
    for seed in [42,20260917,20260918]:
        for r in cases.itertuples():
            g=data[(data.cv_seed==seed)&(data.fipsCode==r.fipsCode)&(data.horizon==r.horizon)].sort_values('target_hour')
            d=Drawing(1050,1090);d.add(Rect(0,0,1050,1090,fillColor=white,strokeColor=None))
            d.add(String(75,1055,f'{g.countyName.iloc[0]}, {g.stateAbbr.iloc[0]} | {r.horizon}h | CV {seed}',fontName='Helvetica-Bold',fontSize=18,fillColor=HexColor('#152c42')))
            d.add(String(75,1031,'Frozen OOF diagnosis. Outage observations end at hour 71; future truth is shown for diagnosis only.',fontName='Helvetica',fontSize=11,fillColor=HexColor('#53677a')))
            cols=['truth','T','S'] if r.horizon>=24 else ['truth','T'];labels=['Official OSI','Current tree T','Stella L2'][:len(cols)]
            panel(d,75,795,895,167,g,cols,labels,'Final OSI: common error and timing')
            panel(d,75,551,895,166,g,['P_true','P_pred','D_true','D_pred'],['Official P','Tree P','Official D','Tree D'],'P and D: level and persistence')
            panel(d,75,307,895,166,g,['N_true','N_pred','R_true','R_pred'],['Official N','Tree N','Official R','Tree R'],'N and R: growth and recovery components')
            panel(d,75,63,895,166,g,['gust'],['Provided gust (mph)'],'Allowed weather context')
            d.add(String(355,19,'Target hour: 72=Mar14, 96=Mar15, 120=Mar16, 168=Mar18',fontName='Helvetica',fontSize=10,fillColor=HexColor('#53677a')))
            save(d,OUT/f'figures/split{seed}_{r.fipsCode}_h{r.horizon:02d}.svg',env)
        print('Figures finished',seed,flush=True)
    hist=pd.read_csv(OUT/'summary/focus_known_history.csv',dtype={'fipsCode':str},float_precision='round_trip')
    hist=hist[(hist.fipsCode=='42053')&(hist.target_hour>=48)].sort_values('target_hour')
    d=Drawing(1050,450);d.add(Rect(0,0,1050,450,fillColor=white,strokeColor=None))
    d.add(String(75,415,'Forest County: information already observed before the cutoff',fontName='Helvetica-Bold',fontSize=17,fillColor=HexColor('#152c42')))
    panel(d,75,70,895,260,hist,['P_t','N_t','R_t'],['Observed P','Official observed N','Official observed R'],'History ends at hour 71')
    save(d,OUT/'figures/Forest_known_history.svg',env)

if __name__=='__main__':main()
