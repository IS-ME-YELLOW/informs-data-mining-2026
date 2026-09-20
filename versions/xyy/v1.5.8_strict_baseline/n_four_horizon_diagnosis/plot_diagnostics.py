"""Static scientific charts using the bundled ReportLab charts and Sharp runtimes."""
from pathlib import Path
import os
import subprocess
import pandas as pd
from reportlab.graphics.shapes import Drawing, String, Line, Rect
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderSVG
from reportlab.lib.colors import HexColor, white

OUT=Path(__file__).resolve().parent
NODE='/home/jacklo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
SHARP='/home/jacklo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp'
COLORS=['#0f172a','#d97706','#2563eb','#059669']


def panel(d,x,y,width,height,data,labels,title):
    plot=LinePlot();plot.x=x;plot.y=y;plot.width=width;plot.height=height
    plot.data=[[(float(a),float(b)) for a,b in values] for values in data]
    xx=[a for values in data for a,b in values];yy=[b for values in data for a,b in values]
    plot.xValueAxis.valueMin=min(xx);plot.xValueAxis.valueMax=max(xx)
    if max(xx)<=4:
        plot.xValueAxis.valueSteps=[1,2,3,4]
        plot.xValueAxis.labelTextFormat=lambda z:{1:'1h',2:'6h',3:'24h',4:'48h'}.get(int(z),'')
    else:
        plot.xValueAxis.valueSteps=[v for v in (72,96,120,144,168,192,215) if min(xx)<=v<=max(xx)]
    low=min(0,min(yy));high=max(yy);pad=max((high-low)*.06,.0001)
    plot.yValueAxis.valueMin=low-pad if low<0 else 0
    plot.yValueAxis.valueMax=high+pad
    plot.xValueAxis.labels.fontName='Helvetica';plot.yValueAxis.labels.fontName='Helvetica'
    plot.xValueAxis.labels.fontSize=8;plot.yValueAxis.labels.fontSize=8
    plot.yValueAxis.labelTextFormat=lambda z:f'{z:.3f}' if abs(high)<1 else f'{z:.1f}'
    plot.xValueAxis.strokeColor=HexColor('#94a3b8');plot.yValueAxis.strokeColor=HexColor('#94a3b8')
    plot.yValueAxis.visibleGrid=True;plot.yValueAxis.gridStrokeColor=HexColor('#e2e8f0')
    for i in range(len(data)):
        plot.lines[i].strokeColor=HexColor(COLORS[i]);plot.lines[i].strokeWidth=1.5
        if i==1:plot.lines[i].strokeDashArray=[4,2]
    d.add(plot);d.add(String(x,y+height+30,title,fontName='Helvetica-Bold',fontSize=11,fillColor=HexColor('#0f172a')))
    for i,label in enumerate(labels):
        lx=x+i*205;ly=y+height+13
        d.add(Line(lx,ly,lx+20,ly,strokeColor=HexColor(COLORS[i]),strokeWidth=2))
        d.add(String(lx+26,ly-3,label,fontName='Helvetica',fontSize=9,fillColor=HexColor('#334155')))


def save(d,path):
    assert path.resolve().is_relative_to(OUT)
    path.parent.mkdir(parents=True,exist_ok=True)
    renderSVG.drawToFile(d,str(path))
    js='const sharp=require(process.argv[1]); sharp(process.argv[2],{density:120}).png().toFile(process.argv[3]).catch(e=>{console.error(e);process.exit(1)});'
    render_env=dict(os.environ,FONTCONFIG_FILE=str(OUT/'figures_runtime/fonts.conf'))
    subprocess.run([NODE,'-e',js,SHARP,str(path),str(path.with_suffix('.png'))],check=True,env=render_env)


def main():
    runtime=OUT/'figures_runtime';cache=runtime/'font-cache';cache.mkdir(parents=True,exist_ok=True)
    (runtime/'fonts.conf').write_text(f'<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd"><fontconfig><dir>/usr/share/fonts</dir><dir>/usr/local/share/fonts</dir><cachedir>{cache}</cachedir><include ignore_missing="yes">/etc/fonts/conf.d</include></fontconfig>')
    selection=pd.read_csv(OUT/'summary/case_selection.csv',dtype={'fipsCode':str})
    selected=set(selection.loc[(selection.reason=='seed42_top_N_weighted_sse')&(selection['rank']==1),'fipsCode'])|{'42053','39117'}
    for seed in [42,20260917,20260918]:
        data=pd.read_csv(OUT/f'runs/split{seed}/tables/case_trajectories.csv',dtype={'fipsCode':str},float_precision='round_trip')
        for f in sorted(selected):
            h=48 if f=='39115' else 1
            g=data.loc[(data.fipsCode==f)&(data.horizon==h)].sort_values('target_hour');x=g.target_hour.to_numpy()
            d=Drawing(1050,970)
            d.add(Rect(0,0,1050,970,fillColor=white,strokeColor=None))
            title=f'{g.countyName.iloc[0]}, {g.stateAbbr.iloc[0]} | split {seed} | {h}h'
            d.add(String(72,939,title,fontName='Helvetica-Bold',fontSize=18,fillColor=HexColor('#0f172a')))
            d.add(String(72,917,'Fixed F2 reference. Diagnostic curves; no model refitting.',fontName='Helvetica',fontSize=11,fillColor=HexColor('#475569')))
            panel(d,72,705,900,155,[list(zip(x,g[c])) for c in ['N_true','N_raw','N_aligned']],
                  ['Official N','Raw source N','C3 aligned N'],'N: amplitude and timing')
            panel(d,72,480,900,155,[list(zip(x,g[c])) for c in ['P_true','P_pred','R_true','R_pred']],
                  ['Official P','Current P','Official R','Current R'],'P and R context (current main-rule components)')
            panel(d,72,255,900,155,[list(zip(x,g[c])) for c in ['OSI_true','OSI_pred']],
                  ['Official OSI','Current OSI'],'Final OSI')
            panel(d,72,45,900,140,[list(zip(x,g.gust_target))],['Provided target-hour gust'],'Allowed weather context')
            d.add(String(485,12,'Target hour s (history ends at 71)',fontName='Helvetica',fontSize=9,fillColor=HexColor('#475569')))
            if h>=24:
                d.add(String(72,904,'24/48h main OSI uses the same-horizon source; C3 is shown for diagnosis.',fontName='Helvetica',fontSize=10,fillColor=HexColor('#475569')))
            save(d,OUT/f'runs/split{seed}/figures/county_{f}_h{h:02d}.svg')
    s=pd.read_csv(OUT/'summary/four_horizon_summary.csv',float_precision='round_trip')
    d=Drawing(1050,550)
    d.add(Rect(0,0,1050,550,fillColor=white,strokeColor=None))
    d.add(String(72,519,'N oracle and fixed-anchor sensitivity across three CV splits',fontName='Helvetica-Bold',fontSize=17,fillColor=HexColor('#0f172a')))
    d.add(String(72,496,'Relative OSI RMSE change (%). Negative is lower error. Oracle uses future truth and is not deployable.',fontName='Helvetica',fontSize=10,fillColor=HexColor('#475569')))
    # Categorical horizon positions avoid treating 1/6/24/48 as uniform time steps.
    for k,(col,title) in enumerate([('O_N_all_change_pct','Replace all N sources by true N'),('A_N_zero_change_pct','Fixed N = 0 anchor')]):
        xx=[1,2,3,4];data=[]
        for seed in [42,20260917,20260918]:
            g=s[s.split_seed==seed].set_index('horizon').loc[[1,6,24,48]]
            data.append(list(zip(xx,g[col])))
        panel(d,72,280-k*225,900,145,data,['seed42','seed20260917','seed20260918'],title)
    d.add(String(470,15,'Forecast horizon',fontName='Helvetica',fontSize=10))
    save(d,OUT/'summary/figures/N_sensitivity.svg')


if __name__=='__main__':main()
