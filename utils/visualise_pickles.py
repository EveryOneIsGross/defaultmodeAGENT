#!/usr/bin/env python3
import argparse,sys,pickle,re
import numpy as np,pandas as pd,matplotlib.pyplot as plt,umap
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

'''ensure you install umap-learn via pip if not already:
    pip install umap-learn'''

STOP=set(("a,an,the,of,in,on,at,for,to,from,by,with,without,as,is,are,was,were,be,been,being,am,do,does,did,doing,have,has,had,having,not,no,nor,or,and,if,then,else,than,that,this,these,those,there,here,when,where,why,how,what,which,who,whom,whose,i,me,my,mine,im,i'm,ive,i’ve,you're,ur,u,we,us,our,ours,they,them,their,theirs,he,him,his,she,her,hers,it,its,you,your,yours,just,like,into,own,our,loop").replace("’","'").split(","))
_norm=re.compile(r"[’`]+");_ws=re.compile(r"\s+");_nonw=re.compile(r"[^a-z0-9' ]+")
def clean(t):t=_norm.sub("'",t.lower());t=_nonw.sub(" ",t);return _ws.sub(" ",t).strip()
def toks(t):t=clean(t);return[w for w in t.split(" ") if len(w)>=2 and w not in STOP]

def load(p):
    with open(p,"rb") as f:return pickle.load(f)

def df_from(d):
    mem=d["memories"];um=d["user_memories"];m2u=defaultdict(list)
    for u,ids in um.items():
        for j in set(ids):
            if 0<=j<len(mem) and isinstance(mem[j],str):m2u[j].append(str(u))
    act=[(i,m) for i,m in enumerate(mem) if isinstance(m,str)]
    df=pd.DataFrame({"memory_id":[i for i,_ in act],"text":[m for _,m in act]})
    df["users"]=df["memory_id"].map(lambda i:"|".join(m2u.get(i,[])))
    df["label"]=df["users"].map(lambda s:(s.split("|")[0] if s else "∅"))
    df["clean"]=df["text"].map(clean);df["words"]=df["clean"].str.split().apply(len)
    return df

def embed(texts,an,min_df,max_feat,metric,neighbors,seed,dim):
    v=CountVectorizer(analyzer=an,min_df=min_df,max_features=max_feat)
    X=v.fit_transform(texts)
    return umap.UMAP(n_neighbors=neighbors,metric=metric,random_state=seed,n_components=dim).fit_transform(X)

def plot(df,emb,alpha,pmin,pmax,pickr,elev,azim):
    k=emb.shape[1]; fig=plt.figure(figsize=(11,9))
    labs=sorted(df["label"].unique());cmap=plt.cm.get_cmap("tab20",max(20,len(labs)))
    sz=np.clip(2*np.sqrt(df["words"].to_numpy()+1),pmin,pmax);arts={}
    if k==3:
        ax=fig.add_subplot(111,projection="3d"); ax.view_init(elev=elev,azim=azim)
        for i,l in enumerate(labs):
            idx=df.index[df["label"]==l].to_numpy()
            a=ax.scatter(emb[idx,0],emb[idx,1],emb[idx,2],s=sz[idx],c=[cmap(i)],alpha=alpha,label=l,depthshade=False,picker=pickr);arts[a]=idx
        ax.set_title("UMAP·TF (3D)"); ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
    else:
        ax=fig.add_subplot(111)
        for i,l in enumerate(labs):
            idx=df.index[df["label"]==l].to_numpy()
            a=ax.scatter(emb[idx,0],emb[idx,1],s=sz[idx],c=[cmap(i)],alpha=alpha,label=l,edgecolors="none",picker=pickr);arts[a]=idx
        ax.set_title("UMAP·TF (2D)"); ax.set_xticks([]);ax.set_yticks([])
    ax.legend(loc="best",fontsize=8,markerscale=0.7,frameon=False)
    def pick(e):
        A=e.artist
        if A in arts and len(e.ind):
            i=int(arts[A][e.ind[0]]);r=df.loc[i]
            sys.stdout.write(f"\n[{r.memory_id}] {r.label} :: {r.text.replace('\\n',' ')}\n");sys.stdout.flush()
    fig.canvas.mpl_connect("pick_event",pick)
    plt.tight_layout();plt.show()

def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument("pickle_path")
    ap.add_argument("--dim",type=int,choices=[2,3],default=2)
    ap.add_argument("--min-df",type=int,default=5)
    ap.add_argument("--max-features",type=int,default=60000)
    ap.add_argument("--metric",default="cosine")
    ap.add_argument("--neighbors",type=int,default=20)
    ap.add_argument("--alpha",type=float,default=0.8)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--point-min",type=float,default=5)
    ap.add_argument("--point-max",type=float,default=60)
    ap.add_argument("--pick-radius",type=float,default=9)
    ap.add_argument("--elev",type=float,default=20)
    ap.add_argument("--azim",type=float,default=-60)
    a=ap.parse_args()
    d=load(a.pickle_path); df=df_from(d)
    emb=embed(df["text"],toks,a.min_df,a.max_features,a.metric,a.neighbors,a.seed,a.dim)
    plot(df,emb,a.alpha,a.point_min,a.point_max,a.pick_radius,a.elev,a.azim)

if __name__=="__main__":main()
