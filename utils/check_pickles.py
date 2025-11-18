#!/usr/bin/env python3
import argparse,sys,pickle,re,hashlib,math
import pandas as pd
from collections import defaultdict,Counter

STOP=set(("a,an,the,of,in,on,at,for,to,from,by,with,without,as,is,are,was,were,be,been,being,am,do,does,did,doing,have,has,had,having,not,no,nor,or,and,if,then,else,than,that,this,these,those,there,here,when,where,why,how,what,which,who,whom,whose,i,me,my,mine,im,i'm,ive,i’ve,you're,ur,u,we,us,our,ours,they,them,their,theirs,he,him,his,she,her,hers,it,its,you,your,yours,just,like,into,own,our,loop").replace("’","'").split(","))

_norm=re.compile(r"[’`]+"); _ws=re.compile(r"\s+"); _nonw=re.compile(r"[^a-z0-9' ]+")
def clean(t): t=_norm.sub("'",t.lower()); t=_nonw.sub(" ",t); return _ws.sub(" ",t).strip()
def toks(t): t=clean(t); return [w for w in t.split(" ") if len(w)>=2 and w not in STOP]

def load(path): 
    with open(path,"rb") as f: return pickle.load(f)

def validate(d):
    ok=True; errs=[]; inv=d.get("inverted_index",{}); mem=d.get("memories",[]); um=d.get("user_memories",{})
    if not isinstance(d,dict): return {"ok":False,"errors":["root_not_dict"]}
    for k in ("inverted_index","memories","user_memories"):
        if k not in d: ok=False; errs.append(f"missing_key:{k}")
    if not isinstance(inv,dict): ok=False; errs.append("inverted_index_not_dict")
    if not isinstance(mem,list): ok=False; errs.append("memories_not_list")
    if not isinstance(um,dict): ok=False; errs.append("user_memories_not_dict")
    n=len(mem); bad_key=bad_postlist=bad_postint=oob=nullref=0; dup_terms=unsorted_terms=0; postings=uniq_postings=0
    if isinstance(inv,dict):
        for w,ids in inv.items():
            if not isinstance(w,str): bad_key+=1
            if not isinstance(ids,list): bad_postlist+=1; continue
            postings+=len(ids); sids=set(); last=-1; sorted_flag=True; dupped=False
            for j in ids:
                if not isinstance(j,int): bad_postint+=1; continue
                if j<0 or j>=n: oob+=1
                elif mem[j] is None: nullref+=1
                if j in sids: dupped=True
                else: sids.add(j)
                if j<last: sorted_flag=False
                last=j
            uniq_postings+=len(sids)
            if dupped: dup_terms+=1
            if not sorted_flag: unsorted_terms+=1
    return {
        "ok": ok and bad_key==bad_postlist==bad_postint==oob==nullref==0,
        "errors": errs,
        "index_stats":{
            "terms": int(len(inv) if isinstance(inv,dict) else 0),
            "postings_total": int(postings),
            "postings_unique": int(uniq_postings),
            "terms_with_duplicates": int(dup_terms),
            "terms_unsorted": int(unsorted_terms),
            "bad_keys": int(bad_key),
            "bad_postings_list": int(bad_postlist),
            "bad_posting_int": int(bad_postint),
            "out_of_bounds": int(oob),
            "null_refs": int(nullref)
        }
    }

def analyze(d):
    inv=d["inverted_index"]; mem=d["memories"]; um=d["user_memories"]
    act=[(i,m) for i,m in enumerate(mem) if m is not None]
    m2u=defaultdict(list)
    for u,ids in um.items():
        s=set()
        for j in ids:
            if 0<=j<len(mem) and mem[j] is not None and j not in s: m2u[j].append(str(u)); s.add(j)
    df=pd.DataFrame({"memory_id":[i for i,_ in act],"text":[m for _,m in act]})
    df["clean"]=df["text"].map(clean); df["tokens"]=df["text"].map(toks)
    df["words"]=df["tokens"].apply(len); df["chars"]=df["clean"].str.len()
    df["users"]=df["memory_id"].map(lambda i:"|".join(m2u.get(i,[]))); df["n_users"]=df["users"].apply(lambda s:0 if not s else s.count("|")+1)
    user_counts=Counter()
    for u,ids in um.items():
        c=0
        for j in set(ids):
            if 0<=j<len(mem) and mem[j] is not None: c+=1
        user_counts[str(u)]=c
    udf=pd.DataFrame([{"user_id":u,"count":c,"pct":(100.0*c/len(df)) if len(df) else 0.0} for u,c in user_counts.items()]).sort_values(["count","user_id"],ascending=[False,True]).reset_index(drop=True)
    tf=Counter(); dfreq=Counter()
    for tt in df["tokens"]: tf.update(tt); dfreq.update(set(tt))
    stats={
        "total_memories": int(len(mem)),
        "active_memories": int(len(df)),
        "total_users": int(len(um)),
        "vocab_size": int(len(dfreq)),
        "avg_tokens_per_doc": float(df["words"].mean() if len(df) else 0.0),
        "median_tokens_per_doc": int(df["words"].median() if len(df) else 0),
        "hash": hashlib.sha1(("|".join(df["clean"].tolist())).encode()).hexdigest()
    }
    return df,udf,stats

def text_report(val,udf,stats,head=10):
    lines=[]
    lines.append("== SUMMARY ==")
    lines.append(f"memories_total:{stats['total_memories']} active:{stats['active_memories']} users:{stats['total_users']} vocab:{stats['vocab_size']}")
    lines.append(f"tokens_avg:{stats['avg_tokens_per_doc']:.2f} tokens_median:{stats['median_tokens_per_doc']}")
    lines.append(f"hash:{stats['hash']}")
    i=val["index_stats"]; intact=("YES" if (val["ok"] and i["out_of_bounds"]==0 and i["null_refs"]==0) else "NO")
    lines.append("== INVERTED_INDEX ==")
    lines.append(f"terms:{i['terms']} postings_total:{i['postings_total']} postings_unique:{i['postings_unique']} dup_terms:{i['terms_with_duplicates']} unsorted_terms:{i['terms_unsorted']} oob:{i['out_of_bounds']} null_refs:{i['null_refs']} intact:{intact}")
    if val["errors"]: lines.append("errors:"+",".join(val["errors"]))
    lines.append("== USERS (ALL) ==")
    lines.append("user_id,count,pct")
    for _,r in udf.iterrows(): lines.append(f"{r['user_id']},{int(r['count'])},{r['pct']:.2f}")
    return "\n".join(lines)+"\n"

def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument("pickle_path")
    ap.add_argument("--json","-j")
    ap.add_argument("--csv","-c")
    ap.add_argument("--userscsv","-u")
    ap.add_argument("--head","-H",type=int,default=10)
    args=ap.parse_args()
    d=load(args.pickle_path)
    val=validate(d)
    df,udf,stats=analyze(d)
    if args.json:
        import json
        out={"validate":val,"summary":stats,"users":udf.to_dict(orient="records")}
        s=json.dumps(out,ensure_ascii=False,separators=(",",":"))
        if args.json=="-": sys.stdout.write(s)
        else:
            with open(args.json,"w",encoding="utf-8") as w: w.write(s)
    else:
        sys.stdout.write(text_report(val,udf,stats,args.head))
    if args.csv: 
        df[["memory_id","text","words","chars","users","n_users"]].to_csv(args.csv,index=False)
    if args.userscsv:
        udf.to_csv(args.userscsv,index=False)

if __name__=="__main__": main()
