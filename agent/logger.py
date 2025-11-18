import os, json, sqlite3, logging, asyncio, concurrent.futures
from datetime import datetime
from typing import Dict, Any, Set, List
from collections import defaultdict
from colorama import Fore, Back, Style, init
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    COLORS={'DEBUG':Fore.BLUE,'INFO':Fore.LIGHTGREEN_EX,'WARNING':Fore.YELLOW,'ERROR':Fore.RED,'CRITICAL':Fore.RED+Back.WHITE}
    def format(self,r):
        c=self.COLORS.get(r.levelname,''); r.msg=f"{c}{r.msg}{Style.RESET_ALL}"; return super().format(r)

class BotLogger:
    _initialized=False; _schema_cache=defaultdict(set); _config=None
    COMMON_FIELDS={'id':'INTEGER PRIMARY KEY AUTOINCREMENT','timestamp':'TEXT NOT NULL','user_id':'TEXT','user_name':'TEXT','channel':'TEXT','data':'JSON','created_at':'TIMESTAMP DEFAULT CURRENT_TIMESTAMP','event_type':'TEXT NOT NULL'}
    TABLE_SCHEMAS={'error':{'error_type':'TEXT','error_message':'TEXT','stack_trace':'TEXT'},'dmn':{'thought_type':'TEXT','seed_memory':'TEXT','generated_thought':'TEXT'},'memory':{'memory_id':'TEXT','memory_text':'TEXT','operation':'TEXT'}}
    SQL_TYPE_MAP={str:'TEXT',int:'INTEGER',float:'REAL',bool:'INTEGER',dict:'JSON',list:'JSON',type(None):'TEXT'}

    @classmethod
    def setup_global_logging(cls,level:str=None,format:str=None):
        if cls._initialized:return
        from bot_config import config
        cls._config=config
        level=level or config.logging.log_level
        format=format or config.logging.log_format
        root=logging.getLogger()
        for h in list(root.handlers): root.removeHandler(h)
        if config.logging.enable_console:
            h=logging.StreamHandler(); h.setFormatter(ColoredFormatter(format)); root.addHandler(h)
        root.setLevel(level); cls._initialized=True

    def __new__(cls,bot_id:str=None):
        bot_id=bot_id or"default"
        if not hasattr(cls,'_instances'): cls._instances={}
        if bot_id not in cls._instances: cls._instances[bot_id]=super().__new__(cls)
        return cls._instances[bot_id]

    def __init__(self,bot_id:str=None):
        if hasattr(self,'initialized'): return
        from bot_config import config
        self._config=config
        self.bot_id=bot_id or"default"
        self.executor=concurrent.futures.ThreadPoolExecutor(max_workers=2)
        root_dir = self._config.logging.base_log_dir or 'logs'
        self.log_dir = os.path.join(root_dir, self.bot_id, 'logs'); os.makedirs(self.log_dir, exist_ok=True)
        self.db_path=os.path.join(self.log_dir,self._config.logging.db_pattern.format(bot_id=self.bot_id))
        self.jsonl_path=os.path.join(self.log_dir,self._config.logging.jsonl_pattern.format(bot_id=self.bot_id))
        self._logger=logging.getLogger(f'bot.{self.bot_id}'); self._logger.propagate=False
        if not self._logger.handlers and self._config.logging.enable_console:
            h=logging.StreamHandler(); h.setFormatter(ColoredFormatter(self._config.logging.log_format)); self._logger.addHandler(h)
        self._logger.setLevel(self._config.logging.log_level)
        if self._config.logging.enable_sql: self._init_db()
        self.initialized=True

    def _get_sql_type(self,v:Any)->str: return self.SQL_TYPE_MAP.get(type(v),'TEXT')

    def _init_db(self)->None:
        with sqlite3.connect(self.db_path) as c:
            c.execute('CREATE TABLE IF NOT EXISTS schema_version (id INTEGER PRIMARY KEY AUTOINCREMENT,table_name TEXT,version INTEGER,updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')

    def _get_table_name(self,event_type:str)->str:
        if not event_type: return 'events'
        return f"{event_type.lower().split('_')[0]}_events"

    def _ensure_table_exists(self,table_name:str,event_type:str)->None:
        cat=event_type.lower().split('_')[0]
        cols={**self.COMMON_FIELDS,**self.TABLE_SCHEMAS.get(cat,{})}
        with sqlite3.connect(self.db_path) as conn:
            cur=conn.cursor(); cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            if not cur.fetchone():
                sql=', '.join(f'{k} {v}' for k,v in cols.items())
                conn.execute(f'CREATE TABLE IF NOT EXISTS {table_name} ({sql})')
                conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp)')
                conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_et ON {table_name}(event_type)')
                conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_uid ON {table_name}(user_id)')
                self._schema_cache[table_name].update(cols.keys())

    def _sync_log(self,data:Dict[Any,Any])->None:
        data.setdefault('timestamp',datetime.now().isoformat())
        if self._config.logging.enable_jsonl:
            with open(self.jsonl_path,'a',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False); f.write('\n')
        if self._config.logging.enable_sql:
            et=data.get('event','').lower(); tn=self._get_table_name(et)
            self._ensure_table_exists(tn,et)
            vals={'timestamp':data.get('timestamp'),'user_id':data.get('user_id'),'user_name':data.get('user_name'),'channel':data.get('channel'),'data':json.dumps(data,ensure_ascii=False),'event_type':et}
            for k,v in data.items():
                if k not in vals and k!='event':
                    vals[k]=v if not isinstance(v,(dict,list)) else json.dumps(v,ensure_ascii=False)
            with sqlite3.connect(self.db_path) as c:
                cur=c.cursor(); cur.execute(f"PRAGMA table_info({tn})"); existing={r[1] for r in cur.fetchall()}
                for k,v in vals.items():
                    if k not in existing and k not in self.COMMON_FIELDS:
                        try: c.execute(f"ALTER TABLE {tn} ADD COLUMN {k} {self._get_sql_type(v)}")
                        except sqlite3.OperationalError as e:
                            if "duplicate column name" not in str(e): raise
                ph=', '.join(['?' for _ in vals]); cols=', '.join(vals.keys())
                c.execute(f'INSERT INTO {tn} ({cols}) VALUES ({ph})', list(vals.values()))

    async def _async_log(self,data:Dict[Any,Any])->None:
        loop=asyncio.get_running_loop()
        await loop.run_in_executor(self.executor,self._sync_log,data)

    def _schedule(self,data:Dict[Any,Any])->None:
        try:
            asyncio.get_running_loop().create_task(self._async_log(data))
        except RuntimeError:
            import threading; threading.Thread(target=self._sync_log,args=(data,),daemon=True).start()

    def _emit(self,level:str,msg:str)->None:
        if self._config.logging.enable_console: self._logger.log(getattr(logging,level),msg)
        if self._config.logging.enable_jsonl or self._config.logging.enable_sql:
            self._schedule({'event':level.lower(),'message':msg,'level':level})

    # public API
    def log(self,data:Dict[Any,Any],*,sync:bool=False)->None:
        if sync: self._sync_log(data); return
        self._schedule(data)
    def debug(self,msg:str)->None: self._emit('DEBUG',msg)
    def info(self,msg:str)->None: self._emit('INFO',msg)
    def warning(self,msg:str)->None: self._emit('WARNING',msg)
    def error(self,msg:str)->None: self._emit('ERROR',msg)
    def critical(self,msg:str)->None: self._emit('CRITICAL',msg)

    # optional analytics utilities (kept for compatibility)
    def query_events(self,table_name:str=None,event_type:str=None,user_id:str=None,start_time:str=None,end_time:str=None,limit:int=100)->List[dict]:
        out=[]
        with sqlite3.connect(self.db_path) as conn:
            cur=conn.cursor(); cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_events'")
            avail=[r[0] for r in cur.fetchall()]
            tables=[table_name] if table_name and table_name in avail else avail
            for t in tables:
                q=f"SELECT * FROM {t} WHERE 1=1"; p=[]
                if event_type: q+=" AND event_type = ?"; p.append(event_type)
                if user_id: q+=" AND user_id = ?"; p.append(user_id)
                if start_time: q+=" AND timestamp >= ?"; p.append(start_time)
                if end_time: q+=" AND timestamp <= ?"; p.append(end_time)
                q+=" ORDER BY timestamp DESC LIMIT ?"; p.append(limit)
                conn.row_factory=sqlite3.Row; cur=conn.execute(q,p); out.extend([dict(r) for r in cur.fetchall()])
        return sorted(out,key=lambda x:x['timestamp'],reverse=True)[:limit]

    def _get_unique_events(self,table:str,user_id:str)->Set[str]:
        with sqlite3.connect(self.db_path) as conn:
            cur=conn.execute(f"SELECT DISTINCT event_type FROM {table} WHERE user_id = ?",(user_id,))
            return {r[0] for r in cur.fetchall()}

    def get_user_stats(self,user_id:str)->dict:
        s={'total_events':0,'unique_events':set(),'first_seen':None,'last_seen':None,'tables':{}}
        with sqlite3.connect(self.db_path) as conn:
            cur=conn.cursor(); cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_events'"); tables=[r[0] for r in cur.fetchall()]
            for t in tables:
                cur.execute(f"SELECT COUNT(*) , COUNT(DISTINCT event_type), MIN(timestamp), MAX(timestamp) FROM {t} WHERE user_id = ?",(user_id,))
                c,ue,fs,ls=cur.fetchone()
                if c>0:
                    s['tables'][t]={'count':c,'unique_events':ue,'first_seen':fs,'last_seen':ls}
                    s['total_events']+=c; s['unique_events'].update(self._get_unique_events(t,user_id))
                    s['first_seen']=fs if not s['first_seen'] or (fs and fs<s['first_seen']) else s['first_seen']
                    s['last_seen']=ls if not s['last_seen'] or (ls and ls>s['last_seen']) else s['last_seen']
        s['unique_events']=len(s['unique_events']); return s

    def _get_all_events(self,table:str)->Set[str]:
        with sqlite3.connect(self.db_path) as conn:
            cur=conn.execute(f"SELECT DISTINCT event_type FROM {table}"); return {r[0] for r in cur.fetchall()}

    def _get_all_users(self,table:str)->Set[str]:
        with sqlite3.connect(self.db_path) as conn:
            cur=conn.execute(f"SELECT DISTINCT user_id FROM {table} WHERE user_id IS NOT NULL"); return {r[0] for r in cur.fetchall()}

    def get_event_stats(self)->dict:
        s={'total_events':0,'unique_events':set(),'unique_users':set(),'first_event':None,'last_event':None,'tables':{}}
        with sqlite3.connect(self.db_path) as conn:
            cur=conn.cursor(); cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_events'"); tables=[r[0] for r in cur.fetchall()]
            for t in tables:
                cur.execute(f"SELECT COUNT(*), COUNT(DISTINCT event_type), COUNT(DISTINCT user_id), MIN(timestamp), MAX(timestamp) FROM {t}")
                c,ue,uu,fe,le=cur.fetchone()
                if c>0:
                    s['tables'][t]={'count':c,'unique_events':ue,'unique_users':uu,'first_event':fe,'last_event':le}
                    s['total_events']+=c; s['unique_events'].update(self._get_all_events(t)); s['unique_users'].update(self._get_all_users(t))
                    s['first_event']=fe if not s['first_event'] or (fe and fe<s['first_event']) else s['first_event']
                    s['last_event']=le if not s['last_event'] or (le and le>s['last_event']) else s['last_event']
        s['unique_events']=len(s['unique_events']); s['unique_users']=len(s['unique_users']); return s
