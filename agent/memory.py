import os, json, pickle, string, re, math, asyncio, tempfile, shutil, uuid
from collections import defaultdict, Counter, deque
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
from bot_config import config
from tokenizer import get_tokenizer, count_tokens
from logger import BotLogger, logging

MAX_TOKENS = config.search.max_tokens
CONTEXT_CHUNKS = config.search.context_chunks

class CacheManager:
    def __init__(self, bot_name, temp_file_ttl=3600):
        self.bot_name=bot_name
        self.temp_file_ttl=temp_file_ttl
        self.base_cache_dir=os.path.join('cache',self.bot_name)
        self.logger=logging.getLogger(f'bot.{self.bot_name}.cache')
        self._save_task=None
        self._save_lock=asyncio.Lock()
        self._debounce=3.0
        os.makedirs(self.base_cache_dir, exist_ok=True)

    def get_cache_dir(self, cache_type):
        cache_dir=os.path.join(self.base_cache_dir,cache_type)
        os.makedirs(cache_dir,exist_ok=True)
        return cache_dir

    def get_temp_dir(self):
        temp_dir=self.get_cache_dir('temp')
        self.cleanup_temp_files()
        return temp_dir

    def get_user_temp_dir(self,user_id):
        d=os.path.join(self.get_temp_dir(),str(user_id))
        os.makedirs(d,exist_ok=True)
        return d

    def create_temp_file(self,user_id,prefix=None,suffix=None,content=None):
        file_id=str(uuid.uuid4())
        timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        parts=[]
        if prefix: parts.append(prefix)
        parts.extend([timestamp,file_id])
        filename='_'.join(parts)
        if suffix: filename+=suffix
        user_temp_dir=self.get_user_temp_dir(user_id)
        p=os.path.join(user_temp_dir,filename)
        mode='wb' if isinstance(content,bytes) else 'w'
        encoding=None if isinstance(content,bytes) else 'utf-8'
        with open(p,mode,encoding=encoding) as f:
            if content is not None: f.write(content)
        try: self.logger.info(f"Created temporary file for user {user_id}: {p}")
        except AttributeError: pass
        metadata={'created_at':datetime.now().isoformat(),'user_id':user_id,'file_id':file_id,'original_filename':filename}
        mp=f"{p}.meta"
        with open(mp,'w') as f: json.dump(metadata,f)
        return p,file_id

    def get_temp_file(self,user_id,file_id):
        user_temp_dir=self.get_user_temp_dir(user_id)
        try:
            for filename in os.listdir(user_temp_dir):
                if filename.endswith('.meta'):
                    mp=os.path.join(user_temp_dir,filename)
                    with open(mp,'r') as f: metadata=json.load(f)
                    if metadata['file_id']==file_id and metadata['user_id']==user_id:
                        fp=mp[:-5]
                        if os.path.exists(fp): return fp
            return None
        except Exception as e:
            try: self.logger.error(f"Error retrieving temporary file {file_id} for user {user_id}: {str(e)}")
            except AttributeError: pass
            return None

    def cleanup_temp_files(self,force=False):
        current_time=datetime.now()
        temp_dir=self.get_cache_dir('temp')
        try:
            for user_id in os.listdir(temp_dir):
                user_temp_dir=os.path.join(temp_dir,user_id)
                if not os.path.isdir(user_temp_dir): continue
                for filename in os.listdir(user_temp_dir):
                    if filename.endswith('.meta'): continue
                    fp=os.path.join(user_temp_dir,filename)
                    mp=f"{fp}.meta"
                    try:
                        if os.path.exists(mp):
                            with open(mp,'r') as f: metadata=json.load(f)
                            created_at=datetime.fromisoformat(metadata['created_at'])
                            file_age=current_time-created_at
                            if force or file_age>timedelta(seconds=self.temp_file_ttl):
                                self.remove_temp_file(metadata['user_id'],metadata['file_id'])
                        else:
                            file_age=current_time-datetime.fromtimestamp(os.path.getctime(fp))
                            if force or file_age>timedelta(seconds=self.temp_file_ttl):
                                try:
                                    os.remove(fp)
                                    try: self.logger.info(f"Removed orphaned temporary file {filename} for user {user_id}")
                                    except AttributeError: pass
                                except Exception as e:
                                    try: self.logger.error(f"Error removing orphaned file {fp}: {str(e)}")
                                    except AttributeError: pass
                    except Exception as e:
                        try: self.logger.error(f"Error processing temporary file {fp}: {str(e)}")
                        except AttributeError: pass
                if not os.listdir(user_temp_dir): os.rmdir(user_temp_dir)
        except Exception as e:
            try: self.logger.error(f"Error during temp file cleanup: {str(e)}")
            except AttributeError: pass

    def remove_temp_file(self,user_id,file_id):
        fp=self.get_temp_file(user_id,file_id)
        if not fp: return
        try:
            if os.path.exists(fp): os.remove(fp)
            mp=f"{fp}.meta"
            if os.path.exists(mp): os.remove(mp)
            try: self.logger.info(f"Removed temporary file {file_id} for user {user_id}")
            except AttributeError: pass
        except Exception as e:
            try: self.logger.error(f"Error removing temporary file {file_id} for user {user_id}: {str(e)}")
            except AttributeError: pass

class UserMemoryIndex:
    def __init__(self, cache_type, max_tokens=MAX_TOKENS, context_chunks=CONTEXT_CHUNKS, logger=None):
        parts=cache_type.split('/')
        if len(parts)>=2:
            self.bot_name=parts[0]; cache_subtype=parts[-1]
        else:
            self.bot_name='default'; cache_subtype=cache_type
        self.cache_manager=CacheManager(self.bot_name)
        self.cache_dir=self.cache_manager.get_cache_dir(cache_subtype)
        self.max_tokens=max_tokens
        self.context_chunks=context_chunks
        self.tokenizer=get_tokenizer()
        self.inverted_index=defaultdict(list)
        self.memories=[]
        self.stopwords=set(['the','a','an','and','or','but','nor','yet','so','in','on','at','to','for','of','with','by','from','up','about','i','you','he','she','it','we','they','me','him','her','us','them','is','are','was','were','be','been','have','has','had','can','could','may','might','must','shall','should','will','would','this','that','these','those'])
        self.user_memories=defaultdict(list)
        self.logger=logger or logging.getLogger('bot.default')
        self._save_task=None
        self._save_lock=asyncio.Lock()
        self._debounce=3.0
        self.load_cache()

    def clean_text(self,text):
        text=text.replace("<|endoftext|>","").replace("<|im_start|>","").replace("<|im_end|>","").lower()
        text=text.translate(str.maketrans(string.punctuation,' '*len(string.punctuation)))
        text=re.sub(r'\d+','',text)
        words=[w for w in text.split() if w]
        words=[w for w in words if w not in self.stopwords]
        return ' '.join(words)

    def add_memory(self,user_id,memory_text):
        mid=len(self.memories)
        self.memories.append(memory_text)
        self.user_memories[user_id].append(mid)
        for w in self.clean_text(memory_text).split(): self.inverted_index[w].append(mid)
        self.logger.info(f"Added new memory for user {user_id}: {memory_text[:100]}...")
        try: asyncio.get_running_loop().create_task(self._schedule_save())
        except RuntimeError: self._save_cache_sync()

    async def add_memory_async(self,user_id,memory_text):
        mid=len(self.memories)
        self.memories.append(memory_text)
        self.user_memories[user_id].append(mid)
        for w in self.clean_text(memory_text).split(): self.inverted_index[w].append(mid)
        self.logger.info(f"Added new memory for user {user_id}: {memory_text[:100]}...")
        await self._schedule_save()

    def clear_user_memories(self,user_id):
        if user_id in self.user_memories:
            ids=sorted(self.user_memories[user_id],reverse=True)
            for memory_id in ids:
                self.memories.pop(memory_id)
                for uid,mems in self.user_memories.items():
                    self.user_memories[uid]=[mid if mid<memory_id else mid-1 for mid in mems if mid!=memory_id]
                for word in list(self.inverted_index.keys()):
                    self.inverted_index[word]=[mid if mid<memory_id else mid-1 for mid in self.inverted_index[word] if mid!=memory_id]
                    if not self.inverted_index[word]: del self.inverted_index[word]
            del self.user_memories[user_id]
            self.logger.info(f"Cleared and rebuilt index after removing {len(ids)} memories for user {user_id}")
            self.save_cache()

    def search(self,query,k=5,user_id=None,dedup_threshold=0.95):
        cleaned_query=self.clean_text(query)
        query_words=cleaned_query.split()
        memory_scores=Counter()
        total_memories=len([m for m in self.memories if m is not None])
        doc_freqs={word:len(self.inverted_index.get(word,[])) for word in query_words}
        relevant_memory_ids=self.user_memories.get(user_id,[]) if user_id else range(len(self.memories))
        k1=1.2; b=0.75
        doc_lengths={}; total_length=0; valid_count=0
        for mid in relevant_memory_ids:
            if mid<len(self.memories) and self.memories[mid] is not None:
                dl=len(self.memories[mid].split()); doc_lengths[mid]=dl; total_length+=dl; valid_count+=1
        avgdl=total_length/valid_count if valid_count>0 else 1.0
        relevant_set=set(relevant_memory_ids) if user_id else None
        for word in query_words:
            posting_list=self.inverted_index.get(word,[])
            if not posting_list: continue
            posting=[mid for mid in posting_list if (mid in relevant_set)] if relevant_set else posting_list
            if not posting: continue
            tf_counts=Counter(posting)
            df=len(tf_counts)
            idf=math.log((total_memories-df+0.5)/(df+0.5)+1.0)
            for mid,tf in tf_counts.items():
                dl=doc_lengths.get(mid,1)
                length_norm=k1*((1-b)+b*(dl/avgdl))
                w=idf*((k1+1)*tf)/(length_norm+tf)
                memory_scores[mid]+=w
        for mid,score in list(memory_scores.items()):
            dl=doc_lengths.get(mid,1)
            memory_scores[mid]=score/math.log(1+dl)
        max_score=max(memory_scores.values()) if memory_scores else 1.0
        for mid in list(memory_scores.keys()):
            memory_scores[mid]/=max_score
        sorted_memories=sorted(memory_scores.items(),key=lambda x:x[1],reverse=True)
        results=[]; total_tokens=0; seen=set()
        for mid,score in sorted_memories:
            m=self.memories[mid]
            mt=self.count_tokens(m)
            if total_tokens+mt>self.max_tokens: break
            cm=self.clean_text(m)
            dup=False
            for s in seen:
                if self._calculate_similarity(cm,s)>dedup_threshold:
                    dup=True; break
            if not dup:
                results.append((m,score))
                seen.add(cm)
                total_tokens+=mt
                if len(results)>=k: break
        self.logger.info(f"Found {len(results)} unique memories for query: {query[:100]}...")
        return results

    def _calculate_similarity(self,t1,t2):
        def grams(t,n=3): return set(t[i:i+n] for i in range(len(t)-n+1))
        g1=grams(t1); g2=grams(t2)
        inter=len(g1.intersection(g2)); uni=len(g1.union(g2))
        return inter/uni if uni>0 else 0

    def count_tokens(self,text): return count_tokens(text)

    def save_cache(self): self._save_cache_sync()

    async def _schedule_save(self):
        async with self._save_lock:
            if self._save_task and not self._save_task.done(): return
            async def _d():
                await asyncio.sleep(self._debounce)
                await self.save_cache_async()
            self._save_task=asyncio.create_task(_d())

    def _save_cache_sync(self):
        cache_file=os.path.join(self.cache_dir,'memory_cache.pkl')
        tmp=cache_file+'.tmp'
        cache_data={'inverted_index':self.inverted_index,'memories':self.memories,'user_memories':self.user_memories}
        with open(tmp,'wb') as f: pickle.dump(cache_data,f,protocol=5)
        os.replace(tmp,cache_file)
        self.logger.info("Memory cache saved successfully.")

    async def save_cache_async(self):
        loop=asyncio.get_running_loop()
        await loop.run_in_executor(None,self._save_cache_sync)

    def load_cache(self,cleanup_orphans=False,cleanup_nulls=True):
        cache_file=os.path.join(self.cache_dir,'memory_cache.pkl')
        if os.path.exists(cache_file):
            with open(cache_file,'rb') as f: cache_data=pickle.load(f)
            self.inverted_index=cache_data.get('inverted_index',defaultdict(list))
            self.memories=cache_data.get('memories',[])
            self.user_memories=cache_data.get('user_memories',defaultdict(list))
            memory_count=len(self.memories)
            for word in list(self.inverted_index.keys()):
                self.inverted_index[word]=[mid for mid in self.inverted_index[word] if mid<memory_count and self.memories[mid] is not None]
                if not self.inverted_index[word]: del self.inverted_index[word]
            for user_id in list(self.user_memories.keys()):
                self.user_memories[user_id]=[mid for mid in self.user_memories[user_id] if mid<memory_count and self.memories[mid] is not None]
                if not self.user_memories[user_id]: del self.user_memories[user_id]
            if cleanup_orphans:
                all_user_mems=set()
                for mems in self.user_memories.values(): all_user_mems.update(mems)
                orphaned=[(i,m) for i,m in enumerate(self.memories) if i not in all_user_mems]
                if orphaned:
                    for i,m in orphaned: self.logger.warning(f"Found orphaned memory {i}: {m[:100]}...")
                    self.logger.warning(f"Found {len(orphaned)} orphaned memories. Set cleanup_orphans=True to remove them.")
            index_stats={
                'total_memories':len(self.memories),
                'active_memories':len([m for m in self.memories if m is not None]),
                'total_users':len(self.user_memories),
                'vocabulary_size':len(self.inverted_index),
                'index_distribution':{w:len(v) for w,v in self.inverted_index.items()}
            }
            index_stats['memories_per_user']={u:len(m) for u,m in self.user_memories.items()}
            self.logger.info("Memory Index Structure:")
            self.logger.info(f"Total Memories: {index_stats['total_memories']}")
            self.logger.info(f"Active Memories: {index_stats['active_memories']}")
            self.logger.info(f"Total Users: {index_stats['total_users']}")
            self.logger.info(f"Vocabulary Size: {index_stats['vocabulary_size']}")
            self.logger.info(f"Average memories per word: {sum(index_stats['index_distribution'].values())/len(self.inverted_index) if self.inverted_index else 0:.2f}")
            self.logger.info(f"Average memories per user: {sum(index_stats['memories_per_user'].values())/len(self.user_memories) if self.user_memories else 0:.2f}")
            if index_stats['total_memories']!=index_stats['active_memories']:
                nulls=[(i,m) for i,m in enumerate(self.memories) if m is None]
                self.logger.warning(f"Found {len(nulls)} null memories in index:")
                for idx,_ in nulls: self.logger.warning(f"Null entry at index {idx}")
                if cleanup_nulls:
                    for idx,_ in sorted(nulls,reverse=True):
                        self.memories.pop(idx)
                        for user_id in self.user_memories:
                            self.user_memories[user_id]=[mid if mid<idx else mid-1 for mid in self.user_memories[user_id] if mid!=idx]
                        for word in list(self.inverted_index.keys()):
                            self.inverted_index[word]=[mid if mid<idx else mid-1 for mid in self.inverted_index[word] if mid!=idx]
                            if not self.inverted_index[word]: del self.inverted_index[word]
                    self.logger.info(f"Removed {len(nulls)} null entries")
                    self.save_cache()
            self.logger.info("Memory cache loaded successfully.")
            return True
        return False

    async def search_async(self,query,k=32,user_id=None,dedup_threshold=0.95):
        loop=asyncio.get_event_loop()
        self.logger.info(f"Starting async search for query: {query[:100]}...")
        results=await loop.run_in_executor(None,lambda: self.search(query,k,user_id,dedup_threshold))
        self.logger.info(f"Async search completed with {len(results)} results")
        return results

class RepoIndex:
    def __init__(self,cache_dir):
        os.makedirs(cache_dir,exist_ok=True)
        self.cache_dir=cache_dir
        self.repo_index=defaultdict(set)
        self.load_cache()

    def index_file(self,file_path,content,branch='main'):
        words=set(re.findall(r'\b\w{3,}\b',content.lower()))
        self.repo_index[branch].add(file_path)
        with open(os.path.join(self.cache_dir,f'{branch}_{file_path.replace("/","_")}.txt'),'w',encoding='utf-8') as f:
            f.write(content)

    def search_repo(self,query,branch='main',limit=5):
        query_words=set(re.findall(r'\b\w{3,}\b',query.lower()))
        if not query_words: return []
        results=[]
        for file_path in self.repo_index.get(branch,set()):
            try:
                with open(os.path.join(self.cache_dir,f'{branch}_{file_path.replace("/","_")}.txt'),'r',encoding='utf-8') as f:
                    content=f.read()
                file_words=set(re.findall(r'\b\w{3,}\b',content.lower()))
                match_count=sum(1 for w in query_words if w in file_words)
                if match_count>0:
                    score=match_count/len(query_words)
                    results.append((file_path,score))
            except Exception: continue
        results.sort(key=lambda x:x[1],reverse=True)
        return results[:limit]

    async def search_repo_async(self,query,branch='main',limit=5):
        loop=asyncio.get_event_loop()
        return await loop.run_in_executor(None,lambda: self.search_repo(query,branch,limit))

    def clear_cache(self):
        self.repo_index=defaultdict(set)
        for f in os.listdir(self.cache_dir):
            if f.endswith('.txt'):
                os.remove(os.path.join(self.cache_dir,f))

    def save_cache(self):
        with open(os.path.join(self.cache_dir,'repo_index.json'),'w') as f:
            json.dump({k:list(v) for k,v in self.repo_index.items()},f)

    def load_cache(self):
        p=os.path.join(self.cache_dir,'repo_index.json')
        if os.path.exists(p):
            with open(p,'r') as f:
                try:
                    loaded=json.load(f)
                    self.repo_index=defaultdict(set,{k:set(v) for k,v in loaded.items()})
                except:
                    self.repo_index=defaultdict(set)
