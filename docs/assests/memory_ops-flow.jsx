import { useState, useCallback } from "react";

const MODULES = {
  discord_bot: {
    label: "discord_bot.py",
    color: "#e8d44d",
    bg: "#2a2718",
    ops: [
      { id: "db_search", type: "READ", fn: "search_async(query, k=32, user_id)", trigger: "on_message / process_message", desc: "parallel with history fetch. query = sanitized content + @user + #channel. user_id scoped in DMs, None in guilds." },
      { id: "db_add_interaction", type: "WRITE", fn: "add_memory_async(user_id, memory_text)", trigger: "after response sent", desc: "stores full exchange: @user in #channel (timestamp): message\\n@bot: response" },
      { id: "db_add_thought", type: "WRITE", fn: "add_memory_async(user_id, memory_string)", trigger: "generate_and_save_thought (fire-and-forget task)", desc: "stores 'Reflections on interactions with @user (timestamp): {llm thought}'. runs in background after every interaction." },
      { id: "db_add_cmd", type: "WRITE", fn: "add_memory(user_id, memory_text)", trigger: "!add_memory command", desc: "sync write from command handler. direct user-initiated memory injection." },
      { id: "db_clear", type: "DELETE", fn: "clear_user_memories(user_id)", trigger: "!clear_memories command", desc: "nulls all user slots → _compact() → full reindex. admin/ally only." },
      { id: "db_search_cmd", type: "READ", fn: "search_async(query, user_id)", trigger: "!search_memories command", desc: "explicit user-facing search. returns top results with scores." },
      { id: "db_first_check", type: "READ", fn: "user_memories.get(user_id, [])", trigger: "process_message entry", desc: "checks if user has any memories to select introduction vs chat_with_memory prompt." },
    ]
  },
  defaultmode: {
    label: "defaultmode.py (DMN)",
    color: "#d35db3",
    bg: "#2a1827",
    ops: [
      { id: "dmn_select", type: "READ", fn: "user_memories[uid] → memories[mid]", trigger: "_select_random_memory (each tick)", desc: "weighted random walk: pick user by memory count → pick memory by decay weight. direct list access, no search." },
      { id: "dmn_search", type: "READ", fn: "search(seed, user_id, k=top_k)", trigger: "_generate_thought", desc: "seeds BM25 with random memory. finds related memories for combination. run in executor to avoid blocking event loop." },
      { id: "dmn_add", type: "WRITE", fn: "add_memory_async(user_id, thought)", trigger: "after LLM generates thought", desc: "stores 'Reflections on priors with @user (timestamp): {generated thought}'. attributed to seed user." },
      { id: "dmn_prune_idx", type: "MUTATE", fn: "inverted_index[term].remove(mid)", trigger: "post-thought term overlap processing", desc: "removes overlapping terms from related memories' index entries. the core pruning mechanism — erodes retrieval paths of consolidated memories." },
      { id: "dmn_cleanup", type: "DELETE", fn: "_cleanup_disconnected_memories()", trigger: "after thought gen + after orphan spike", desc: "finds memories with zero index terms → removes from list → remaps ALL ids across memories, user_memories, inverted_index, AND memory_weights. this is the second compaction path." },
      { id: "dmn_save", type: "PERSIST", fn: "_saver.request()", trigger: "after index pruning", desc: "triggers debounced pickle save after term removal." },
      { id: "dmn_index_read", type: "READ", fn: "memories.index(memory)", trigger: "building term maps", desc: "O(n) scan to find mid from memory text. used for seed + all related memories. fragile if duplicates exist." },
    ]
  },
  spike: {
    label: "spike.py",
    color: "#4dc9f6",
    bg: "#172028",
    ops: [
      { id: "sp_search", type: "READ", fn: "search_async(search_key, k=12, user_id=None)", trigger: "process_spike", desc: "global search (no user scoping). search_key = compressed surface context only — orphan excluded to avoid self-bias." },
      { id: "sp_score", type: "READ", fn: "clean_text(content)", trigger: "score_match", desc: "uses memory_index.clean_text for tokenization but does NOT touch the index. inline BM25 against compressed surface." },
      { id: "sp_add_interaction", type: "WRITE", fn: "add_memory_async(bot.user.id, memory_text)", trigger: "after spike response sent", desc: "stores spike event under BOT's user id: 'spike reached #channel (timestamp): orphan: ... response: ...'" },
      { id: "sp_add_reflection", type: "WRITE", fn: "add_memory_async(bot.user.id, reflection)", trigger: "_reflect_on_spike (background task)", desc: "stores 'Reflections on spike to #channel (timestamp): {thought}'. also under bot's user id." },
    ]
  },
  attention: {
    label: "attention.py",
    color: "#7bc67e",
    bg: "#1a2a1b",
    ops: [
      { id: "at_global_read", type: "READ", fn: "memories[] (via _texts_global)", trigger: "theme refresh (background thread)", desc: "reads ALL non-null memories. extracts trigrams + skip-grams. O(corpus) scan." },
      { id: "at_user_read", type: "READ", fn: "user_memories[uid] → memories[mid]", trigger: "theme refresh per user", desc: "reads user's memory subset for personalized theme extraction." },
      { id: "at_trigger_read", type: "READ", fn: "themes cache (derived from index)", trigger: "check_attention_triggers_fuzzy", desc: "fuzzy-matches incoming message against dynamic theme triggers. themes are periodically rebuilt from memory corpus." },
      { id: "at_stats", type: "READ", fn: "user_memories, memories", trigger: "_get_memory_stats", desc: "counts total/active memories and users. diagnostic only." },
    ]
  },
  hippocampus: {
    label: "hippocampus.py",
    color: "#f5a623",
    bg: "#2a2218",
    ops: [
      { id: "hp_rerank", type: "READ", fn: "rerank_memories(query, memories, threshold, blend)", trigger: "rerank_if_enabled (shared pipeline)", desc: "takes BM25 results, generates embeddings via ollama, cosine similarity, blends scores. does NOT touch the index — pure transform on search results." },
    ]
  },
  memory: {
    label: "memory.py (UserMemoryIndex)",
    color: "#aaa",
    bg: "#1e1e1e",
    ops: [
      { id: "mi_add", type: "WRITE", fn: "add_memory(uid, text)", trigger: "all writers above", desc: "mid = len(memories). append to list. append mid to user_memories[uid]. tokenize → post to inverted_index. trigger debounced save." },
      { id: "mi_search", type: "READ", fn: "search(query, k, user_id, dedup)", trigger: "all readers above", desc: "BM25 over inverted_index. length-normalized. trigram jaccard dedup. token-budget windowed. all under RLock." },
      { id: "mi_clear", type: "DELETE", fn: "clear_user_memories(uid)", trigger: "!clear_memories", desc: "null all user's slots → _compact() → renumber everything. nuclear option." },
      { id: "mi_compact", type: "MUTATE", fn: "_compact()", trigger: "clear_user_memories + load_cache(cleanup_nulls)", desc: "builds remap table. filters memories list. remaps user_memories + inverted_index. all ids shift." },
      { id: "mi_save", type: "PERSIST", fn: "AtomicSaver (debounced)", trigger: "after any write/mutate", desc: "snapshot under RLock → pickle to .tmp → atomic os.replace. debounce 300ms." },
      { id: "mi_load", type: "PERSIST", fn: "load_cache()", trigger: "init / startup", desc: "pickle.load → optionally compact nulls → rebuild state." },
    ]
  }
};

const FLOW_STEPS = [
  { title: "1. message arrives", desc: "on_message fires. attention system checks fuzzy triggers against theme cache (derived from memory corpus).", actors: ["at_trigger_read"], highlight: "attention" },
  { title: "2. parallel fetch", desc: "if triggered: search_async + fetch_history + scrape_urls launch concurrently. search query = content + @user + #channel.", actors: ["db_search", "db_first_check"], highlight: "discord_bot" },
  { title: "3. hippocampus rerank", desc: "BM25 candidates pass through embedding reranker. ollama generates vectors, cosine blends with initial scores. pure transform — no index mutation.", actors: ["hp_rerank"], highlight: "hippocampus" },
  { title: "4. response + memory store", desc: "LLM generates response. interaction stored as memory under user_id. fires background thought generation.", actors: ["db_add_interaction", "db_add_thought"], highlight: "discord_bot" },
  { title: "5. DMN tick (background)", desc: "every ~240s: weighted random memory selection → BM25 search for related → LLM combines → new thought stored. overlapping terms pruned from related memories' index entries.", actors: ["dmn_select", "dmn_search", "dmn_add", "dmn_prune_idx"], highlight: "defaultmode" },
  { title: "6. orphan → spike", desc: "if DMN finds no related memories: orphan delegated to spike. spike scores compressed surfaces via inline BM25 (no index mutation). fires into best channel.", actors: ["dmn_cleanup", "sp_search", "sp_score", "sp_add_interaction", "sp_add_reflection"], highlight: "spike" },
  { title: "7. cleanup convergence", desc: "DMN's _cleanup_disconnected_memories finds memories with zero index terms (fully pruned) → removes + remaps all ids. ghosts from stale mids self-resolve here.", actors: ["dmn_cleanup", "mi_compact", "mi_save"], highlight: "memory" },
  { title: "8. theme refresh (background)", desc: "attention system periodically re-extracts trigrams from full corpus. themes feed back into trigger detection + prompt formatting.", actors: ["at_global_read", "at_user_read"], highlight: "attention" },
];

const OP_COLORS = { READ: "#4dc9f6", WRITE: "#7bc67e", DELETE: "#e85d5d", MUTATE: "#d35db3", PERSIST: "#f5a623" };

export default function MemoryFlow() {
  const [activeStep, setActiveStep] = useState(null);
  const [activeModule, setActiveModule] = useState(null);
  const [hoveredOp, setHoveredOp] = useState(null);

  const isHighlighted = useCallback((opId) => {
    if (activeStep !== null) return FLOW_STEPS[activeStep].actors.includes(opId);
    return false;
  }, [activeStep]);

  const isModuleActive = useCallback((modKey) => {
    if (activeModule === modKey) return true;
    if (activeStep !== null) return FLOW_STEPS[activeStep].highlight === modKey;
    return false;
  }, [activeModule, activeStep]);

  return (
    <div style={{ fontFamily: "'JetBrains Mono', 'Fira Code', monospace", background: "#0d0d0d", color: "#ccc", minHeight: "100vh", padding: "24px" }}>
      <div style={{ maxWidth: 1200, margin: "0 auto" }}>
        <h1 style={{ fontSize: 20, fontWeight: 400, color: "#666", marginBottom: 4, letterSpacing: "0.15em", textTransform: "uppercase" }}>defaultMODE</h1>
        <h2 style={{ fontSize: 14, color: "#444", marginBottom: 32, fontWeight: 300 }}>memory index operation flow across all modules</h2>

        <div style={{ display: "flex", gap: 8, marginBottom: 24, flexWrap: "wrap" }}>
          {Object.entries(OP_COLORS).map(([type, color]) => (
            <span key={type} style={{ fontSize: 10, padding: "2px 8px", border: `1px solid ${color}40`, color, borderRadius: 2, letterSpacing: "0.1em" }}>{type}</span>
          ))}
        </div>

        <div style={{ marginBottom: 32 }}>
          <div style={{ fontSize: 11, color: "#555", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.15em" }}>intent flow (click to trace)</div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {FLOW_STEPS.map((step, i) => (
              <button
                key={i}
                onClick={() => setActiveStep(activeStep === i ? null : i)}
                onMouseEnter={() => setActiveModule(null)}
                style={{
                  background: activeStep === i ? `${MODULES[step.highlight].color}20` : "#151515",
                  border: `1px solid ${activeStep === i ? MODULES[step.highlight].color : "#2a2a2a"}`,
                  color: activeStep === i ? MODULES[step.highlight].color : "#666",
                  padding: "6px 12px",
                  fontSize: 11,
                  cursor: "pointer",
                  borderRadius: 2,
                  transition: "all 0.15s",
                  whiteSpace: "nowrap"
                }}
              >
                {step.title}
              </button>
            ))}
          </div>
          {activeStep !== null && (
            <div style={{ marginTop: 12, padding: "12px 16px", background: `${MODULES[FLOW_STEPS[activeStep].highlight].color}08`, border: `1px solid ${MODULES[FLOW_STEPS[activeStep].highlight].color}30`, borderRadius: 2, fontSize: 12, lineHeight: 1.6, color: "#999" }}>
              {FLOW_STEPS[activeStep].desc}
            </div>
          )}
        </div>

        <div style={{ display: "grid", gap: 12 }}>
          {Object.entries(MODULES).map(([key, mod]) => {
            const active = isModuleActive(key);
            return (
              <div
                key={key}
                onClick={() => { setActiveModule(activeModule === key ? null : key); setActiveStep(null); }}
                style={{
                  background: active ? mod.bg : "#111",
                  border: `1px solid ${active ? mod.color + "60" : "#1a1a1a"}`,
                  borderRadius: 3,
                  padding: "16px 20px",
                  cursor: "pointer",
                  transition: "all 0.2s"
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: active ? 12 : 0 }}>
                  <span style={{ width: 8, height: 8, borderRadius: "50%", background: mod.color, opacity: active ? 1 : 0.3, transition: "opacity 0.2s" }} />
                  <span style={{ fontSize: 13, color: active ? mod.color : "#555", fontWeight: 500, transition: "color 0.2s" }}>{mod.label}</span>
                  <span style={{ fontSize: 10, color: "#444", marginLeft: "auto" }}>{mod.ops.length} ops</span>
                </div>
                {(active) && (
                  <div style={{ display: "grid", gap: 6, marginTop: 8 }}>
                    {mod.ops.map(op => {
                      const lit = isHighlighted(op.id);
                      const hovered = hoveredOp === op.id;
                      return (
                        <div
                          key={op.id}
                          onMouseEnter={() => setHoveredOp(op.id)}
                          onMouseLeave={() => setHoveredOp(null)}
                          onClick={(e) => e.stopPropagation()}
                          style={{
                            background: lit ? `${OP_COLORS[op.type]}10` : hovered ? "#1a1a1a" : "#141414",
                            border: `1px solid ${lit ? OP_COLORS[op.type] + "50" : "#1e1e1e"}`,
                            borderRadius: 2,
                            padding: "10px 14px",
                            transition: "all 0.15s"
                          }}
                        >
                          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: hovered ? 8 : 0 }}>
                            <span style={{ fontSize: 9, padding: "1px 6px", background: OP_COLORS[op.type] + "20", color: OP_COLORS[op.type], border: `1px solid ${OP_COLORS[op.type]}40`, borderRadius: 2, letterSpacing: "0.08em", fontWeight: 600 }}>{op.type}</span>
                            <code style={{ fontSize: 11, color: lit ? "#eee" : "#888" }}>{op.fn}</code>
                            <span style={{ fontSize: 10, color: "#444", marginLeft: "auto", whiteSpace: "nowrap" }}>{op.trigger}</span>
                          </div>
                          {hovered && (
                            <div style={{ fontSize: 11, color: "#777", lineHeight: 1.5, paddingLeft: 2 }}>{op.desc}</div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div style={{ marginTop: 32, padding: "16px 20px", background: "#111", border: "1px solid #1a1a1a", borderRadius: 3 }}>
          <div style={{ fontSize: 11, color: "#555", textTransform: "uppercase", letterSpacing: "0.15em", marginBottom: 12 }}>ghost lifecycle</div>
          <div style={{ fontSize: 12, color: "#666", lineHeight: 1.8 }}>
            <span style={{ color: "#7bc67e" }}>birth</span> — add_memory appends, mid = position in list. inverted_index posts terms.
            <br />
            <span style={{ color: "#d35db3" }}>erosion</span> — DMN prunes overlapping terms from related memories' index entries. retrieval paths decay.
            <br />
            <span style={{ color: "#f5a623" }}>orphan</span> — memory with zero index terms. DMN search returns nothing. delegated to spike.
            <br />
            <span style={{ color: "#e85d5d" }}>death</span> — _cleanup_disconnected_memories removes + remaps. or clear_user_memories nukes + _compact.
            <br />
            <span style={{ color: "#4dc9f6" }}>ghost</span> — stale mid held across an await boundary after reindex. hits None slot or wrong memory. self-corrects next search.
            <br />
            <span style={{ color: "#aaa" }}>convergence</span> — all ghosts are transient. cleanup is convergent. the system reconverges because nothing holds mids as durable identity.
          </div>
        </div>

        <div style={{ marginTop: 16, padding: "16px 20px", background: "#111", border: "1px solid #1a1a1a", borderRadius: 3 }}>
          <div style={{ fontSize: 11, color: "#555", textTransform: "uppercase", letterSpacing: "0.15em", marginBottom: 12 }}>concurrency model</div>
          <div style={{ fontSize: 12, color: "#666", lineHeight: 1.8 }}>
            single process. single UserMemoryIndex instance created in setup_bot, shared by reference to DMN, spike, attention, hippocampus.
            <br />
            RLock serializes all mutations. async ops use run_in_executor (thread pool) but still acquire the same lock.
            <br />
            background threads: AtomicSaver (debounced pickle), TempJanitor (file cleanup), theme refresh workers. all read-only or lock-acquiring.
            <br />
            fire-and-forget tasks: generate_and_save_thought, _reflect_on_spike, _cleanup_disconnected_memories. all go through add_memory_async → RLock.
            <br />
            the only unprotected direct access: DMN's _select_random_memory reads memories[mid] and user_memories without lock. safe because list reads are atomic in CPython (GIL) and worst case is a stale read that self-corrects.
          </div>
        </div>

        <div style={{ marginTop: 16, padding: "16px 20px", background: "#111", border: "1px solid #1a1a1a", borderRadius: 3 }}>
          <div style={{ fontSize: 11, color: "#555", textTransform: "uppercase", letterSpacing: "0.15em", marginBottom: 12 }}>two compaction paths</div>
          <div style={{ fontSize: 12, color: "#666", lineHeight: 1.8 }}>
            <span style={{ color: "#e85d5d" }}>_compact()</span> in memory.py — triggered by clear_user_memories. nulls slots, builds remap, rewrites everything.
            <br />
            <span style={{ color: "#d35db3" }}>_cleanup_disconnected_memories()</span> in defaultmode.py — triggered after DMN thought gen. finds zero-term memories, removes, remaps including memory_weights dict. different remap logic (count-based offset vs dict lookup).
            <br />
            both are correct independently. both run under _mut lock. they cannot race each other. but they implement parallel reindex logic — if one changes, the other needs to match. this is the real maintenance risk, not the ghosts.
          </div>
        </div>
      </div>
    </div>
  );
}
