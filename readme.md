<div align="center" style="pointer-events: none;">

<img src="docs/assests/pink_title.png" alt="title" width="75%" style="image-rendering: pixelated;">

</div>

# defaultMODE: Emergent Self-Regulating AI Entities

`defaultMODE` is a cognitive architecture for discord agents that remember, forget, and dream.

memory systems prune and specialize through use. attention emerges from what the agent already knows. arousal modulates creativity based on context richness. a background process walks memory while idle, generating reflections, strengthening some connections, letting others decay.

local-first, model-agnostic. works with ollama, openai, anthropic, vllm, gemini. the skeleton stays coherent even when the model is small.

not just another chatbot framework. a framework for entities that persist.

[UPDATES](docs/updates.md)

----

# why choose defaultMODE?

multi-user chatbots lose themselves. large cloud models can hold character across long conversations, but smaller open-source models collapse—mirroring whoever spoke last, forgetting their own voice after one turn. the longer the context, the more the self dissolves.

most frameworks ignore this. they assume the model will just figure it out.

defaultMODE is an animated skeleton. 💀 the cognitive architecture maintains shape even when the underlying model is small or forgetful. memory, attention, and arousal systems do the work of coherence so the model doesn't have to hold everything in context. you can strip bones out and the thing still stands.

tune `bot_config.py` when running lighter models. the framework adapts; context rot becomes optional.

---


<table align="center" width="100%">
    <tr>
        <td width="40%" valign="middle">
            <img src="docs/assests/dmn-visualise.gif" alt="dmn demo" width="100%" style="image-rendering: pixelated;">
        </td>
        <td width="60%" valign="middle">
            <img src="docs/assests/pink_banner.png" alt="dm banner" width="100%" style="image-rendering: pixelated;">
        </td>
    </tr>
</table>

---



# Features and Abilities

```
input → attention filter → hippocampal retrieval → reranking by embedding
                                    ↓
              context assembly ← temporal parsing ← conversation history
                                    ↓
                    amygdala samples arousal from memory density
                                    ↓
                         prompt construction → llm → response
                                    ↓
                    memory storage → thought generation → dmn integration
                                    ↑
              [background: dmn walks, prunes, dreams, forgets]
```

## cognitive architecture

- **default mode network** — background process performs associative memory walks, generates reflective thoughts, prunes term overlap between related memories, and manages graceful forgetting. the agent dreams between conversations.
- **amygdala complex** — memory density modulates arousal which scales llm temperature dynamically. sparse context → careful, deterministic. rich context → creative, exploratory. emotional tone emerges from cognitive state.
- **hippocampal formation** — hybrid retrieval blending inverted index with tf-idf scoring and embedding-based reranking at inference time. bandwidth adapts to arousal level for human-like recall under pressure.
- **temporal integration** — timestamps parsed as natural language expressions ("yesterday morning", "last week") rather than raw datetime, giving the agent intuitive temporal reasoning about its memories.

## attention and engagement

- **fuzzy topic matching** — attention triggers use semantic similarity against defined interests plus emergent themes mined from memory. agents join conversations that resonate with what's already on their mind.
- **theme emergence** — preferences crystallize from interaction patterns. the agent develops interests it wasn't explicitly given, contributing to attention triggers organically.
- **distributed homeostasis** — all modules regulate each other. attention depends on themes from memory. arousal depends on memory density. memory quality depends on dmn pruning. no central controller, just coupled oscillators.

## context and memory

- **channel vs dm siloing** — memories respect privacy boundaries. dm conversations stay private to that user. channel context stays scoped to that space. context switching handled intelligently.
- **term pruning and decay** — overlapping terms between connected memories are removed during reflection, forcing specialization. memories with no remaining connections are forgotten. the index breathes.
- **persistence** — pickled inverted index survives restarts. the agent wakes up remembering.

## content ingestion

- **web and youtube grokking** — shared links scraped and processed using holistic "skim" reading rather than narrow chunking. content understood in context, not fragments.
- **file and image processing** — attachments analyzed with vision models when available. text files, code, images all flow into memory and context.
- **github integration** — repository indexing, file-specific chat, and rag-style repo questions. code becomes part of the agent's extended mind.

## discord-native design

- **message conditioning** — username logic, mention handling, reaction tracking, chunking for discord limits, code block preservation. seamless integration without fighting the platform.
- **multi-agent ready** — multiple bot instances with separate memory indices, api configurations, and personalities. they can coexist and potentially interact.
- **graceful degradation** — kill/resume commands, processing toggles, attention on/off. operators maintain control without losing state.

## observability

- **dual logging** — jsonl for streaming analysis, sqlite for structured queries. every interaction, thought generation, and memory operation tracked.
- **runtime adjustable** — temperature, reranking thresholds, attention sensitivity all tunable without restart. watch the agent shift in real time.



---

**Getting Started**

1.  **Clone:** `git clone https://github.com/everyoneisgross/defaultmodeAGENT && cd defaultmodeAGENT`
2.  **Install:** `pip install -r requirements.txt`
3.  **Configure:** Create a `.env` file (refer to `.env.example`) and populate it with your Discord token and any necessary API keys.
4.  **Define Your Agent:** Create `system_prompts.yaml` and `prompt_formats.yaml` within the `/agent/prompts/your_agent_name/` directory. (Example files are provided.)

    ```yaml
    # Example system_prompts.yaml snippet:
    default_chat: |
      You are a curious AI entity.  Your name is {bot_name}.  You have a persistent memory and can reflect on past interactions. Your current intensity level is {amygdala_response}%. At 0% you are boring at 100% you are too much fun. Your preferences for things are {themes}. 
    ```

5.  **Run:** `python agent/discord_bot.py --api ollama --model hermes3 --bot-name your_agent_name`

**Technical Overview**

*   **Persistence:** Memories are persisted using a pickled inverted-index, ensuring data is preserved between sessions and can be all held in memory for fast inference.
*   **Analysis:** JSONL logs and an SQLite database are included for auditing and analysis.
*   **Configuration:** Managed via YAML files for prompt definitions and environment variables for sensitive credentials and API keys.
*   **Code:** Python, with an emphasis on tool modularity. Abstractions will transfer to other social platforms eventually.
*   **Dependencies:** Detailed in `requirements.txt`, including libraries for Discord interaction, LLM APIs, and data handling.

---

# Further Reading:

1.  [Cognition Analogy](docs/cognitionanalogy.md)
2.  [Memory Module](docs/memory.md)
3.  [Memory Editor](docs/memory_editor.md)
4.  [Default Mode Network Flow](docs/defaultmode_flow.md)
5.  [Prompting Guide](docs/prompting.md)
6.  [Attention Triggers](docs/attention.md)