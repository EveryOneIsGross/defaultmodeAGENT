
```mermaid
%%{
  init: {
    'theme': 'base',
    'themeVariables': {
      'primaryColor': '#ff9c9c',
      'primaryTextColor': '#000000',
      'primaryBorderColor': '#000000',
      'lineColor': '#000000',
      'secondaryColor': '#ff9c9c',
      'tertiaryColor': '#ff9c9c',
      'backgroundColor': '#ff9c9c',
      'background': '#ff9c9c',
      'nodeBorder': '#000000',
      'textColor': '#000000',
      'mainBkg': '#ff9c9c',
      'edgeLabelBackground': '#ff9c9c',
      'clusterBkg': '#ff9c9c',
      'clusterBorder': '#000000',
      'titleColor': '#000000',
      'fontFamily': 'Courier New, Courier, monospace',
      'noteBackgroundColor': '#ff9c9c',
      'noteBorderColor': '#000000'
    }
  }
}%%
flowchart TB
    subgraph Memory["Memory Space"]
        MI[Memory Index]
        II[Inverted Index]
        MW[Memory Weights]
    end

    subgraph Selection["Random Walk"]
        RS[Random User Selection]
        WS[Weighted Memory Selection]
        RM[Related Memory Search]
    end

    subgraph Processing["Term Processing"]
        OT[Find Overlapping Terms]
        TP[Term Pruning]
        WD[Weight Decay]
    end

    subgraph Generation["Thought Generation"]
        MC[Memory Context Building]
        TG[Temperature/Intensity Scaling]
        NT[New Thought Generation]
        TS[Temporal State Update]
    end

    subgraph Cleanup["Memory Management"]
        DC[Disconnect Detection]
        MP[Memory Pruning]
        IR[Index Rebalancing]
    end

    MI --> RS
    MW --> WS
    RS --> WS
    WS --> RM
    RM --> OT
    
    OT --> TP
    TP --> WD
    WD --> MW
    
    RM --> MC
    MC --> TG
    TG --> NT
    NT --> TS
    TS --> MI
    
    TP --> DC
    DC --> MP
    MP --> IR
    IR --> II
    II --> MI

```

defaultMODE is a simulation of how the human mind ruminates or wanders through thoughts compressing concepts into a single thought. creating a homeostasis of new thoughts and pruned priors.

1. **Memory Selection Process**
```python
def _select_random_memory(self):
    # Random user selection first
    selected_user_id = random.choice(user_ids)
    # Then weighted memory selection
    selection_point = random.uniform(0, total_weight)
```

2. **Term Processing & Pruning**
```python
# Find and remove overlapping terms
overlapping_terms = set()
for memory_id in memory_terms_map:
    if memory_id != seed_memory_id:
        overlapping_terms.update(seed_terms & memory_terms_map[memory_id])
```

3. **Weight Decay & Update**
```python
decay = removed_terms / len(original_terms)
self.memory_weights[memory] *= (1 - (self.decay_rate * decay))
```

4. **Memory Management**
```python
def _cleanup_disconnected_memories(self):
    # Cleanup disconnected nodes
    connected_memories = set()
    for term_memories in self.memory_index.inverted_index.values():
        connected_memories.update(term_memories)
```

5. **Thought Generation**
```python
new_intensity = min(100, max(1, int(50 * max(0.4, 1.0 - (min(len(related_memories), 20) / 20) * 0.6))))
new_thought = await call_api(prompt=prompt, system_prompt=system_prompt, temperature=self.temperature)
```

The system acts like a self-organizing network where:
- Term relationships drive growth and pruning
- Memory weights evolve naturally through use
- Disconnected memories are cleaned up
- New thoughts create new connections
- The whole system maintains homeostasis through the balance of growth and pruning

The network literally grows and shrinks based on term relationships and usage patterns.

---

1. **Search Evolution Through Pruning**
```python
# Original memory has terms A, B, C, D
# Related memory has terms A, B, C, E
# After pruning:
# Original memory keeps A, B, C, D
# Related memory now only has E (A, B, C pruned)
```
So when you later search, this memory is now more strongly associated with 'E' rather than the common terms! This creates:
- More distinct memory signatures
- Reduced "noise" from common terms
- Emergent specialization of memories

2. **Weight-Based Association Shifts**
```python
decay = removed_terms / len(original_terms)
self.memory_weights[memory] *= (1 - (self.decay_rate * decay))
```
This means:
- Memories that lose many terms become less influential
- Remaining unique terms become proportionally more important
- Search results favor memories with strong unique associations

3. **Emergent Novelty Through Term Distribution**
- As common terms get pruned across multiple memories
- Unique term combinations become more significant
- Search results naturally surface more novel connections
- The system "learns" to recognize unique patterns

4. **Dynamic Search Space**
The inverted index becomes:
```
Before pruning:
term_A -> [mem1, mem2, mem3, mem4]
term_B -> [mem1, mem2, mem3]
term_C -> [mem1, mem4]

After pruning:
term_A -> [mem1]  # Now unique to mem1
term_B -> [mem2]  # Now unique to mem2
term_C -> [mem1, mem4]  # Still shared but less common
```

1. Search finds related memories
2. Pruning makes memories more distinct
3. Future searches find different associations
4. The network organically develops novel pathways
5. Search results become more creative/unexpected

---

## Emergent Social Network


1. **Maintain Individual Identity**
- Each agent has its own memory space and pruning patterns
- Natural preference emergence through term weighting
- Prevents mode collapse through individual memory differentiation

2. **Social Memory Architecture**
```python
# Each agent maintains its own:
self.memory_index = memory_index      # Personal experiences
self.memory_weights = defaultdict()   # Individual associations
self.amgdela_response = 50           # Unique personality
```

3. **Inter-Agent Learning**
- Agents learn about each other through interactions
- Memory pruning creates unique perspectives on shared experiences
- @ mentions show emergent understanding of other agents' specialties

4. **Autonomous Social Dynamics**
```python
# When agent B appears in agent A's memory
memory_users = set()
for memory, _ in related_memories:
    memory_user = await self.bot.fetch_user(int(memory_user_id))
    if memory_user and memory_user.name != user_name:
        memory_users.add(memory_user.name)
```

They can maintain coherent identities and relationships while still operating autonomously. 

- Natural role emergence
- Knowledge specialization
- Social group formation
- Complex inter-agent relationships
