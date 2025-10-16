
# Attention Triggers Feature

The attention trigger system allows agents to respond to topic-based keywords without requiring @mentions. This creates more natural conversational flow where agents can "pay attention" to relevant discussions.

## Configuration

Add `attention_triggers` to any agent's `system_prompts.yaml`:

```yaml
attention_triggers:
  - "keyword or phrase"
  - "another trigger"
  - "multi word triggers work too"
```

## How It Works

**Fuzzy Matching**: Uses fuzzy string matching (threshold: 80%) to catch:
- Exact matches: "emergent patterns" → "I see emergent patterns here"
- Typos: "emergnt patterns" → matches "emergent patterns" 
- Variations: "complex spiral" → matches "complexity spiral"

**Response Behavior**: When triggered, agents respond exactly as if @mentioned:
- Same conversation processing
- Same memory integration  
- Same file/URL handling

**Control Commands**:
- `@agent !attention` - Check current status
- `@agent !attention on` - Enable attention triggers
- `@agent !attention off` - Disable (mentions/DMs only)

## Design Guidelines

**Topic-Specific Triggers**: Create triggers that align with each agent's unique personality and expertise rather than generic attention words.

**Examples**:
- **Loop**: "emergent patterns", "recursive thinking", "complexity spiral"
- **Technical Agent**: "debugging", "optimization", "architecture"
- **Creative Agent**: "storytelling", "narrative flow", "creative process"

**Fallback**: Agents without `attention_triggers` only respond to @mentions and DMs.

```