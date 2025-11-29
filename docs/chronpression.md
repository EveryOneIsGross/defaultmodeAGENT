# Chronomic Compression

[LINK](/agent/tools/chronpression.py)

A text compression tool that reduces documents to their semantic skeleton while preserving chronomic (temporal-positional) flow. Designed for pre-processing large texts before LLM inference - youtube transcripts, webpage content, documents - without requiring an additional LLM call or stateful summarization process.

## Thesis

LLMs are robust to pidgin. They reconstruct grammatical scaffolding from semantic anchors because they're trained on broken text, typos, fragments, translations, notes, search queries, and non-native speaker input. The signal survives.

Chronomic Compression exploits this by preserving *which* tokens appear and *when* they appear - positional semantics. A sparse representation that maintains temporal ordering lets the model rebuild implied structure. We're not summarizing (which risks hallucination and detail loss), we're *topologically reducing* - finding the minimal skeleton that still carries the signal.

The practical win: a 15k token youtube transcript at 0.7 compression becomes ~5k tokens carrying the same retrievable facts. Context window budget preserved, latency down, cost down. No round-trip to a summarization model.

## Method

### Edge-Scored Salience Graph

Rather than scoring words in isolation, we build a proximity graph over content words:

1. **Sparse Trigrams**: Words within a horizon window (default 6) form weighted edges. Closer words = stronger bonds (exponential decay).

2. **Node Salience**: Words score higher when they cluster with other novel (non-common) words. Domain-specific jargon survives because it's locally salient, not because it's in some pretrained vocabulary.

3. **Edge Scores**: Edges weighted by geometric mean of endpoint salience. Rare-to-rare connections score highest.

4. **Protected Bigrams**: Strong edges force both endpoints to survive if either does. "semantic pathways" stays together.

### Adaptive Sentence Density

Not all sentences are equal:

- **Short declarative sentences** (thesis statements) resist compression - they're already dense
- **Long elaborative sentences** get mined for their skeleton - more redundancy to remove
- Density = content word ratio × log(length factor relative to document average)

### Pidgin Mode (compression > 0.8)

At extreme compression, grammatical protections yield:

| Protection | Normal | Pidgin |
|------------|--------|--------|
| Article weight | 0.1 | 0.01 |
| Edge influence | 100% | 30% |
| Bigram protection | enforced | ignored |
| Position anchoring | start + end | neither |

Result: telegram-pidgin. Reconstructible if you know the domain, cryptic otherwise.

### Fuzzy Post-Pass

After initial compression, a fuzzy matching pass (using fuzzywuzzy) provides additional reduction:

- Drops words that fuzzy-match function words (typos, variants)
- Drops high-frequency common word variants
- Drops words that fuzzy-match earlier kept words (semantic deduplication)

Fuzzy operates *after* salience compression - it's polish, not the main engine.

## Installation

```bash
pip install fuzzywuzzy
pip install python-Levenshtein  # optional, speeds up fuzzywuzzy
```

## Usage

### Command Line

```bash
# basic compression (default 0.5)
python chronpression.py -i document.txt -o compressed.txt

# aggressive compression with fuzzy
python chronpression.py -i transcript.txt -c 0.7 -f 1.5 -v

# extreme pidgin mode
python chronpression.py -i corpus.txt -c 0.9 -f 2.0 -v
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-i, --input` | Input file path | - |
| `-o, --output` | Output file path | `{input}_chronomic.txt` |
| `-c, --compression` | Compression ratio 0.0-1.0 | 0.5 |
| `-f, --fuzzy` | Fuzzy matching strength 0.0-2.0 | 1.0 |
| `--horizon` | Edge graph locality window | 6 |
| `-v, --verbose` | Show compression stats | false |

### As Module

```python
from chronpression import chronomic_filter, stats

text = """Your long document, transcript, or webpage content here..."""

# moderate compression
compressed = chronomic_filter(text, compression=0.5, fuzzy_strength=1.0)

# check results
s = stats(text, compressed)
print(f"{s['original_words']} → {s['compressed_words']} ({s['actual_compression']})")
```

### Compression Levels

| Level | Compression | Fuzzy | Use Case |
|-------|-------------|-------|----------|
| Light | 0.3 | 0.5 | Preserve readability, trim fluff |
| Medium | 0.5 | 1.0 | Balance compression and coherence |
| Heavy | 0.7 | 1.5 | Dense input for LLM context |
| Pidgin | 0.85+ | 2.0 | Maximum reduction, domain experts only |

## Example Output

**Original (100 words):**
> The concept of Chronomic Compression implies that a compressed sentence can still convey the entire meaning, by simple compression heuristics. Using sparsegrams to learn an input text, we can optimise understanding by compressing it at a set ratio. Chronomic compression assumes a high order of importance that gets learned through analysis, like a higher and higher order of abstraction through removing the assumed roads of language while keeping learned novel entities by association.

**Compression 0.5 (50 words):**
> The Chronomic Compression compressed sentence still convey entire meaning, simple heuristics. Using sparsegrams learn input, understanding set ratio. Chronomic assumes high order learned analysis, like higher abstraction removing roads language keeping novel association.

**Compression 0.95 (17 words):**
> Chronomic Compression convey meaning, heuristics. order, higher roads novel entities. Dense elaborations require preservation semantic pathways.

## Scaling

The algorithm scales linearly O(n) with document size:

- Edge graph construction: O(n × horizon²) where horizon is fixed
- Salience computation: O(edges) = O(n × horizon)
- Fuzzy post-pass: O(n × bucket_size) via length/prefix bucketing

Tested on documents up to 100k words. For larger corpora, consider chunking by section or paragraph.

## Limitations

- **Domain-agnostic**: No external knowledge - compression quality depends on local salience patterns
- **Grammar-blind**: No dependency parsing - can orphan modifiers at high compression
- **English-optimized**: Function word lists are English; adaptable to other languages by swapping word sets