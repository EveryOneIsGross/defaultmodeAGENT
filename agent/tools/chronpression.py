from collections import defaultdict
from functools import lru_cache
from fuzzywuzzy import fuzz
import re
import os
import sys
import argparse
import math

COMMON_WORDS = {
    "the": 0.0616, "of": 0.0292, "and": 0.0284, "to": 0.0275, "a": 0.0235,
    "in": 0.0234, "is": 0.0119, "that": 0.0109, "for": 0.0107, "it": 0.0099,
    "with": 0.0095, "as": 0.0093, "was": 0.0087, "be": 0.0083, "on": 0.0075,
    "not": 0.0072, "he": 0.0069, "by": 0.0064, "are": 0.0062, "this": 0.0061,
    "at": 0.0056, "from": 0.0055, "but": 0.0053, "have": 0.0052, "an": 0.0049,
    "they": 0.0048, "which": 0.0047, "or": 0.0046, "his": 0.0045, "had": 0.0043,
    "we": 0.0042, "there": 0.0041, "can": 0.0040, "were": 0.0039, "been": 0.0038,
    "has": 0.0037, "their": 0.0035, "more": 0.0035, "will": 0.0034, "would": 0.0034,
    "about": 0.0033, "if": 0.0033, "no": 0.0032, "when": 0.0032, "who": 0.0031,
    "so": 0.0031, "all": 0.0030, "she": 0.0029, "you": 0.0027, "said": 0.0025,
}

ARTICLES = {'the', 'a', 'an'}
COPULAS = {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am'}
PREPOSITIONS = {
    'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'about',
    'over', 'into', 'onto', 'upon', 'through', 'during', 'before', 'after',
    'between', 'among', 'against', 'within', 'without', 'under', 'above', 'below'
}
PRONOUNS = {'he', 'she', 'it', 'they', 'we', 'i', 'you', 'my', 'your', 'our', 'their', 'his', 'her'}
CONJUNCTIONS = {'and', 'but', 'or', 'nor', 'yet', 'so', 'for', 'because', 'although', 'while', 'if', 'when', 'than'}
FUNCTION_WORDS = ARTICLES | COPULAS | PREPOSITIONS | PRONOUNS | CONJUNCTIONS

ANCHOR_VERBS = {
    'anchor', 'require', 'imply', 'assume', 'convey', 'remain', 'preserve',
    'matter', 'mean', 'learn', 'compress', 'remove', 'keep', 'get', 'make',
    'show', 'prove', 'demonstrate', 'indicate', 'suggest', 'reveal'
}

ADJECTIVES = {
    'short', 'high', 'dense', 'novel', 'simple', 'entire', 'internal', 'careful',
    'semantic', 'compressed', 'higher', 'assumed', 'learned', 'familiar', 'important'
}

FUNCTION_WORD_LIST = list(FUNCTION_WORDS)
COMMON_WORD_LIST = list(COMMON_WORDS.keys())

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*\Z")
ANCHOR_RE = re.compile(r'^[A-Z][a-z]+$|^\d+[\d,.]*$|^[A-Z]{2,}$')
WINDOW = 40
PIDGIN_THRESHOLD = 0.8

HTML_ENTITIES = {
    '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>',
    '&quot;': '"', '&apos;': "'", '&#39;': "'", '&#x27;': "'",
    '&ndash;': '-', '&mdash;': '-', '&hellip;': '...',
    '&lsquo;': "'", '&rsquo;': "'", '&ldquo;': '"', '&rdquo;': '"',
    '&bull;': '', '&middot;': '', '&copy;': '', '&reg;': '',
    '&trade;': '', '&times;': 'x', '&divide;': '/',
}

def clean_input(text):
    for entity, replacement in HTML_ENTITIES.items():
        text = text.replace(entity, replacement)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#?\w+;', ' ', text)
    text = re.sub(r'&;+', ' ', text)
    text = re.sub(r'&+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    #text = re.sub(r'\[MUSIC[^\]]*\]', ' ', text, flags=re.IGNORECASE)
    #text = re.sub(r'\[[A-Z]+:[^\]]*\]', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:\'"()\-–—]', ' ', text)
    text = re.sub(r'\.{4,}', '...', text)
    text = re.sub(r'-{3,}', '--', text)
    text = re.sub(r'([,;:!?])\1+', r'\1', text)
    text = re.sub(r'\s*-\s*-\s*', ' - ', text)
    text = re.sub(r"'+", "'", text)
    text = re.sub(r'"+', '"', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_word(t):
    return bool(WORD_RE.fullmatch(t))

def is_anchor(t):
    return bool(ANCHOR_RE.match(t))

def tokenize_text(text):
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*|[^\w\s]|\s+", text)

def pidgin_factor(compression):
    if compression <= PIDGIN_THRESHOLD:
        return 0.0
    return min(1.0, (compression - PIDGIN_THRESHOLD) / (1.0 - PIDGIN_THRESHOLD))

@lru_cache(maxsize=None)
def fuzzy_function_score(word, threshold=80):
    word = word.lower()
    if word in FUNCTION_WORDS:
        return 1.0
    if len(word) < 2:
        return 0.0
    best = 0
    for fw in FUNCTION_WORD_LIST:
        score = fuzz.ratio(word, fw)
        if score > best:
            best = score
        if best >= 95:
            break
    return best / 100.0 if best >= threshold else 0.0

@lru_cache(maxsize=None)
def fuzzy_common_score(word, threshold=75):
    word = word.lower()
    if word in COMMON_WORDS:
        return COMMON_WORDS[word]
    if len(word) < 2:
        return 0.0
    best_score = 0
    best_freq = 0.0
    for cw, freq in COMMON_WORDS.items():
        score = fuzz.ratio(word, cw)
        if score > best_score:
            best_score = score
            best_freq = freq
    return best_freq if best_score >= threshold else 0.0

def length_bucket(word):
    n = len(word)
    if n <= 3:
        return 'xs'
    if n <= 5:
        return 's'
    if n <= 8:
        return 'm'
    if n <= 12:
        return 'l'
    return 'xl'

def prefix_key(word, n=3):
    return word[:n].lower() if len(word) >= n else word.lower()

def remove_consecutive_duplicates(text):
    return re.sub(r'\b(\w+)\b(\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)

def collapse_ngrams(tokens):
    ngram_patterns = {
        r'\bin\s+order\s+to\b': 'to',
        r'\bdue\s+to\s+the\s+fact\s+that\b': 'because',
        r'\bat\s+this\s+point\s+in\s+time\b': 'now',
        r'\bfor\s+the\s+purpose\s+of\b': 'to',
        r'\bin\s+spite\s+of\s+the\s+fact\s+that\b': 'although',
        r'\bby\s+means\s+of\b': 'via',
        r'\bwith\s+regard\s+to\b': 'regarding',
        r'\bin\s+the\s+event\s+that\b': 'if',
        r'\bprior\s+to\b': 'before',
        r'\bsubsequent\s+to\b': 'after',
        r'\bin\s+addition\s+to\b': 'besides',
        r'\bas\s+a\s+result\s+of\b': 'because',
    }
    text = ''.join(tokens)
    for pattern, replacement in ngram_patterns.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return tokenize_text(text)

def extract_sentences(text):
    parts = re.split(r'([.!?]+)', text)
    out = []
    for i in range(0, len(parts) - 1, 2):
        if parts[i].strip():
            out.append(parts[i].strip() + (parts[i + 1] if i + 1 < len(parts) else '.'))
    if len(parts) % 2 == 1 and parts[-1].strip():
        out.append(parts[-1].strip())
    return out

def build_edge_graph(content_words, horizon=6, decay=0.7):
    edges = defaultdict(float)
    n = len(content_words)
    for i in range(n):
        for j in range(i + 1, min(i + horizon, n)):
            dist = j - i
            weight = decay ** (dist - 1)
            pair = tuple(sorted([content_words[i], content_words[j]]))
            edges[pair] += weight
    return edges

def get_protected_bigrams(edges, threshold=1.5, compression=0.5):
    pf = pidgin_factor(compression)
    adjusted_threshold = threshold + (pf * 10.0)
    protected = set()
    for (w1, w2), weight in edges.items():
        if weight >= adjusted_threshold:
            protected.add((w1, w2))
            protected.add((w2, w1))
    return protected

def compute_node_salience(content_words, edges, compression=0.5):
    pf = pidgin_factor(compression)
    edge_weight_factor = 1.0 - (pf * 0.7)
    salience = defaultdict(float)
    for (w1, w2), weight in edges.items():
        novelty = 0
        if w1 not in COMMON_WORDS:
            novelty += 1
        if w2 not in COMMON_WORDS:
            novelty += 1
        boost = weight * (1.0 + novelty * 0.5) * edge_weight_factor
        salience[w1] += boost
        salience[w2] += boost
    for w in set(content_words):
        base = 1.0 if w in COMMON_WORDS else 5.0
        salience[w] = base + salience.get(w, 0.0) * 0.1
    return salience

def compute_edge_scores(edges, node_salience, compression=0.5):
    pf = pidgin_factor(compression)
    edge_scores = {}
    for (w1, w2), freq in edges.items():
        s1 = node_salience.get(w1, 1.0)
        s2 = node_salience.get(w2, 1.0)
        score = freq * math.sqrt(s1 * s2)
        score *= (1.0 - pf * 0.8)
        edge_scores[(w1, w2)] = score
    return edge_scores

def is_noun_after_adjective(words, idx):
    if idx == 0:
        return False
    prev = words[idx - 1].lower()
    return prev in ADJECTIVES

def compute_word_weight(word, idx, words, base_salience, node_salience, compression):
    wl = word.lower()
    pf = pidgin_factor(compression)
    if wl in ARTICLES:
        if pf > 0.3:
            return 0.01
        return 0.1
    if wl in FUNCTION_WORDS:
        penalty = 0.1 * (1.0 - pf * 0.5)
        return penalty
    weight = base_salience
    verb_boost = 1.8 * (1.0 - pf * 0.4)
    if wl in ANCHOR_VERBS:
        weight *= verb_boost
    adj_noun_boost = 1.5 * (1.0 - pf * 0.3)
    if is_noun_after_adjective(words, idx):
        weight *= adj_noun_boost
    anchor_boost = 50.0 * (1.0 - pf * 0.2)
    if is_anchor(word):
        weight += anchor_boost
    position_start = 1.6 - (pf * 0.4)
    position_end = 1.3 - (pf * 0.2)
    if idx == 0:
        weight *= position_start
    if idx == len(words) - 1:
        weight *= position_end
    return weight

def extract_spanning_path(words, edge_scores, keep_ratio, node_salience, protected_bigrams, compression):
    if len(words) <= 2:
        return set(range(len(words)))
    pf = pidgin_factor(compression)
    word_weights = {}
    for i, w in enumerate(words):
        wl = w.lower()
        base = node_salience.get(wl, 1.0)
        edge_contrib = sum(
            score for (w1, w2), score in edge_scores.items()
            if wl == w1 or wl == w2
        )
        base_salience = base + edge_contrib
        word_weights[i] = compute_word_weight(w, i, words, base_salience, node_salience, compression)
    ranked = sorted(range(len(words)), key=lambda i: word_weights[i], reverse=True)
    num_keep = max(1, int(len(words) * keep_ratio))
    kept = set(ranked[:num_keep])
    if pf < 0.5:
        kept.add(0)
    if pf < 0.7:
        kept.add(len(words) - 1)
    if pf < 0.8:
        wl_list = [w.lower() for w in words]
        for i in range(len(words) - 1):
            pair = (wl_list[i], wl_list[i + 1])
            if pair in protected_bigrams:
                if i in kept or i + 1 in kept:
                    kept.add(i)
                    kept.add(i + 1)
    return kept

def sentence_density(sentence, avg_len):
    words = [t for t in tokenize_text(sentence) if is_word(t)]
    wlen = len(words)
    if wlen == 0:
        return 1.0
    content = sum(1 for w in words if w.lower() not in FUNCTION_WORDS)
    content_ratio = content / wlen
    length_factor = wlen / avg_len if avg_len > 0 else 1.0
    return content_ratio * math.log1p(length_factor)

def adaptive_compression(base_compression, density, global_density):
    if global_density == 0:
        return base_compression
    ratio = density / global_density
    if ratio > 1.2:
        return base_compression * 0.7
    elif ratio < 0.8:
        return min(base_compression * 1.3, 0.95)
    return base_compression

def rebuild_text(tokens):
    out = []
    for t in tokens:
        if not t:
            continue
        if t.isspace():
            if out and not out[-1].endswith(' '):
                out.append(' ')
        else:
            out.append(t)
    return ''.join(out).strip()

def compress_sentence(sentence, edge_scores, node_salience, keep_ratio, protected_bigrams, compression):
    tokens = tokenize_text(sentence)
    word_indices = []
    words = []
    for i, t in enumerate(tokens):
        if is_word(t):
            word_indices.append(i)
            words.append(t)
    if len(words) <= 2:
        pf = pidgin_factor(compression)
        if pf > 0.5 and len(words) == 2:
            content = [w for w in words if w.lower() not in FUNCTION_WORDS]
            if content:
                return ' '.join(content)
        return sentence.strip()
    kept_positions = extract_spanning_path(
        words, edge_scores, keep_ratio, node_salience, protected_bigrams, compression
    )
    filtered = tokens[:]
    for local_idx, tok_idx in enumerate(word_indices):
        if local_idx not in kept_positions:
            filtered[tok_idx] = ""
    text = rebuild_text(filtered)
    text = re.sub(r'\s+([,;:.!?])', r'\1', text)
    result = text.strip()
    content_words = [w for w in tokenize_text(result) if is_word(w) and w.lower() not in FUNCTION_WORDS]
    if len(content_words) < 2 and pidgin_factor(compression) > 0.5:
        return ""
    return result

def fuzzy_post_compress(text, fuzzy_strength, compression):
    if fuzzy_strength <= 0:
        return text
    tokens = tokenize_text(text)
    word_indices = []
    words = []
    for i, t in enumerate(tokens):
        if is_word(t):
            word_indices.append(i)
            words.append(t)
    if len(words) <= 3:
        return text
    pf = pidgin_factor(compression)
    base_threshold = max(45, int(85 - compression * 40 * fuzzy_strength - pf * 15))
    drops = set()
    seen_by_bucket = defaultdict(dict)
    for idx, w in enumerate(words):
        wl = w.lower()
        if wl in FUNCTION_WORDS:
            continue
        if is_anchor(w) and pf < 0.5:
            continue
        if idx == 0 and pf < 0.5:
            continue
        if idx == len(words) - 1 and pf < 0.7:
            continue
        if wl in ANCHOR_VERBS and pf < 0.6:
            continue
        func_score = fuzzy_function_score(wl, threshold=base_threshold)
        if func_score >= 0.7:
            drops.add(idx)
            continue
        common_freq = fuzzy_common_score(wl, threshold=base_threshold)
        if common_freq > 0.02:
            drop_prob = common_freq * 10 * fuzzy_strength * compression * (1.0 + pf)
            if drop_prob > 0.5:
                drops.add(idx)
                continue
        bucket = length_bucket(wl)
        prefix = prefix_key(wl)
        bucket_dict = seen_by_bucket[bucket]
        candidates = bucket_dict.get(prefix, [])
        matched = False
        for prev_word, prev_idx in candidates:
            if idx - prev_idx >= WINDOW:
                continue
            sim = fuzz.ratio(wl, prev_word)
            if sim >= base_threshold and sim < 100:
                drops.add(idx)
                matched = True
                break
        if not matched:
            if prefix not in bucket_dict:
                bucket_dict[prefix] = []
            bucket_dict[prefix].append((wl, idx))
    max_drops = int(len(words) * fuzzy_strength * compression * 0.4 * (1.0 + pf * 0.5))
    drop_list = sorted(drops)[:max_drops]
    filtered = tokens[:]
    for local_idx in drop_list:
        tok_idx = word_indices[local_idx]
        filtered[tok_idx] = ""
    return rebuild_text(filtered)

def window_dedupe(text, window=WINDOW):
    tokens = tokenize_text(text)
    last_pos = {}
    out = []
    for i, t in enumerate(tokens):
        if is_word(t):
            k = t.lower()
            if k in last_pos and i - last_pos[k] < window and k not in FUNCTION_WORDS:
                out.append("")
            else:
                last_pos[k] = i
                out.append(t)
        else:
            out.append(t)
    return rebuild_text(out)

def clean_punctuation(text):
    tokens = tokenize_text(text)
    cleaned = []
    for i, t in enumerate(tokens):
        if t in {"'", '"'}:
            prev_word = i > 0 and is_word(tokens[i - 1])
            next_word = i + 1 < len(tokens) and is_word(tokens[i + 1])
            if not (prev_word or next_word):
                continue
        cleaned.append(t)
    s = rebuild_text(cleaned)
    s = re.sub(r'\s+([,;:.!?])', r'\1', s)
    s = re.sub(r'([,;:])([.?!])', r'\2', s)
    s = re.sub(r'([,;:.!?]){2,}', r'\1', s)
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip()

def clean_output(text):
    text = re.sub(r'\s*,\s*,+', ',', text)
    text = re.sub(r'\s*;\s*;+', ';', text)
    text = re.sub(r'\s+-\s*-+\s*', ' - ', text)
    text = re.sub(r'([.!?])\s*([.!?])+', r'\1', text)
    text = re.sub(r',\s*\.', '.', text)
    text = re.sub(r';\s*\.', '.', text)
    text = re.sub(r':\s*\.', '.', text)
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'"\s*"', '', text)
    text = re.sub(r"'\s*'", '', text)
    text = re.sub(r'\s+([,;:.!?])', r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text)
    sentences = text.split('. ')
    cleaned = []
    for s in sentences:
        s = s.strip()
        if s and len(s) > 1:
            words = [w for w in s.split() if len(w) > 0]
            if len(words) >= 1:
                cleaned.append(' '.join(words))
    return '. '.join(cleaned)

def chronomic_filter(text, compression=0.5, fuzzy_strength=1.0, horizon=6):
    text = clean_input(text)
    tokens = collapse_ngrams(tokenize_text(text))
    text = "".join(tokens)
    all_words = [t.lower() for t in tokenize_text(text) if is_word(t)]
    content_words = [w for w in all_words if w not in FUNCTION_WORDS]
    edges = build_edge_graph(content_words, horizon=horizon)
    protected_bigrams = get_protected_bigrams(edges, threshold=1.5, compression=compression)
    node_salience = compute_node_salience(content_words, edges, compression=compression)
    edge_scores = compute_edge_scores(edges, node_salience, compression=compression)
    sentences = extract_sentences(text)
    if not sentences:
        return text
    word_counts = [len([t for t in tokenize_text(s) if is_word(t)]) for s in sentences]
    avg_len = sum(word_counts) / len(word_counts) if word_counts else 1.0
    densities = [sentence_density(s, avg_len) for s in sentences]
    global_density = sum(densities) / len(densities) if densities else 1.0
    compressed = []
    for s, d in zip(sentences, densities):
        local_compression = adaptive_compression(compression, d, global_density)
        keep_ratio = 1.0 - local_compression
        c = compress_sentence(s, edge_scores, node_salience, keep_ratio, protected_bigrams, compression)
        if c:
            compressed.append(c)
    out = " ".join(compressed)
    out = remove_consecutive_duplicates(out)
    out = window_dedupe(out, WINDOW)
    out = fuzzy_post_compress(out, fuzzy_strength, compression)
    out = clean_punctuation(out)
    out = clean_output(out)
    return out

def stats(original, compressed):
    orig_words = len([t for t in tokenize_text(original) if is_word(t)])
    comp_words = len([t for t in tokenize_text(compressed) if is_word(t)])
    ratio = 1.0 - (comp_words / orig_words) if orig_words > 0 else 0.0
    return {
        'original_words': orig_words,
        'compressed_words': comp_words,
        'actual_compression': f"{ratio:.1%}",
        'original_chars': len(original),
        'compressed_chars': len(compressed),
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Chronomic Text Filter")
    p.add_argument("-i", "--input")
    p.add_argument("-o", "--output")
    p.add_argument("-c", "--compression", type=float, default=0.5)
    p.add_argument("-f", "--fuzzy", type=float, default=1.0)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()
    if a.input:
        with open(a.input, "r", encoding="utf-8") as f:
            txt = f.read()
        res = chronomic_filter(txt, a.compression, a.fuzzy, a.horizon)
        if not a.output:
            b, e = os.path.splitext(a.input)
            a.output = f"{b}_chronomic{e}"
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(res)
        if a.verbose:
            st = stats(txt, res)
            print(f"wrote {a.output}")
            print(f"  {st['original_words']} -> {st['compressed_words']} words ({st['actual_compression']})")
    else:
        demo = """
        The concept of Chronomic Compression implies that a compressed sentence can still convey the entire meaning, by simple compression heuristics. Using sparsegrams to learn an input text, we can optimise understanding by compressing it at a set ratio. Chronomic compression assumes a high order of importance that gets learned through analysis, like a higher and higher order of abstraction through removing the assumed roads of language while keeping learned novel entities by association. Short statements matter. They anchor meaning. Dense elaborations require more careful preservation of their internal semantic pathways to remain reconstructible by a reader familiar with the domain.
        """
        print("=== original ===")
        print(demo.strip())
        for c in [0.5, 0.7, 0.85, 0.95]:
            for f in [0.0, 1.0, 2.0]:
                result = chronomic_filter(demo, compression=c, fuzzy_strength=f)
                st = stats(demo, result)
                pf = pidgin_factor(c)
                print(f"\n=== c={c} fuzzy={f} pidgin={pf:.2f} ({st['actual_compression']}) ===")
                print(result)