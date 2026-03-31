"""
Shared context-building pipeline used by both discord_bot.py and spike.py.

Extracted from discord_bot.py to avoid circular imports (discord_bot imports spike,
so spike can't import discord_bot). Both pipelines now get identical context formatting.
"""

import re
from datetime import datetime
from temporality import TemporalParser
from chunker import truncate_middle
from discord_utils import sanitize_mentions
from hippocampus import Hippocampus, HippocampusConfig
from bot_config import config


async def fetch_history_with_reactions(channel, limit, skip_id=None):
    """fetch history and reaction data without mutating Message objects"""
    msgs = []
    reactions_map = {}  # msg.id -> {emoji: [usernames]}

    async for msg in channel.history(limit=limit):
        if skip_id and msg.id == skip_id:
            continue
        msgs.append(msg)
        if msg.reactions:
            reactions_map[msg.id] = {}
            for rxn in msg.reactions:
                users = [u async for u in rxn.users()]
                reactions_map[msg.id][str(rxn.emoji)] = [u.name for u in users]

    return msgs, reactions_map


def process_history_dual(msgs, reactions_map, temporal_parser, truncation_len, harsh_truncation_len=None):
    """single pass over history → two outputs"""
    simple_lines = []
    formatted_list = []
    trunc_len = harsh_truncation_len or truncation_len

    for msg in msgs:
        name = msg.author.name
        mentions = list(msg.mentions) + list(msg.channel_mentions) + list(msg.role_mentions)
        sanitized = sanitize_mentions(msg.content, mentions)
        simple_lines.append(f"@{name}: {sanitized}")
        truncated = truncate_middle(sanitized, max_tokens=trunc_len)
        local_ts = msg.created_at.astimezone().replace(tzinfo=None)
        ts_str = local_ts.strftime("%H:%M [%d/%m/%y]")
        temporal = temporal_parser.get_temporal_expression(ts_str)
        formatted = f"@{name} ({temporal.base_expression}): {truncated}"

        # look up reactions from separate map
        msg_reactions = reactions_map.get(msg.id, {})
        if msg_reactions:
            rxn_parts = [f"@{u}: {emoji}" for emoji, users in msg_reactions.items() for u in users]
            if rxn_parts:
                formatted += f"\n(Reactions: {' '.join(rxn_parts)})"

        formatted_list.append(formatted)

    simple_lines.reverse()
    formatted_list.reverse()
    return '\n'.join(simple_lines), formatted_list


def build_memory_context(relevant_memories, temporal_parser, truncation_len):
    """format memories with temporal parsing"""
    if not relevant_memories:
        return ""
    ctx = f"{len(relevant_memories)} Potentially Relevant Memories:\n<memories>\n"
    timestamp_pattern = r'\((\d{2}):(\d{2})\s*\[(\d{2}/\d{2}/\d{2})\]\)'
    for memory, score in relevant_memories:
        parsed = re.sub(
            timestamp_pattern,
            lambda m: f"({temporal_parser.get_temporal_expression(datetime.strptime(f'{m.group(1)}:{m.group(2)} {m.group(3)}', '%H:%M %d/%m/%y')).base_expression})",
            memory
        )
        truncated = truncate_middle(parsed, max_tokens=truncation_len)
        ctx += f"[Relevance: {score:.2f}] {truncated}\n"
    ctx += "</memories>\n\n"
    return ctx


def build_conversation_context(formatted_msgs):
    """wrap formatted messages in conversation tags"""
    ctx = "**Ongoing Channel Conversation:**\n\n<conversation>\n"
    for msg in formatted_msgs:
        ctx += f"{msg}\n"
    ctx += "</conversation>\n"
    return ctx


async def get_or_create_hippocampus(bot):
    """lazy init hippocampus per bot instance"""
    if not hasattr(bot, '_hippocampus') or bot._hippocampus is None:
        bot._hippocampus = Hippocampus(HippocampusConfig(blend_factor=config.persona.reranking_blend_factor))
    return bot._hippocampus


async def rerank_if_enabled(bot, candidate_memories, search_query, logger=None):
    """Shared reranking logic — wraps hippocampus reranking with threshold math.

    Returns the (possibly reranked) memory list.
    """
    if not candidate_memories:
        return []

    if config.persona.use_hippocampus_reranking:
        hippocampus = await get_or_create_hippocampus(bot)
        amygdala_scale = bot.amygdala_response / 100.0
        bandwidth = config.persona.hippocampus_bandwidth
        mood_coeff = config.persona.mood_coefficient
        threshold = max(
            config.persona.minimum_reranking_threshold,
            bandwidth - (mood_coeff * amygdala_scale)
        )
        if logger:
            logger.info(
                f"Memory reranking threshold: {threshold:.3f} "
                f"(bandwidth: {bandwidth}, amygdala: {bot.amygdala_response}%, "
                f"influence: {mood_coeff * amygdala_scale:.3f})"
            )
        relevant = await hippocampus.rerank_memories(
            query=search_query,
            memories=candidate_memories,
            threshold=threshold,
            blend_factor=config.persona.reranking_blend_factor
        )
        if logger:
            logger.info(f"Found {len(relevant)} memories above threshold {threshold:.3f}")
        return relevant
    else:
        if logger and candidate_memories:
            logger.info(f"Using memories without reranking: {len(candidate_memories)} memories")
        return candidate_memories
