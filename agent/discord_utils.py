import re
import discord
import logging
import asyncio

def sanitize_mentions(content: str, mentions: list) -> str:
    if not content or not mentions:
        return content
    lines = content.split('\n')
    out, in_code = [], False
    punc = ['.', ',', '!', '?', ';', ':', ')', ']', '}', '"', "'"]
    for line in lines:
        if '```' in line: in_code = not in_code
        mm = {}
        users = [m for m in mentions if isinstance(m, (discord.Member, discord.User))]
        roles = [m for m in mentions if isinstance(m, discord.Role)]
        chans = [m for m in mentions if isinstance(m, (discord.TextChannel, discord.ForumChannel, discord.VoiceChannel, discord.CategoryChannel))]
        mm.update({f'<@{m.id}>': (f'@{m.name}' if not in_code else m.name) for m in users})
        mm.update({f'<@!{m.id}>': (f'@{m.name}' if not in_code else m.name) for m in users})
        mm.update({f'<@&{m.id}>': (f'@{m.name}' if not in_code else m.name) for m in roles})
        for m in users:
            for q in punc:
                mm[f'<@{m.id}>{q}'] = (f'@{m.name}{q}' if not in_code else f'{m.name}{q}')
                mm[f'<@!{m.id}>{q}'] = (f'@{m.name}{q}' if not in_code else f'{m.name}{q}')
        for r in roles:
            for q in punc:
                mm[f'<@&{r.id}>{q}'] = (f'@{r.name}{q}' if not in_code else f'{r.name}{q}')
        for c in chans:
            mm[f'<#{c.id}>'] = (f'#{c.name}' if not in_code else c.name)
            for q in punc:
                mm[f'<#{c.id}>{q}'] = (f'#{c.name}{q}' if not in_code else f'{c.name}{q}')
        logging.debug("\n".join([f"Sanitize transform: {k} -> {v}" for k,v in mm.items()]))
        cur = line
        for k in sorted(mm.keys(), key=len, reverse=True):
            cur = cur.replace(k, mm[k])
        out.append(cur)
    res = '\n'.join(out)
    logging.debug(f"Sanitized mentions result: {res[:100]}...")
    return res

def format_discord_mentions(content: str, guild: discord.Guild, mentions_enabled: bool = True, bot=None) -> str:
    if not content:
        return content
    MENTION_RE = re.compile(r'@([\w.\-]+)')  # user handles; roles handled explicitly below
    code_annotations = ['property','staticmethod','classmethod','decorator','param','return','override','abstractmethod']
    for a in code_annotations:
        content = content.replace(f"@{a}", f"__CODE_ANNOTATION_{a}__")
    if not guild:
        logging.info(f"DM message detected - guild is None. Content preview: {content[:100]}...")
        if bot:
            all_members, all_channels = {}, {}
            for g in bot.guilds:
                for m in g.members:
                    all_members[m.name] = m
                    all_members[m.name.lower()] = m
                    if m.name != m.display_name: all_members[m.display_name] = m
                for ch in g.channels:
                    if hasattr(ch,'name'):
                        all_channels[ch.name] = ch
                        all_channels[ch.name.lower()] = ch
            for name in sorted(all_members.keys(), key=len, reverse=True):
                if f"@{name}" in content:
                    content = content.replace(f"@{name}", all_members[name].display_name)
            for name in sorted(all_members.keys(), key=len, reverse=True):
                if name in content and name.lower() not in [a.lower() for a in code_annotations]:
                    content = re.sub(r'\b' + re.escape(name) + r'\b', all_members[name].display_name, content)
            for name in sorted(all_channels.keys(), key=len, reverse=True):
                if f"#{name}" in content:
                    content = content.replace(f"#{name}", f"#{all_channels[name].name}")
        else:
            logging.warning("Bot instance not provided - can't look up users across guilds")
            if not mentions_enabled:
                content = MENTION_RE.sub(r'\1', content)
    else:
        members, channels, roles = {}, {}, {}
        for m in guild.members:
            members[m.name] = m
            members[m.name.lower()] = m
            if m.name != m.display_name: members[m.display_name] = m
        for ch in guild.channels:
            if hasattr(ch,'name'):
                channels[ch.name] = ch
                channels[ch.name.lower()] = ch
        for r in guild.roles:
            roles[r.name] = r
            roles[r.name.lower()] = r
        if mentions_enabled:
            lines, out, in_code = content.split('\n'), [], False
            for line in lines:
                if '```' in line: in_code = not in_code
                cur = line
                spans = []
                while '`' in cur:
                    s = cur.find('`'); 
                    if s == -1: break
                    e = cur.find('`', s+1)
                    if e == -1: break
                    span = cur[s:e+1]; spans.append(span)
                    cur = cur[:s] + f"__INLINE_CODE_{len(spans)-1}__" + cur[e+1:]
                if not in_code:
                    def repl(m):
                        u = m.group(1)
                        if u in members: return f"<@{members[u].id}>"
                        if u.lower() in members: return f"<@{members[u.lower()].id}>"
                        return m.group(0)
                    cur = MENTION_RE.sub(repl, cur)
                    for name, mem in sorted(members.items(), key=lambda x: len(x[0]), reverse=True):
                        if name in cur and name.lower() not in [a.lower() for a in code_annotations]:
                            cur = re.sub(r'\b' + re.escape(name) + r'\b', f"<@{mem.id}>", cur)
                    for name, role in sorted(roles.items(), key=lambda x: len(x[0]), reverse=True):
                        token = f"@{name}"
                        if token in cur:
                            cur = cur.replace(token, f"<@&{role.id}>")
                    for name, ch in sorted(channels.items(), key=lambda x: len(x[0]), reverse=True):
                        if f"#{name}" in cur:
                            cur = cur.replace(f"#{name}", f"<#{ch.id}>")
                for i, sp in enumerate(spans):
                    cur = cur.replace(f"__INLINE_CODE_{i}__", sp)
                out.append(cur)
            content = '\n'.join(out)
        else:
            for name, mem in sorted(members.items(), key=lambda x: len(x[0]), reverse=True):
                if name.lower() in [a.lower() for a in code_annotations]: continue
                if f"@{name}" in content:
                    content = content.replace(f"@{name}", mem.display_name)
            for name, mem in sorted(members.items(), key=lambda x: len(x[0]), reverse=True):
                if name in content and name.lower() not in [a.lower() for a in code_annotations]:
                    content = re.sub(r'\b' + re.escape(name) + r'\b', mem.display_name, content)
            for name, role in sorted(roles.items(), key=lambda x: len(x[0]), reverse=True):
                token = f"@{name}"
                if token in content:
                    content = content.replace(token, role.name)
            ch_mentions = re.findall(r'#([\w.\-_]+)', content)
            for nm in sorted(ch_mentions, key=len, reverse=True):
                if nm in channels:
                    content = content.replace(f"#{nm}", f"#{channels[nm].name}")
    for a in code_annotations:
        content = content.replace(f"__CODE_ANNOTATION_{a}__", f"@{a}")
    logging.info(f"Format mentions result (enabled={mentions_enabled}): {content[:100]}...")
    return content
