import asyncio
import random
from collections import defaultdict
import logging
from datetime import datetime
import json
import re
import os
from chunker import truncate_middle, clean_response
from temporality import TemporalParser
from fuzzywuzzy import fuzz
from logger import BotLogger
from bot_config import DMNConfig

class DMNProcessor:
    """
    Default Mode Network (DMN) processor that implements background thought generation
    through random memory walks and associative combination.
    """
    def __init__(self, memory_index, prompt_formats, system_prompts, bot, dmn_config=None, mode="conservative", dmn_api_type=None, dmn_model=None):
        # Core components
        self.memory_index = memory_index
        self.prompt_formats = prompt_formats
        self.system_prompts = system_prompts
        self.bot = bot
        # Use bot's logger
        self.logger = bot.logger if hasattr(bot, 'logger') else logging.getLogger('bot.default')
        # Load DMN configuration
        if dmn_config is None:
            from bot_config import config
            dmn_config = config.dmn
        # Operational settings
        self.tick_rate = dmn_config.tick_rate
        self.enabled = False
        self.task = None
        # Thought generation parameters
        self.temperature = dmn_config.temperature
        self.amygdala_response = 50  # Default intensity
        self.combination_threshold = dmn_config.combination_threshold
        # Memory decay settings
        self.decay_rate = dmn_config.decay_rate
        self.memory_weights = defaultdict(lambda: defaultdict(lambda: 1.0))  # user_id -> memory -> weight
        self.top_k = dmn_config.top_k
        # Retrived Memory Density Temperature Multiplier settings
        self.density_multiplier = dmn_config.density_multiplier
        # Fuzzy matching settings
        self.fuzzy_overlap_threshold = dmn_config.fuzzy_overlap_threshold
        self.fuzzy_search_threshold = dmn_config.fuzzy_search_threshold
        # Memory context compression settings
        self.max_memory_length = dmn_config.max_memory_length
        self.temporal_parser = TemporalParser()  # Add temporal parser instance
        # Search similarity settings
        self.similarity_threshold = dmn_config.similarity_threshold
        # Store modes from config
        self.modes = dmn_config.modes
        # Set initial mode
        self.set_mode(mode)
        # DMN-specific API settings
        self.dmn_api_type = dmn_api_type
        self.dmn_model = dmn_model
        # Consciousness settings
        self.temperature_max=dmn_config.temperature_max
        self.consciousness_state=dmn_config.consciousness_default
        self.consciousness_presets=dmn_config.consciousness_presets

        self.logger.info(f"DMN Processor initialized with API: {dmn_api_type or 'default'}, Model: {dmn_model or 'default'}")

    def set_consciousness(self,name:str): 
        if name in self.consciousness_presets: self.consciousness_state=name

    async def start(self):
        """Start the DMN processing loop."""
        if not self.enabled:
            self.enabled = True
            self.task = asyncio.create_task(self._process_loop())
            self.logger.info("DMN processing loop started")

    async def stop(self):
        """Stop the DMN processing loop."""
        if self.enabled:
            self.enabled = False
            if self.task:
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass
                self.task = None
            self.logger.info("DMN processing loop stopped")

    def set_amygdala_response(self, intensity: int):
        """Update amygdala arousal and temperature scaling."""
        self.amygdala_response = intensity
        self.temperature = intensity / 100.0
        self.logger.info(f"Updated amygdala arousal to {intensity} (temperature: {self.temperature})")

    async def _process_loop(self):
        """Main DMN processing loop."""
        while self.enabled:
            try:
                await self._generate_thought()
            except Exception as e:
                self.logger.error(f"Error in DMN thought generation: {str(e)}")
            await asyncio.sleep(self.tick_rate)

    def _select_random_memory(self):
        """Select a random memory based on contextual weights."""
        user_ids = list(self.memory_index.user_memories.keys())
        if not user_ids:
            return None
        # Use memory count directly as weights (not ranks)
        user_weights = []
        user_memory_counts = []
        for user_id in user_ids:
            memory_count = len(self.memory_index.user_memories[user_id])
            user_weights.append(memory_count)
            user_memory_counts.append((user_id, memory_count))
        # Sort by memory count for logging (highest first)
        user_memory_counts.sort(key=lambda x: x[1], reverse=True)
        # Log top users by memory count with names
        top_users = user_memory_counts[:5]  # Show top 5
        top_users_with_names = []
        for user_id, count in top_users:
            try:
                user = self.bot.get_user(int(user_id))
                user_name = user.name if user else f"Unknown({user_id})"
            except:
                user_name = f"Unknown({user_id})"
            top_users_with_names.append((user_name, count))
        
        self.logger.info(f"User memory ranking - Top users: {top_users_with_names}")
        # Weighted random selection
        total_weight = sum(user_weights)
        if total_weight <= 0:
            selected_user_id = random.choice(user_ids)
        else:
            selection_point = random.uniform(0, total_weight)
            current_weight = 0
            selected_user_id = user_ids[0]  # fallback
            
            for user_id, weight in zip(user_ids, user_weights):
                current_weight += weight
                if current_weight >= selection_point:
                    selected_user_id = user_id
                    break
        user_memories = self.memory_index.user_memories[selected_user_id]
        if not user_memories:
            return None
        # Use user-specific weights
        weighted_memories = [
            (self.memory_index.memories[memory_id], 
             self.memory_weights[selected_user_id][self.memory_index.memories[memory_id]])
            for memory_id in user_memories
            if self.memory_index.memories[memory_id] is not None
        ]
        if not weighted_memories:
            return None
        total_weight = sum(weight for _, weight in weighted_memories)
        if total_weight <= 0:
            return None
        # Random selection based on weights
        selection_point = random.uniform(0, total_weight)
        current_weight = 0
        
        for memory, weight in weighted_memories:
            current_weight += weight
            if current_weight >= selection_point:
                return selected_user_id, memory

        return None

    async def _generate_thought(self):
        """Generate new thought through memory combination and insight generation."""
        max_retries = 8
        for attempt in range(max_retries):
            selection_result = self._select_random_memory()
            if not selection_result:
                return
            
            user_id, seed_memory = selection_result
            
            try:
                user = await self.bot.fetch_user(int(user_id))
                user_name = user.name if user else "Unknown User"
            except Exception as e:
                user_name = "Unknown User"
            # Run memory search in executor to prevent blocking
            try:
                loop = asyncio.get_event_loop()
                related_memories = await loop.run_in_executor(
                    None,
                    lambda: self.memory_index.search(seed_memory, user_id=user_id, k=int(self.top_k))
                )
            except Exception as e:
                continue
            # Filter out the seed memory from results and apply weight threshold
            related_memories = [
                (memory, score) for memory, score in related_memories 
                if memory != seed_memory and score >= self.similarity_threshold
            ]
            # Log similarity threshold filtering results
            self.logger.info(f"After similarity threshold ({self.similarity_threshold}): {len(related_memories)} memories selected")
            # If we found any related memories, we can proceed
            if related_memories:
                break
            self.logger.info(f"Attempt {attempt + 1}: No related memories found, trying another seed memory")
            if attempt == max_retries - 1:
                self.logger.info("Max retries reached without finding any memories")
                return

        # Log DMN process start
        self.logger.log({
            'event': 'dmn_process_start',
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'user_name': user_name,
            'seed_memory': seed_memory,
            'related_memories_count': len(related_memories)
        })

        # Build memory context using ALL related memories
        memory_context = f"{len(related_memories)} Connected memories:\n\n"
        if related_memories:
            for memory, score in sorted(related_memories, key=lambda x: x[1], reverse=True):
                # Convert any timestamp in the memory to temporal expression
                timestamp_pattern = r'\((\d{2}):(\d{2})\s*\[(\d{2}/\d{2}/\d{2})\]\)'
                
                parsed_memory = re.sub(timestamp_pattern, 
                    lambda m: f"({self.temporal_parser.get_temporal_expression(datetime.strptime(f'{m.group(1)}:{m.group(2)} {m.group(3)}', '%H:%M %d/%m/%y')).base_expression})", 
                    memory)
                memory_context += f"{parsed_memory} [Weight: {score:.2f}]\n\n"
        else:
            memory_context += "Hmmm... nothing comes to mind.\n"

        # Get high-similarity memories for term processing
        similar_memories = []
        for memory, score in related_memories:
            # Apply fuzzy matching to memory content
            content_ratio = fuzz.token_sort_ratio(seed_memory, memory)
            if content_ratio >= self.fuzzy_overlap_threshold or score >= self.combination_threshold:
                similar_memories.append((memory, max(score, content_ratio/100.0)))
                self.logger.info(f"Memory matched with fuzzy ratio: {content_ratio}%, semantic score: {score:.2f}")

        # Process overlapping terms if we have high-similarity memories
        if similar_memories:
            # Sort by combined score
            similar_memories.sort(key=lambda x: x[1], reverse=True)
            # Log memory processing
            self.logger.log({
                'event': 'dmn_memory_processing',
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'similar_memories_count': len(similar_memories),
                'combination_threshold': self.combination_threshold,
                'memory_context': memory_context
            })
            # Get memory IDs for memories
            seed_memory_id = self.memory_index.memories.index(seed_memory)
            top_memories = [(seed_memory, seed_memory_id)] + [
                (memory, self.memory_index.memories.index(memory))
                for memory, _ in similar_memories[1:]
            ]
            # Find overlapping terms between seed and each result
            memory_terms_map = {}
            for memory, memory_id in top_memories:
                memory_terms = set(
                    term for term, ids in self.memory_index.inverted_index.items()
                    if memory_id in ids
                )
                memory_terms_map[memory_id] = memory_terms
            # Find terms that overlap between seed and each result
            seed_terms = memory_terms_map[seed_memory_id]
            overlapping_terms = set()
            fuzzy_matches = defaultdict(set)
            # First pass - exact matches
            for memory_id in memory_terms_map:
                if memory_id != seed_memory_id:
                    overlapping_terms.update(seed_terms & memory_terms_map[memory_id])
            # Second pass - fuzzy matches
            for seed_term in seed_terms:
                for memory_id in memory_terms_map:
                    if memory_id != seed_memory_id:
                        for term in memory_terms_map[memory_id]:
                            # Skip if terms are identical (prevent self-matches)
                            if seed_term.lower() == term.lower():
                                continue
                            ratio = fuzz.ratio(seed_term.lower(), term.lower())
                            if ratio >= self.fuzzy_search_threshold:  # Threshold for fuzzy matches
                                fuzzy_matches[seed_term].add(term)
                                overlapping_terms.add(term)

            if overlapping_terms or fuzzy_matches:
                self.logger.info(f"Found {len(overlapping_terms)} exact overlapping terms")
                self.logger.info(f"Found {len(fuzzy_matches)} fuzzy term matches")
                for seed_term, matches in fuzzy_matches.items():
                    self.logger.info(f"Fuzzy matches for '{seed_term}': {', '.join(matches)}")
                
                # Store the state for post-processing
                memory_update_state = {
                    'user_id': user_id,
                    'top_memories': top_memories,
                    'memory_terms_map': memory_terms_map,
                    'overlapping_terms': overlapping_terms
                }

        # Replace timestamp generation with temporal expression
        current_time = datetime.now()
        temporal_expr = self.temporal_parser.get_temporal_expression(current_time)
        timestamp = temporal_expr.base_expression
        if temporal_expr.time_context:
            timestamp = f"{timestamp} in the {temporal_expr.time_context}"

        temporally_parsed_seed_memory = re.sub(r'\((\d{2}):(\d{2})\s*\[(\d{2}/\d{2}/\d{2})\]\)', 
            lambda m: f"({self.temporal_parser.get_temporal_expression(datetime.strptime(f'{m.group(1)}:{m.group(2)} {m.group(3)}', '%H:%M %d/%m/%y')).base_expression})", 
            seed_memory)

        prompt = self.prompt_formats['generate_dmn_thought'].format(
            memory_text=memory_context,
            seed_memory=temporally_parsed_seed_memory,
            timestamp=timestamp,  # Using natural language timestamp
            user_name=user_name,
            amygdala_response=self.amygdala_response
        )
        
        # personality temperature scaling based on memory density relative to top_k
        num_results = len(related_memories)

        if int(self.top_k) > 0:
            # Calculate inverse density: fewer memories = higher multiplier
            # Cap density at 1.0 to prevent going below 0.0
            density = min(1.0, num_results / max(1, int(self.top_k)))
            intensity_multiplier = 1.0 - density
        else:
            # Default to max intensity if top_k is 0 (edge case)
            # Corresponds to density = 0 in the formula
            density = 0.0
            intensity_multiplier = 1.0 
            
        # Calculate final intensity using the exact original clamping/scaling
        new_intensity = min(100, max(0, int(100 * intensity_multiplier)))
        # intensity already computed above as new_intensity and density already computed
        self.amygdala_response=new_intensity
        I=new_intensity/100.0
        self.temperature=0.3+I
        self.bot.amygdala_response=new_intensity
        # Convert intensity to temperature before passing to API client
        self.bot.update_api_temperature(self.temperature)

        top_p_value=0.98 if density<.33 else 0.95 if density<.66 else 0.92
        self.bot.update_api_top_p(top_p_value)

        self.logger.info(f"Updated bot amygdala arousal to {new_intensity} based on memory density")
        self.logger.info(f"Updated bot top_p to {top_p_value:.2f} (banded density mapping)")

        system_prompt = self.system_prompts['dmn_thought_generation'].replace(
            '{amygdala_response}',
            str(self.amygdala_response)
        )
        # truncate the middle of each memory in the memory_context using truncate_middle
        memory_context = "\n\n".join([truncate_middle(memory, self.max_memory_length) for memory in memory_context.split("\n\n")])

        try:
            # Use call_api with override parameters without changing global state
            api_kwargs = {
                'prompt': prompt,
                'system_prompt': system_prompt,
                'temperature': self.temperature
            }
            
            # Only add overrides if they're actually set
            if self.dmn_api_type:
                api_kwargs['api_type_override'] = self.dmn_api_type
            if self.dmn_model:
                api_kwargs['model_override'] = self.dmn_model
            
            new_thought = await self.bot.call_api(**api_kwargs)
            new_thought = clean_response(new_thought)
            
            # Gather unique users from related memories
            memory_users = set()
            for memory, _ in related_memories:
                memory_id = self.memory_index.memories.index(memory)
                memory_user_id = next((uid for uid, mems in self.memory_index.user_memories.items() if memory_id in mems), None)
                if memory_user_id:
                    try:
                        memory_user = await self.bot.fetch_user(int(memory_user_id))
                        if memory_user and memory_user.name != user_name:
                            memory_users.add(memory_user.name)
                    except:
                        continue
            
            # Save generated thought as new memory
            timestamp = datetime.now().strftime('(%H:%M [%d/%m/%y])')
            users_str = f" and {', '.join(memory_users)}" if memory_users else ""
            # Clean username - remove all possible leading Discord role markers
            clean_name = re.sub(r'^[.!~*$]', '', user_name).strip()
            thought_memory = f"Reflections on priors with @{clean_name}{users_str} {timestamp}:\n{new_thought}"
            # Store memory without metadata
            await self.memory_index.add_memory_async(user_id, thought_memory)


            # Process memory weights and cleanup if we have memory state
            if 'memory_update_state' in locals():
                state = memory_update_state
                affected_memories = []
                # First remove overlapping terms
                for memory, memory_id in state['top_memories'][1:]:  # Skip seed memory
                    terms_removed = state['memory_terms_map'][memory_id] & state['overlapping_terms']
                    remaining_terms = state['memory_terms_map'][memory_id] - state['overlapping_terms']
                    if terms_removed:
                        affected_memories.append((memory, terms_removed))
                        self.logger.info(f"Memory [{memory_id}]: Removing terms: {', '.join(terms_removed)}")
                        #self.logger.info(f"Memory [{memory_id}]: Remaining terms: {', '.join(remaining_terms)}")
                        # Update inverted index directly
                        for term in terms_removed:
                            if term in self.memory_index.inverted_index:
                                self.memory_index.inverted_index[term] = [
                                    mid for mid in self.memory_index.inverted_index[term] 
                                    if mid != memory_id
                                ]
                                # Clean up empty terms
                                if not self.memory_index.inverted_index[term]:
                                    del self.memory_index.inverted_index[term]
                                    self.logger.info(f"Removed empty term entry: {term}")

                # Then update weights
                for memory, memory_id in state['top_memories'][1:]:  # Skip seed memory
                    original_terms = state['memory_terms_map'][memory_id]
                    removed_terms = len(original_terms & state['overlapping_terms'])
                    if len(original_terms) > 0:
                        decay = removed_terms / len(original_terms)
                        # Use user-specific weight decay
                        self.memory_weights[user_id][memory] *= (1 - (self.decay_rate * decay))
                        #self.logger.info(f"Memory weight updated for user {user_id}: {memory[:50]}... (decay: {decay:.2f})")
                #self.memory_index.save_cache()
                self.memory_index._saver.request()
                self.logger.info(f"Updated memory cache after pruning {len(affected_memories)} memories")

            # Add cleanup here after new memory addition and weight updates
            self._cleanup_disconnected_memories()
            # Log successful thought generation
            self.logger.log({
                'event': 'dmn_thought_generated',
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'user_name': user_name,
                'seed_memory': seed_memory,
                'system_prompt': system_prompt,
                'prompt': prompt,
                'generated_thought': new_thought,
                'amygdala_response': self.amygdala_response,
                'temperature': self.temperature
            })
            
        except Exception as e:
            error_msg = f"Failed to generate DMN thought: {str(e)}"
            self.logger.error(error_msg)
            # Log error in thought generation
            self.logger.log({
                'event': 'dmn_thought_error',
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'user_name': user_name,
                'error': str(e)
            })

    def _cleanup_disconnected_memories(self):
        connected=set(); [connected.update(v) for v in self.memory_index.inverted_index.values()]
        for uid,mems in list(self.memory_index.user_memories.items()):
            disc=sorted([i for i in mems if i not in connected])
            if not disc: continue
            texts=[self.memory_index.memories[i] for i in disc]
            removed=set(disc)
            old_mem=self.memory_index.memories
            self.memory_index.memories=[m for i,m in enumerate(old_mem) if i not in removed]
            def remap(i):
                c=0
                for d in disc:
                    if d<i: c+=1
                    else: break
                return i-c
            for u,ls in list(self.memory_index.user_memories.items()):
                new_ls=[remap(i) for i in ls if i not in removed]
                if new_ls: self.memory_index.user_memories[u]=new_ls
                else: self.memory_index.user_memories.pop(u,None)
            for w,ls in list(self.memory_index.inverted_index.items()):
                nl=[remap(i) for i in ls if i not in removed]
                if nl: self.memory_index.inverted_index[w]=nl
                else: self.memory_index.inverted_index.pop(w,None)
            for u,weights in list(self.memory_weights.items()):
                self.memory_weights[u]={remap(k):v for k,v in weights.items() if k not in removed}
            self.logger.info(f"Cleaned up {len(disc)} disconnected memories for user {uid}")
            self.logger.info(f"Disconnected memories: {texts}")
            self.logger.log({'event':'dmn_memory_cleanup','timestamp':datetime.now().isoformat(),'user_id':uid,'disconnected_memories':texts})


    def set_mode(self, mode):
        """Update DMN parameters based on mode."""
        m = {k.lower(): k for k in self.modes}
        key = str(mode).strip().lower()
        if key not in m:
            raise ValueError(f"unknown mode: {mode}")
        p = self.modes[m[key]]
        self.combination_threshold = float(p["combination_threshold"])
        self.similarity_threshold = float(p["similarity_threshold"])
        self.decay_rate = float(p["decay_rate"])
        self.top_k = int(p["top_k"])
        self.fuzzy_overlap_threshold = int(p["fuzzy_overlap_threshold"])
        self.fuzzy_search_threshold = int(p["fuzzy_search_threshold"])