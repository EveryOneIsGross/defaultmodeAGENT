import asyncio
import random
from collections import defaultdict
import logging
from datetime import datetime
from api_client import call_api, update_api_temperature
import json
import re
from chunker import truncate_middle, clean_response
from temporality import TemporalParser
from fuzzywuzzy import fuzz
from discord_utils import strip_role_prefixes
from logger import BotLogger

class DMNProcessor:
    """
    Default Mode Network (DMN) processor that implements background thought generation
    through random memory walks and associative combination.
    """
    def __init__(self, memory_index, prompt_formats, system_prompts, bot, tick_rate=300, mode="forgetful"):
        # Core components
        self.memory_index = memory_index
        self.prompt_formats = prompt_formats
        self.system_prompts = system_prompts
        self.bot = bot
        
        # Use bot's logger
        self.logger = bot.logger if hasattr(bot, 'logger') else logging.getLogger('bot.default')
        
        # API inference settings

        # Operational settings
        self.tick_rate = tick_rate  # Time between thought generations
        self.enabled = False
        self.task = None
        
        # Thought generation parameters
        self.temperature = 0.7  # Base creative temperature
        self.amygdala_response = 50  # Default intensity
        self.combination_threshold = 0.2  # Minimum relevance score for memory combinations
        
        # Memory decay settings
        self.decay_rate = 0.0  # Rate at which used memory weights decrease, this stays in memory but isn't persisted between sessions
        self.memory_weights = defaultdict(lambda: defaultdict(lambda: 1.0))  # user_id -> memory -> weight
        self.top_k = 16  # Top k memories to consider for combination

        # Fuzzy matching settings
        self.fuzzy_overlap_threshold = 60  # Minimum fuzzy overlap threshold for memory combination
        self.fuzzy_search_threshold = 80  # Minimum fuzzy search threshold for term matching
        
        # Memory context compression settings
        self.max_memory_length = 64  # Maximum length of a memory to display this is for formatting only it seems atm
        
        self.temporal_parser = TemporalParser()  # Add temporal parser instance
        
        # Mode presets
        self.modes = {
            "forgetful": {
                "combination_threshold": 0.1,  # Lower threshold = more memories combined
                "decay_rate": 0.8,            # High decay = aggressive forgetting
                "top_k": 64                  # More memories considered
            },
            "homeostatic": {
                "combination_threshold": 0.2,  # Balanced threshold
                "decay_rate": 0.1,            # Moderate decay
                "top_k": 16                  # Default memory window
            },
            "conservative": {
                "combination_threshold": 0.3,  # Higher threshold = fewer combinations
                "decay_rate": 0.05,           # Very slow decay
                "top_k": 8                   # Fewer memories considered
            }
        }
        
        # Set initial mode
        self.set_mode(mode)
        
        self.logger.info("DMN Processor initialized")

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

        selected_user_id = random.choice(user_ids)
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
                user_name = strip_role_prefixes(user.name) if user else "Unknown User"
            except Exception as e:
                self.logger.error(f"Failed to fetch username for {user_id}: {str(e)}")
                user_name = "Unknown User"
            
            # Query related memories
            related_memories = self.memory_index.search(
                seed_memory,
                user_id=user_id,
                k=self.top_k
            )
            
            # Filter out the seed memory from results
            related_memories = [
                (memory, score) for memory, score in related_memories 
                if memory != seed_memory
            ]
            
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
        memory_context = f"Considering {len(related_memories)} thoughts:\n\n"
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

            # Get memory IDs for memories we'll use
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
                
                # Remove overlapping terms from results (preserve in seed)
                affected_memories = []
                for memory, memory_id in top_memories[1:]:  # Skip seed memory
                    terms_removed = memory_terms_map[memory_id] & overlapping_terms
                    remaining_terms = memory_terms_map[memory_id] - overlapping_terms
                    if terms_removed:
                        affected_memories.append((memory, terms_removed))
                        self.logger.info(f"Memory [{memory_id}]: Removing terms: {', '.join(terms_removed)}")
                        self.logger.info(f"Memory [{memory_id}]: Remaining terms: {', '.join(remaining_terms)}")
                        
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

                # Update memory weights for affected results
                for memory, memory_id in top_memories[1:]:  # Skip seed memory
                    original_terms = memory_terms_map[memory_id]
                    removed_terms = len(original_terms & overlapping_terms)
                    if len(original_terms) > 0:
                        decay = removed_terms / len(original_terms)
                        # Use user-specific weight decay
                        self.memory_weights[user_id][memory] *= (1 - (self.decay_rate * decay))
                        self.logger.info(f"Memory weight updated for user {user_id}: {memory[:50]}... (decay: {decay:.2f})")

                # Save the updated index

                #self.memory_index.save_cache()
                self.logger.info(f"Updated memory cache after pruning {len(affected_memories)} memories")
                
                # Add cleanup here after pruning
                # self._cleanup_disconnected_memories()

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
            user_name=user_name
        )
        
        # Add personality temperature scaling based on memory density
        new_intensity = min(100, max(1, int(50 * max(0.4, 1.0 - (min(len(related_memories), 20) / 20) * 0.6))))
        
        # Update both DMN and global state
        self.amygdala_response = new_intensity
        self.temperature = new_intensity / 100.0
        self.bot.amygdala_response = new_intensity  # Update bot's amygdala arousal
        
        # Update global API temperature through bot's update_temperature function
        update_api_temperature(new_intensity)
        
        self.logger.info(f"Updated global amygdala arousal to {new_intensity} based on memory density")

        system_prompt = self.system_prompts['dmn_thought_generation'].replace(
            '{amygdala_response}',
            str(self.amygdala_response)
        )
        
        # truncate the middle of each memory in the memory_context using truncate_middle
        memory_context = "\n\n".join([truncate_middle(memory, self.max_memory_length) for memory in memory_context.split("\n\n")])

        try:
            new_thought = await call_api(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=self.temperature
            )
            
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
            self.memory_index.add_memory(user_id, thought_memory)
            
            # Add cleanup here after new memory addition
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
        """Remove memories that have no keyword associations in the inverted index."""
        # Get all memory IDs that appear in the inverted index
        connected_memories = set()
        for term_memories in self.memory_index.inverted_index.values():
            connected_memories.update(term_memories)
        
        # Find disconnected memories for each user
        for user_id, memories in list(self.memory_index.user_memories.items()):
            disconnected = sorted([mid for mid in memories if mid not in connected_memories], reverse=True)
            if disconnected:
                # Remove disconnected memories from weights
                for memory_id in disconnected:
                    if memory_id in self.memory_weights[user_id]:
                        del self.memory_weights[user_id][memory_id]
                
                # Remove from main memories list and adjust indices
                for memory_id in disconnected:
                    self.memory_index.memories.pop(memory_id)
                    # Update all higher indices in user_memories
                    for uid, mems in self.memory_index.user_memories.items():
                        self.memory_index.user_memories[uid] = [
                            mid if mid < memory_id else mid - 1 
                            for mid in mems 
                            if mid != memory_id
                        ]
                    
                    # Update inverted index
                    for word in list(self.memory_index.inverted_index.keys()):
                        self.memory_index.inverted_index[word] = [
                            mid if mid < memory_id else mid - 1 
                            for mid in self.memory_index.inverted_index[word] 
                            if mid != memory_id
                        ]
                        # Clean up empty word entries
                        if not self.memory_index.inverted_index[word]:
                            del self.memory_index.inverted_index[word]
                
                # Remove users with no memories left
                if not self.memory_index.user_memories[user_id]:
                    del self.memory_index.user_memories[user_id]
                
                # print the disconnected memories
                self.logger.info(f"Cleaned up {len(disconnected)} disconnected memories for user {user_id}")
                
                #format the disconnected memories for logging
                disconnected_memories = [self.memory_index.memories[mid] for mid in disconnected]
                
                # Save the cleaned up state
                self.memory_index.save_cache()
                
                # print the disconnected memories
                self.logger.info(f"Disconnected memories: {disconnected_memories}")

                # Log the cleanup event
                self.logger.log({
                    'event': 'dmn_memory_cleanup',
                    'timestamp': datetime.now().isoformat(),
                    'user_id': user_id,
                    'disconnected_memories': disconnected_memories
                })

    def set_mode(self, mode):
        """Update DMN parameters based on mode."""
        if mode in self.modes:
            params = self.modes[mode]
            self.combination_threshold = params["combination_threshold"]
            self.decay_rate = params["decay_rate"]
            self.top_k = params["top_k"]
        
