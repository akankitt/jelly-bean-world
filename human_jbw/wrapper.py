"""
ContextualVolatilityWrapper for JBW

This wrapper implements the "Contextual Volatility Foraging" task on top of JBW
without modifying the base JBW code. It adds:

1. PROBABILISTIC WEATHER SWITCHING
   - Context switches randomly (not step-counting based)
   - Prevents agents/humans from memorizing switch times
   - ~1.5% chance per step to switch between BLUE (Safe) and RED (Risky)

2. ITEM LIFECYCLE MANAGEMENT
   - Each item: GREEN (RAW) -> RED (RIPE) -> BROWN (ROTTEN)
   - State transitions are probabilistic and context-dependent
   - GREEN beans: worth 0 pts (collect too early = waste)
   - RED beans: worth +10 pts (collect at right time = reward)
   - BROWN beans: worth -10 pts (collect too late = penalty)

3. CONTEXTUAL TRANSITION PROBABILITIES
   - BLUE (Safe & Stable): High ripen (0.2), low rot (0.01) - Patience rewarded
   - RED (Risky & Volatile): Medium ripen (0.2), high rot (0.30) - Speed rewarded

ARCHITECTURE:
- The wrapper tracks bean states in parallel with JBW's item system
- JBW still manages physics, collision, and visual rendering
- Wrapper applies state-based reward modifications on collection
- Game interface uses state info for color visualization

USAGE:
    import gym
    base_env = gym.make('JBW-v0')
    wrapper = ContextualVolatilityWrapper(base_env)
    obs, context = wrapper.reset()
    
    for _ in range(1000):
        action = base_env.action_space.sample()
        obs, reward, done, info = wrapper.step(action)
        # info['weather']: Current context (BLUE or RED)
        # info['bean_states']: {(x,y): BeanState} for visualization
        # reward: Modified based on bean states
"""

import random
import numpy as np
from enum import Enum

# --- ENUMS FOR STATES ---
class Weather(Enum):
    BLUE = 0  # Safe and Stable
    RED = 1   # High Risk and Unpredictable
    GREY = 2  # Stagnant (NEW)


class BeanState(Enum):
    GREEN = 0 # Raw (0 points)
    RED = 1   # Ripe (+10 points)
    BROWN = 2 # Rotten (-10 points)

class ContextualVolatilityWrapper:
    """
    A wrapper around the JBW environment to implement Restless Dynamics
    and Contextual Volatility without altering the base JBW code.
    
    This wrapper layers contextual volatility on top of JBW's existing
    item system, tracking item states and modifying rewards dynamically
    based on collection timing and current weather context.
    """
    def __init__(self, jbw_env=None):
        self.env = jbw_env # The underlying JBW simulator/environment
        self.agent_pos = (0, 0)
        self._last_items = {}  # Track previous frame's items for collection detection
        
        # 1. Stochastic Weather Settings
        self.current_weather = Weather.BLUE
        
        # LOWERED: 0.005 means a 0.5% chance per tick. 
        # At 10 ticks/sec, weather will shift randomly every ~20 seconds on average.
        self.p_weather_switch = 0.005 
        
        # 2. Contextual Probabilities (The Rules)
        self.context_probs = {
            Weather.BLUE: {"P_ripen": 0.025,  "P_rot": 0.005},
            Weather.RED:  {"P_ripen": 0.025,  "P_rot": 0.050},
            # GREY CONTEXT: Zero growth, zero rotting. 
            Weather.GREY: {"P_ripen": 0.0,    "P_rot": 0.0} 
        }
        
        # 3. Bean Tracker
        # Maps grid coordinates (x, y) to their current BeanState
        # In a real infinite grid, you'd only track beans near the agent (Lazy Evaluation)
        self.active_beans = {} 


    def _inject_weather_into_obs(self, obs):
        """
        Appends the current weather context to the observation space so 
        the RL agent isn't 'blind' compared to a human player.
        """
        if obs is None:
            return None
            
        weather_val = self.current_weather.value  # 0 for BLUE, 1 for RED, 2 for GREY
        
        # If the environment uses dictionary observations (like your Mock Env)
        if isinstance(obs, dict):
            obs['weather_context'] = weather_val
            
        # If the environment uses flattened NumPy arrays (Standard RL)
        elif isinstance(obs, np.ndarray):
            obs = np.append(obs, weather_val)
            
        # If it's a standard Python list
        elif isinstance(obs, list):
            obs.append(weather_val)
            
        return obs

    def reset(self):
        """Resets the environment, weather, and beans."""
        if self.env:
            obs = self.env.reset()
            self.agent_pos = self._get_agent_position()
        else:
            obs = None
            self.agent_pos = (0, 0)
            
        self.current_weather = random.choice(list(Weather))
        self.active_beans.clear()
        
        self._discover_jbw_items()
        
        # INJECT WEATHER BEFORE RETURNING
        obs = self._inject_weather_into_obs(obs if obs is not None else {})
        return obs, self._get_context_info()


    def _discover_jbw_items(self):
        """Discovers items and maps them to our BeanState tracking with pre-ripening."""
        if not self.env: return
        try:
            if hasattr(self.env, 'get_items'):
                items = self.env.get_items()
                # BUG FIX: Handle the Pygame Mock dictionary format
                if isinstance(items, dict):
                    for pos in items.keys():
                        # 20% chance to spawn already ripe!
                        state = BeanState.RED if random.random() < 0.2 else BeanState.GREEN
                        self.active_beans[pos] = state
                # Handle the standard JBW C++ object format
                else:
                    for item in items:
                        if hasattr(item, 'x') and hasattr(item, 'y'):
                            pos = (item.x, item.y)
                            state = BeanState.RED if random.random() < 0.2 else BeanState.GREEN
                            self.active_beans[pos] = state
        except Exception as e:
            print(f"Note: Could not auto-discover items: {e}")

    def _get_agent_position(self):
        """Gets the agent's current position from JBW."""
        try:
            if hasattr(self.env, 'agent_pos'):
                pos = self.env.agent_pos
                return (pos[0], pos[1]) if hasattr(pos, '__getitem__') else pos
            elif hasattr(self.env, 'get_agent_pos'):
                return self.env.get_agent_pos()
            else:
                return (0, 0)
        except:
            return (0, 0)

    def step(self, action):
        # --- 1. STOCHASTIC WEATHER SWITCH ---
        if random.random() < self.p_weather_switch:
            self._switch_weather()

        # --- 2. RESTLESS RESOURCE LIFECYCLE ---
        self._update_beans_stochastically()

        # --- 3. EXECUTE AGENT ACTION ---
        if self.env:
            next_obs, reward, done, info = self.env.step(action)
        else:
            next_obs, reward, done, info = None, 0, False, {}

        # --- 4. APPLY REWARDS AND TRACK NEW BEANS ---
        if hasattr(self, '_last_items') and self.env and hasattr(self.env, 'get_items'):
            raw_items = self.env.get_items() if hasattr(self.env, 'get_items') else {}
            
            # BUG FIX: Extract just the coordinates to prevent Python reference tangles!
            if isinstance(raw_items, dict):
                current_keys = set(raw_items.keys())
            else:
                current_keys = set((i.x, i.y) for i in raw_items if hasattr(i, 'x'))

            if isinstance(self._last_items, dict):
                last_keys = set(self._last_items.keys())
            else:
                last_keys = set((i.x, i.y) for i in self._last_items if hasattr(i, 'x'))
            
            items_removed = last_keys - current_keys
            items_added = current_keys - last_keys
            
            # Apply state-based reward modifier for eaten beans
            for (x, y) in items_removed:
                modifier = self._get_item_reward_modifier(x, y)
                reward += modifier
                if (x, y) in self.active_beans:
                    del self.active_beans[(x, y)]
                    
            # Add newly spawned beans to the tracker so they become visible
            for (x, y) in items_added:
                # 20% chance to spawn already ripe so Grey weather isn't a dead-end
                if random.random() < 0.2:
                    self.active_beans[(x, y)] = BeanState.RED
                else:
                    self.active_beans[(x, y)] = BeanState.GREEN
                
            # Safely store just the keys for the next frame
            self._last_items = {k: True for k in current_keys}

        # Add our custom context info
        info.update(self._get_context_info())
        info['bean_states'] = dict(self.active_beans)
        info['weather'] = self.current_weather.name

        # INJECT WEATHER BEFORE RETURNING
        next_obs = self._inject_weather_into_obs(next_obs)

        return next_obs, reward, done, info

    def _switch_weather(self):
        """
        Flips the context using Weighted Probabilities to ensure
        Grey (Stagnant) appears much less frequently than Blue or Red.
        """
        if self.current_weather == Weather.BLUE:
            # If SAFE: 85% chance to become RISKY, only 15% chance to become STAGNANT
            next_states = [Weather.RED, Weather.GREY]
            weights = [0.85, 0.15] 
            
        elif self.current_weather == Weather.RED:
            # If RISKY: 85% chance to become SAFE, only 15% chance to become STAGNANT
            next_states = [Weather.BLUE, Weather.GREY]
            weights = [0.85, 0.15]
            
        else:
            # If STAGNANT (Grey): 50/50 chance to go back to a normal active state
            next_states = [Weather.BLUE, Weather.RED]
            weights = [0.50, 0.50]
            
        # random.choices returns a list, so we grab the first element [0]
        self.current_weather = random.choices(next_states, weights=weights, k=1)[0]

    def _update_beans_stochastically(self):
        """
        Iterates through all known beans and probabilistically advances 
        their lifecycle based on the CURRENT weather probabilities.
        """
        current_probs = self.context_probs[self.current_weather]
        
        # Create a list of keys to iterate over, since we are modifying the dictionary state
        for (x, y), state in list(self.active_beans.items()):
            
            # RAW (GREEN) -> RIPE (RED)
            if state == BeanState.GREEN:
                if random.random() < current_probs["P_ripen"]:
                    self.active_beans[(x, y)] = BeanState.RED
                    self._sync_jbw_item(x, y, BeanState.RED)
            
            # RIPE (RED) -> ROTTEN (BROWN)
            elif state == BeanState.RED:
                if random.random() < current_probs["P_rot"]:
                    self.active_beans[(x, y)] = BeanState.BROWN
                    self._sync_jbw_item(x, y, BeanState.BROWN)
                    
            # ROTTEN (BROWN) -> DISAPPEAR (Optional Cleanup)
            elif state == BeanState.BROWN:
                # Optional: Let rotten beans disappear after a while so the map doesn't clog
                if random.random() < 0.05: 
                    del self.active_beans[(x, y)]
                    self._remove_jbw_item(x, y)

    def _get_context_info(self):
        return {
            "weather": self.current_weather.name,
            "P_ripen": self.context_probs[self.current_weather]["P_ripen"],
            "P_rot": self.context_probs[self.current_weather]["P_rot"]
        }

    # --- JBW SYNC METHODS ---
    # Note: JBW's C++ simulator manages items through procedural generation and
    # doesn't expose direct item manipulation (add/remove at coordinates).
    # Instead, we track state in the wrapper and use it for reward modification
    # and observation enhancement. The visual representation is handled by the
    # game interface using color information from our state tracking.
    
    def _sync_jbw_item(self, x, y, new_state):
        """
        In our design, we DON'T modify JBW's items directly.
        Instead, we track state changes in self.active_beans.
        The state is used for:
        1. Reward modification (collect GREEN=0, RED=+10, BROWN=-10)
        2. Observation enhancement (pass state info to game interface)
        3. Visualization (color changes based on state)
        
        The JBW simulator still manages the base items; we layer behavior on top.
        """
        # State is already updated in self.active_beans by _update_beans_stochastically()
        # No need to modify JBW directly
        pass

    def _remove_jbw_item(self, x, y):
        """
        Cleanup: Tells the underlying simulator to delete the item 
        so we don't end up with invisible ghost beans.
        """
        if not self.env: return
        
        # Call the removal function if the underlying environment supports it
        if hasattr(self.env, 'remove_item'):
            self.env.remove_item(x, y)
    
    def _get_item_reward_modifier(self, x, y):
        """
        Query the state reward modifier for an item at position (x, y).
        """
        if (x, y) not in self.active_beans:
            return 0  # Unknown state, no modifier
        
        state = self.active_beans[(x, y)]
        
        # --- FIX THIS DICTIONARY ---
        modifiers = {
            BeanState.GREEN: 0,    # Change this from -1 to 0
            BeanState.RED: 10,     # Ripe gives +10
            BeanState.BROWN: -10   # Rotten gives -10
        }
        return modifiers.get(state, 0)
    
    def get_bean_state_for_position(self, x, y):
        """
        Query: What is the current state of the bean at (x, y)?
        Used by visualization/game interface to color items correctly.
        """
        return self.active_beans.get((x, y), BeanState.GREEN)
    
    def get_bean_color_for_state(self, state):
        """
        Returns RGB color tuple for visualization based on bean state.
        
        Returns:
            tuple: (R, G, B) normalized to 0-1
        """
        color_map = {
            BeanState.GREEN: (0.2, 0.8, 0.2),   # Green
            BeanState.RED: (1.0, 0.2, 0.2),     # Red
            BeanState.BROWN: (0.6, 0.4, 0.1)    # Brown
        }
        return color_map.get(state, (0.5, 0.5, 0.5))
    
    def on_bean_eaten(self, x, y):
        """Call this when the agent eats a bean to stop tracking it."""
        if (x, y) in self.active_beans:
            del self.active_beans[(x, y)]
    
    def get_all_beans(self):
        """
        Returns all tracked beans with their current states.
        
        Returns:
            dict: {(x, y): BeanState} mapping
        """
        return dict(self.active_beans)
    
    def get_context_string(self):
        """
        Returns human-readable context description.
        
        Returns:
            str: "SAFE & STABLE" or "RISKY & VOLATILE"
        """
        if self.current_weather == Weather.BLUE:
            return "SAFE & STABLE (Blue)"
        else:
            return "RISKY & VOLATILE (Red)"