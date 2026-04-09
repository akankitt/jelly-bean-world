import pygame
import random
import sys
import math
import csv
import os
import time
from datetime import datetime
from enum import Enum

# =====================================================================
# ENUMS & CONSTANTS
# =====================================================================
class Weather(Enum):
    BLUE = 0  # Safe and Stable
    RED = 1   # High Risk and Unpredictable
    GREY = 2  # Stagnant (Stasis)

class BeanState(Enum):
    GREEN = 0 # Raw (0 points)
    RED = 1   # Ripe (+10 points)
    BROWN = 2 # Rotten (-10 points)

TILE_SIZE = 22       
GRID_WIDTH = 33      
GRID_HEIGHT = 33     
FPS = 10  

COLOR_BG_BLUE = (200, 230, 255)  
COLOR_BG_RED = (255, 200, 200)   
COLOR_BG_GREY = (220, 220, 220)  
COLOR_GRID = (180, 180, 180)
COLOR_AGENT = (50, 50, 255)
COLOR_TEXT = (0, 0, 0)

# =====================================================================
# TELEMETRY HELPER
# =====================================================================
def get_nearest_bean_info(agent_pos, active_beans):
    """Calculates the Manhattan distance and state of the closest bean."""
    if not active_beans:
        return -1, "NONE"
    
    min_dist = float('inf')
    closest_state = "NONE"
    
    for (bx, by), state in active_beans.items():
        dist = abs(agent_pos[0] - bx) + abs(agent_pos[1] - by)
        if dist < min_dist:
            min_dist = dist
            closest_state = state.name # 'GREEN', 'RED', or 'BROWN'
            
    return min_dist, closest_state

# =====================================================================
# 1. DATA LOGGER
# =====================================================================
class DataLogger:
    """Logs high-fidelity behavioral telemetry to test psychological hypotheses."""
    def __init__(self, log_folder="logs"):
        self.log_folder = log_folder
        os.makedirs(self.log_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(self.log_folder, f"human_data_{timestamp}.csv")
        
        # Expanded Headers for Behavioral Analysis
        with open(self.filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Real_Timestamp", "Sim_Tick", 
                "Agent_X", "Agent_Y", "Action_Taken", 
                "Current_Weather", "Ticks_In_Weather", 
                "Nearest_Bean_Dist", "Nearest_Bean_State", 
                "Tick_Reward", "Total_Score"
            ])
        print(f"Logging behavioral telemetry to: {self.filepath}")

    def log_step(self, tick, agent_pos, action, weather, ticks_in_weather, 
                 nearest_dist, nearest_state, reward, total_score):
        with open(self.filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                round(time.time(), 3), tick, 
                agent_pos[0], agent_pos[1], action, 
                weather, ticks_in_weather, 
                nearest_dist, nearest_state, 
                round(reward, 1), round(total_score, 1)
            ])

# =====================================================================
# 2. MOCK JBW ENVIRONMENT (The Spawner & Physics)
# =====================================================================
class MockJBWEnv:
    def __init__(self):
        self.agent_pos = [GRID_WIDTH // 2, GRID_HEIGHT // 2]
        self.items = {} 
        self._initial_spawn()

    def _is_location_clear(self, cx, cy, min_distance):
        for (bx, by) in self.items.keys():
            dist = abs(cx - bx) + abs(cy - by)
            if dist < min_distance:
                return False
        return True

    def spawn_patch(self, center_x, center_y, num_beans, radius, state="GREEN"):
        spawned = 0
        attempts = 0
        while spawned < num_beans and attempts < 20:
            x = center_x + random.randint(-radius, radius)
            y = center_y + random.randint(-radius, radius)
            
            if (x, y) not in self.items and [x, y] != self.agent_pos:
                self.items[(x, y)] = state 
                spawned += 1
            attempts += 1

    def _initial_spawn(self):
        # 2AFC Initialization: Safe patch (5 beans), Rich patch (12 beans)
        self.spawn_patch(self.agent_pos[0] + 4, self.agent_pos[1], num_beans=5, radius=1, state="RED")
        self.spawn_patch(self.agent_pos[0] - 15, self.agent_pos[1], num_beans=12, radius=2, state="GREEN")

    def get_agent_pos(self):
        return self.agent_pos

    def get_items(self):
        return self.items

    def reset(self):
        self.agent_pos = [GRID_WIDTH // 2, GRID_HEIGHT // 2]
        self.items = {}
        self._initial_spawn()
        return {"obs": "dummy"}

    def step(self, action, weather="BLUE"):
        # --- 1. CULLING ---
        keys_to_delete = []
        for (bx, by) in self.items.keys():
            dist = abs(self.agent_pos[0] - bx) + abs(self.agent_pos[1] - by)
            if dist > 25:
                keys_to_delete.append((bx, by))
        for k in keys_to_delete:
            del self.items[k]

        # --- 2. EXPERIMENTAL SPAWNING ---
        if random.random() < 0.03 and len(self.items) < 30: 
            
            if weather == "GREY":
                # HYPOTHESIS 4: BOREDOM BAIT (Isolated Green Beans)
                cx = self.agent_pos[0] + random.randint(-15, 15)
                cy = self.agent_pos[1] + random.randint(-15, 15)
                if self._is_location_clear(cx, cy, min_distance=2):
                    self.spawn_patch(cx, cy, num_beans=2, radius=1, state="GREEN")

                # HYPOTHESIS 4: BOREDOM BAIT (Isolated Green Beans)
                cx = self.agent_pos[0] + random.randint(-15, 15)
                cy = self.agent_pos[1] + random.randint(-15, 15)
                if self._is_location_clear(cx, cy, min_distance=2):
                    self.spawn_patch(cx, cy, num_beans=3, radius=1, state="RED")
            
            else:
                # HYPOTHESES 2 & 3: 2AFC
                angle_safe = random.uniform(0, 2 * math.pi)
                dist_safe = random.randint(4, 7) 
                cx_safe = int(self.agent_pos[0] + dist_safe * math.cos(angle_safe))
                cy_safe = int(self.agent_pos[1] + dist_safe * math.sin(angle_safe))
                
                if self._is_location_clear(cx_safe, cy_safe, min_distance=6):
                    self.spawn_patch(cx_safe, cy_safe, num_beans=random.randint(5, 6), radius=1, state="RED")
                
                angle_rich = angle_safe + math.pi 
                dist_rich = random.randint(15, 20)
                cx_rich = int(self.agent_pos[0] + dist_rich * math.cos(angle_rich))
                cy_rich = int(self.agent_pos[1] + dist_rich * math.sin(angle_rich))
                
                if self._is_location_clear(cx_rich, cy_rich, min_distance=8):
                    self.spawn_patch(cx_rich, cy_rich, num_beans=random.randint(10, 12), radius=2, state="GREEN")

        # --- 3. MOVEMENT & REWARD ---
        if action == 0: self.agent_pos[1] -= 1
        elif action == 1: self.agent_pos[1] += 1
        elif action == 2: self.agent_pos[0] -= 1
        elif action == 3: self.agent_pos[0] += 1

        pos_tuple = (self.agent_pos[0], self.agent_pos[1])
        base_reward = 0 
        
        # Idle = 0 points. Movement = -0.1 points.
        if action in [0, 1, 2, 3]: 
            base_reward = -0.1 
        
        if pos_tuple in self.items:
            del self.items[pos_tuple]

        return {"obs": "dummy"}, base_reward, False, {}

# =====================================================================
# 3. EXPERIMENTAL WRAPPER (Contexts, Epochs, and States)
# =====================================================================
class ContextualVolatilityWrapper:
    def __init__(self, jbw_env):
        self.env = jbw_env 
        self._last_items = {}  
        
        # 1. Epoch-Based Weather Settings
        self.current_weather = Weather.BLUE
        self.ticks_in_current_weather = 0
        self.current_epoch_target = random.randint(300, 600) 
        
        # 2. Contextual Probabilities (Playable Variance)
        self.context_probs = {
            Weather.BLUE: {"P_ripen": 0.020,  "P_rot": 0.005},
            Weather.RED:  {"P_ripen": 0.040,  "P_rot": 0.040}, # 4% per tick rot
            Weather.GREY: {"P_ripen": 0.0,    "P_rot": 0.0} 
        }
        
        self.active_beans = {} 

    def reset(self):
        obs = self.env.reset()
        self.current_weather = Weather.BLUE
        self.ticks_in_current_weather = 0
        self.current_epoch_target = random.randint(450, 900)
        self.active_beans.clear()
        
        raw_items = self.env.get_items()
        for pos, state_str in raw_items.items():
            self.active_beans[pos] = BeanState.RED if state_str == "RED" else BeanState.GREEN
            self._last_items[pos] = True

        return obs, self._get_context_info()

    def step(self, action):
        # --- 1. EPOCH-BASED WEATHER SWITCH ---
        self.ticks_in_current_weather += 1
        if self.ticks_in_current_weather >= self.current_epoch_target:
            self._switch_weather()
            self.ticks_in_current_weather = 0
            self.current_epoch_target = random.randint(300, 600)

        # --- 2. RESTLESS RESOURCE LIFECYCLE ---
        self._update_beans_stochastically()

        # --- 3. EXECUTE AGENT ACTION ---
        next_obs, reward, done, info = self.env.step(action, weather=self.current_weather.name)

        # --- 4. APPLY REWARDS AND TRACK NEW BEANS ---
        raw_items = self.env.get_items() 
        current_keys = set(raw_items.keys())
        last_keys = set(self._last_items.keys())
        
        items_removed = last_keys - current_keys
        items_added = current_keys - last_keys
        
        current_agent_pos = tuple(self.env.get_agent_pos())
        
        for (x, y) in items_removed:
            if (x, y) == current_agent_pos:
                modifier = self._get_item_reward_modifier(x, y)
                reward += modifier
            if (x, y) in self.active_beans:
                del self.active_beans[(x, y)]
                
        for (x, y) in items_added:
            requested_state = raw_items.get((x, y), "GREEN")
            if requested_state == "RED":
                self.active_beans[(x, y)] = BeanState.RED
            else:
                self.active_beans[(x, y)] = BeanState.GREEN
            
        self._last_items = {k: True for k in current_keys}

        info.update(self._get_context_info())
        info['bean_states'] = dict(self.active_beans)
        info['weather'] = self.current_weather.name

        return next_obs, reward, done, info

    def _switch_weather(self):
        if self.current_weather == Weather.BLUE:
            next_states = [Weather.RED, Weather.GREY]
            weights = [0.8, 0.2] 
        elif self.current_weather == Weather.RED:
            next_states = [Weather.BLUE, Weather.GREY]
            weights = [0.8, 0.2]
        else:
            next_states = [Weather.BLUE, Weather.RED]
            weights = [0.50, 0.50]
        self.current_weather = random.choices(next_states, weights=weights, k=1)[0]

    def _update_beans_stochastically(self):
        current_probs = self.context_probs[self.current_weather]
        for (x, y), state in list(self.active_beans.items()):
            if state == BeanState.GREEN:
                if random.random() < current_probs["P_ripen"]:
                    self.active_beans[(x, y)] = BeanState.RED
            elif state == BeanState.RED:
                if random.random() < current_probs["P_rot"]:
                    self.active_beans[(x, y)] = BeanState.BROWN
            elif state == BeanState.BROWN:
                if random.random() < 0.05: 
                    del self.active_beans[(x, y)]
                    if (x,y) in self.env.items:
                        del self.env.items[(x,y)]

    def _get_item_reward_modifier(self, x, y):
        if (x, y) not in self.active_beans: return 0 
        state = self.active_beans[(x, y)]
        modifiers = {BeanState.GREEN: 0, BeanState.RED: 10, BeanState.BROWN: -10}
        return modifiers.get(state, 0)
    
    def get_bean_color_for_state(self, state):
        color_map = {
            BeanState.GREEN: (0.2, 0.8, 0.2),   
            BeanState.RED: (1.0, 0.2, 0.2),     
            BeanState.BROWN: (0.6, 0.4, 0.1)    
        }
        return color_map.get(state, (0.5, 0.5, 0.5))

    def _get_context_info(self):
        return {
            "weather": self.current_weather.name,
            "P_ripen": self.context_probs[self.current_weather]["P_ripen"],
            "P_rot": self.context_probs[self.current_weather]["P_rot"]
        }

# =====================================================================
# 4. GAME ENGINE & RENDERING
# =====================================================================
def draw_grid(screen, camera_x, camera_y):
    offset_x = -(camera_x * TILE_SIZE) % TILE_SIZE
    offset_y = -(camera_y * TILE_SIZE) % TILE_SIZE

    for x in range(offset_x, GRID_WIDTH * TILE_SIZE, TILE_SIZE):
        pygame.draw.line(screen, COLOR_GRID, (x, 0), (x, GRID_HEIGHT * TILE_SIZE))
    for y in range(offset_y, GRID_HEIGHT * TILE_SIZE, TILE_SIZE):
        pygame.draw.line(screen, COLOR_GRID, (0, y), (GRID_WIDTH * TILE_SIZE, y))

def main():
    pygame.init()
    screen = pygame.display.set_mode((GRID_WIDTH * TILE_SIZE, GRID_HEIGHT * TILE_SIZE + 50))
    pygame.display.set_caption("Contextual Volatility Foraging Task")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    mock_env = MockJBWEnv()
    wrapper = ContextualVolatilityWrapper(mock_env)
    logger = DataLogger()
    wrapper.reset()
    
    score = 0
    tick = 0
    running = True
    camera_x = 0
    camera_y = 0

    while running:
        action = None
        action_name = "STAY"
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action, action_name = 0, "UP"
                elif event.key == pygame.K_DOWN: action, action_name = 1, "DOWN"
                elif event.key == pygame.K_LEFT: action, action_name = 2, "LEFT"
                elif event.key == pygame.K_RIGHT: action, action_name = 3, "RIGHT"

        _, reward, _, info = wrapper.step(action)
        score = round(score + reward, 1)
        tick += 1
        agent_x, agent_y = mock_env.get_agent_pos()

        # --- BEHAVIORAL TELEMETRY ---
        nearest_dist, nearest_state = get_nearest_bean_info((agent_x, agent_y), info['bean_states'])
        ticks_in_weather = wrapper.ticks_in_current_weather

        logger.log_step(
            tick=tick, 
            agent_pos=(agent_x, agent_y), 
            action=action_name, 
            weather=info['weather'], 
            ticks_in_weather=ticks_in_weather,
            nearest_dist=nearest_dist,
            nearest_state=nearest_state,
            reward=reward,
            total_score=score
        )

        # --- CAMERA BOUNDS ---
        screen_x = agent_x - camera_x
        screen_y = agent_y - camera_y
        BUFFER = 5

        if screen_x <= BUFFER: camera_x = agent_x - (BUFFER + 2)
        elif screen_x >= GRID_WIDTH - BUFFER: camera_x = agent_x - (GRID_WIDTH - BUFFER - 2)
        if screen_y <= BUFFER: camera_y = agent_y - (BUFFER + 2)
        elif screen_y >= GRID_HEIGHT - BUFFER: camera_y = agent_y - (GRID_HEIGHT - BUFFER - 2)

        # --- RENDERING ---
        if info['weather'] == 'BLUE':
            screen.fill(COLOR_BG_BLUE)
            weather_text = "Context: SAFE (Blue) - Positive EV"
        elif info['weather'] == 'RED':
            screen.fill(COLOR_BG_RED)
            weather_text = "Context: RISKY (Red) - Brutal Rot Variance!"
        else: 
            screen.fill(COLOR_BG_GREY)
            weather_text = "Context: STASIS (Grey) - Nothing grows."

        draw_grid(screen, camera_x, camera_y)

        bean_states = info['bean_states']
        for (bx, by), state in bean_states.items():
            screen_bx = bx - camera_x
            screen_by = by - camera_y
            
            if 0 <= screen_bx < GRID_WIDTH and 0 <= screen_by < GRID_HEIGHT:
                color = wrapper.get_bean_color_for_state(state)
                pg_color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                center_x = screen_bx * TILE_SIZE + (TILE_SIZE // 2)
                center_y = screen_by * TILE_SIZE + (TILE_SIZE // 2)
                pygame.draw.circle(screen, pg_color, (center_x, center_y), TILE_SIZE // 3)

        screen_agent_x = agent_x - camera_x
        screen_agent_y = agent_y - camera_y
        rect = pygame.Rect(screen_agent_x * TILE_SIZE + 5, screen_agent_y * TILE_SIZE + 5, TILE_SIZE - 10, TILE_SIZE - 10)
        pygame.draw.rect(screen, COLOR_AGENT, rect)

        pygame.draw.rect(screen, (255, 255, 255), (0, GRID_HEIGHT * TILE_SIZE, GRID_WIDTH * TILE_SIZE, 50))
        score_surface = font.render(f"Score: {score}", True, COLOR_TEXT)
        weather_surface = font.render(weather_text, True, COLOR_TEXT)
        screen.blit(score_surface, (10, GRID_HEIGHT * TILE_SIZE + 15))
        screen.blit(weather_surface, (150, GRID_HEIGHT * TILE_SIZE + 15))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()