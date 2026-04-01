import pygame
import random
import sys
import math
import csv
import os
import time
from datetime import datetime
from wrapper import ContextualVolatilityWrapper, BeanState, Weather

# --- CONSTANTS & CONFIG ---
# Increased Grid Size for better patch-foraging travel distance!
TILE_SIZE = 22       # Shrunk from 35 so the map fits on your screen
GRID_WIDTH = 33      # Increased from 20
GRID_HEIGHT = 33     # Increased from 20
FPS = 10  

# Colors
COLOR_BG_BLUE = (200, 230, 255)  
COLOR_BG_RED = (255, 200, 200)   
COLOR_GRID = (180, 180, 180)
COLOR_AGENT = (50, 50, 255)
COLOR_TEXT = (0, 0, 0)
 
COLOR_BG_GREY = (220, 220, 220)  # Add this line

class DataLogger:
    """Handles logging all human interactions to a CSV file in a logs directory."""
    def __init__(self, log_folder="logs"):
        # 1. Define and create the directory if it doesn't exist
        self.log_folder = log_folder
        os.makedirs(self.log_folder, exist_ok=True)
        
        # 2. Create the timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"human_data_{timestamp}.csv"
        
        # 3. Join the folder and filename together for the final path
        self.filepath = os.path.join(self.log_folder, filename)
        
        # Write the CSV headers
        with open(self.filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Timestamp", "Tick", "Agent_X", "Agent_Y", 
                "Action_Taken", "Current_Weather", "Reward_Received"
            ])
        print(f"Logging data to: {self.filepath}")

    def log_step(self, tick, agent_pos, action, weather, reward):
        """Appends a single step of data to the CSV."""
        # Use the stored filepath here as well
        with open(self.filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                time.time(), tick, agent_pos[0], agent_pos[1], 
                action, weather, reward
            ])

class MockJBWEnv:
    def __init__(self):
        self.agent_pos = [GRID_WIDTH // 2, GRID_HEIGHT // 2]
        self.items = {}
        for _ in range(4):
            self.spawn_patch(
                center_x=self.agent_pos[0] + random.randint(-12, 12),
                center_y=self.agent_pos[1] + random.randint(-12, 12)
            )

    def spawn_patch(self, center_x, center_y, num_beans=None, radius=2):
        if num_beans is None:
            num_beans = random.randint(5, 9) 
            
        spawned = 0
        attempts = 0
        while spawned < num_beans and attempts < 20:
            x = center_x + random.randint(-radius, radius)
            y = center_y + random.randint(-radius, radius)
            
            if (x, y) not in self.items and [x, y] != self.agent_pos:
                self.items[(x, y)] = 0 
                spawned += 1
            attempts += 1

    def get_agent_pos(self):
        return self.agent_pos

    def get_items(self):
        return self.items
    
    # --- ADD THIS MISSING METHOD ---
    def _is_location_clear(self, cx, cy, min_distance=7):
        """
        Scans the proposed spawn location. 
        Returns False if there are already beans too close.
        """
        for (bx, by) in self.items.keys():
            # Calculate Manhattan distance to existing beans
            dist = abs(cx - bx) + abs(cy - by)
            if dist < min_distance:
                return False
        return True
    # -------------------------------

    # --- ADD THIS NEW METHOD ---
    def remove_item(self, x, y):
        """Allows the wrapper to successfully garbage collect rotten beans."""
        if (x, y) in self.items:
            del self.items[(x, y)]
    # ---------------------------

    def reset(self):
        self.agent_pos = [GRID_WIDTH // 2, GRID_HEIGHT // 2]
        self.items = {}
        for _ in range(4):
            self.spawn_patch(
                center_x=self.agent_pos[0] + random.randint(-12, 12),
                center_y=self.agent_pos[1] + random.randint(-12, 12)
            )
        return {"obs": "dummy"}

    def step(self, action):
        # --- 1. SPATIAL CULLING (Garbage Collection) ---
        # Delete items that are more than 25 tiles away from the agent.
        # This prevents the map from filling up with "forgotten" patches behind the player.
        keys_to_delete = []
        for (bx, by) in self.items.keys():
            # Manhattan distance calculation
            dist = abs(self.agent_pos[0] - bx) + abs(self.agent_pos[1] - by)
            if dist > 25:
                keys_to_delete.append((bx, by))
                
        for k in keys_to_delete:
            del self.items[k]


        # --- 2. DIRECTIONAL "DONUT" SPAWNING ---
        # Kept spawn rate at 3% (0.03). Increased cap to 200 for the larger world.
        if random.random() < 0.03 and len(self.items) < 200:
            
            # Pushed the spawn distance out to the edge of the new 33x33 screen
            distance = random.randint(14, 17) 
            
            if action == 0:   # UP
                cx = self.agent_pos[0] + random.randint(-5, 5)
                cy = self.agent_pos[1] - distance
            elif action == 1: # DOWN
                cx = self.agent_pos[0] + random.randint(-5, 5)
                cy = self.agent_pos[1] + distance
            elif action == 2: # LEFT
                cx = self.agent_pos[0] - distance
                cy = self.agent_pos[1] + random.randint(-5, 5)
            elif action == 3: # RIGHT
                cx = self.agent_pos[0] + distance
                cy = self.agent_pos[1] + random.randint(-5, 5)
            else:             # STAY 
                angle = random.uniform(0, 2 * math.pi)
                cx = int(self.agent_pos[0] + distance * math.cos(angle))
                cy = int(self.agent_pos[1] + distance * math.sin(angle))
                
            # Smart Fix: Ensure patches don't spawn on top of each other
            if self._is_location_clear(cx, cy, min_distance=7):
                self.spawn_patch(cx, cy)


        # --- 3. MOVEMENT & REWARD LOGIC ---
        if action == 0: self.agent_pos[1] -= 1
        elif action == 1: self.agent_pos[1] += 1
        elif action == 2: self.agent_pos[0] -= 1
        elif action == 3: self.agent_pos[0] += 1

        pos_tuple = (self.agent_pos[0], self.agent_pos[1])
        base_reward = 0 
        
        if action is not None:
            base_reward = -0.1 
        
        if pos_tuple in self.items:
            del self.items[pos_tuple]
            base_reward += 0 

        return {"obs": "dummy"}, base_reward, False, {}
    

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
    pygame.display.set_caption("Contextual Volatility Foraging Task - INFINITE")
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

        logger.log_step(tick, mock_env.get_agent_pos(), action_name, info['weather'], reward)

        # --- UPDATED CAMERA MATH ---
        screen_x = agent_x - camera_x
        screen_y = agent_y - camera_y

        # Give the agent a comfortable "buffer" of 5 tiles from the edge before the camera moves
        BUFFER = 5

        # X-Axis Push
        if screen_x <= BUFFER:
            camera_x = agent_x - (BUFFER + 2)
        elif screen_x >= GRID_WIDTH - BUFFER:
            camera_x = agent_x - (GRID_WIDTH - BUFFER - 2)

        # Y-Axis Push
        if screen_y <= BUFFER:
            camera_y = agent_y - (BUFFER + 2)
        elif screen_y >= GRID_HEIGHT - BUFFER:
            camera_y = agent_y - (GRID_HEIGHT - BUFFER - 2)


        # --- RENDERING ---
        if info['weather'] == 'BLUE':
            screen.fill(COLOR_BG_BLUE)
            weather_text = "Context: SAFE (Blue) - Patience Rewarded"
        elif info['weather'] == 'RED':
            screen.fill(COLOR_BG_RED)
            weather_text = "Context: RISKY (Red) - High Rot Chance!"
        else: # GREY
            screen.fill(COLOR_BG_GREY)
            weather_text = "Context: STAGNANT (Grey) - Nothing grows. Explore!"

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