import pandas as pd
import pygame
import random
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

class SimConfig:
    grid_size: int = 50
    cell_size: int = 30
    cell_draw_size: int = 12
    
    base_spread: float = 0.18
    heat_transfer_coef: float = 0.15
    ember_prob: float = 0.05
    ember_max_dist: int = 3
    
    wind_max_boost: float = 3.5
    wind_max_reduce: float = 0.15
    wind_gust_variation: float = 0.2
    
    water_drop_rad: int = 2
    water_eff: float = 0.85
    retardant_dur: int = 15
    firebreak_width: int = 1
    
    firef_speed: float = 1.5
    firef_cap: int = 5
    firef_range: int = 2
    #set parametri
config = SimConfig()

class CellState(Enum):
    UNBURNED = 0
    BURNING = 1
    BURNED = 2
    WET = 3
    FIREBREAK = 4

class Cell:
    state: CellState = CellState.UNBURNED
    fuel_load: float = 1.0       
    moisture: float = 0.4        
    elevation: float = 0.0         
    temperature: float = 20.0           
    fire_intensity: float = 0.0        
    burn_time_remaining: int = 0
    water_coverage: float = 0.0       
    retardant_age: int = 0
    ash_age: int = 0
    
    def update_temperature(self, neighbor_heat: float, ambient_temp: float = 20.0):
        #transfer termic
        if self.state == CellState.BURNING:
            self.temperature = 800 + random.uniform(-100, 100)
        elif self.state == CellState.BURNED:                                
            self.temperature = ambient_temp + (self.temperature - ambient_temp) * 0.85
        else:                                    
            heat_gain = neighbor_heat * config.heat_transfer_coef * (1 - self.moisture)
            self.temperature = min(self.temperature + heat_gain, 800)

@dataclass
class FirefightingUnit:
    x: float
    y: float
    water_remaining: int = 5
    #cantitatea de apă per pompier
    target: Optional[Tuple[int, int]] = None
    deployed: bool = False
    
    def move_towards(self, target_x: int, target_y: int):
        #mutam unitatea un pas spre o tinta
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.hypot(dx, dy)
        # hypot(dx, dy) → √(dx² + dy²)
        if dist > 0:
            self.x += (dx / dist) * config.firef_speed
            self.y += (dy / dist) * config.firef_speed
            #dx dy vect direcție, dist - distanța reală până la țintă
            
    def can_suppress(self, x: int, y: int) -> bool:
        dist = math.hypot(self.x - x, self.y - y)
        return dist <= config.firef_range and self.water_remaining > 0

class WildfireSimulation:
    def __init__(self, tree_data_path: str, mtbs_data_path: str):               
        self.tree_df = pd.read_csv(tree_data_path)
        self.mtbs_df = pd.read_csv(mtbs_data_path)
        self.mtbs_df = self.mtbs_df[self.mtbs_df["BurnBndAc"] > 0]
        #datele sunt preluate din mtbs, și tipurile de vegetație
                          
        self.grid_size = config.grid_size
        self.cells = [[Cell() for _ in range(self.grid_size)] 
                      for _ in range(self.grid_size)]
        
                        
        self.tree_data = {row['specie']: {
            'veg_factor': row['veg_factor'], 
            'burn_time': row['burn_time']
        } for _, row in self.tree_df.iterrows()}
        #veg factor -> cat de inflamabila e, burn_time -> cat va arde
        self.veg_types = {i: specie for i, specie in enumerate(self.tree_df['specie'])}
        self.current_veg = 0
        #permite sch tipului de vegetatie
                                
        self.wind_x = 0.0
        self.wind_y = 0.0
        self.ambient_moisture = 0.4
        self.ambient_temp = 25.0
        self.wind_gust_timer = 0
                  
        self.firefighters: List[FirefightingUnit] = []
                    
        self.burned_area_history = []
        self.fire_intensity_history = []
        self.time_step = 0
                             
        self._generate_topography()
        self.reset()
    
    def _generate_topography(self):
        #creaza panta globala cu zgomot aleator (pseudo Perlin)
        #focul urca mai repede la deal
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                                             
                base_elev = (x + y) / self.grid_size * 50
                variation = random.uniform(-15, 15)
                self.cells[x][y].elevation = max(0, base_elev + variation)
    
    def reset(self):
        #sim reset
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell = self.cells[x][y]
                cell.state = CellState.UNBURNED
                cell.fuel_load = random.uniform(0.7, 1.0)
                cell.moisture = self.ambient_moisture + random.uniform(-0.1, 0.1)
                cell.temperature = self.ambient_temp
                cell.fire_intensity = 0.0
                cell.burn_time_remaining = 0
                cell.water_coverage = 0.0
                cell.retardant_age = 0
                cell.ash_age = 0
                         
        cx = cy = self.grid_size // 2
        self.ignite_cell(cx, cy)
        
        self.firefighters = []
        self.burned_area_history = []
        self.fire_intensity_history = []
        self.time_step = 0
    
    def ignite_cell(self, x: int, y: int):
        #aprinde o celula daca nu ii arsa si daca nu ii prea ud
        cell = self.cells[x][y]
        if cell.state == CellState.UNBURNED and cell.water_coverage < 0.5:
            cell.state = CellState.BURNING
            cell.burn_time_remaining = self.tree_data[self.veg_types[self.current_veg]]['burn_time']
            cell.fire_intensity = 1000 * cell.fuel_load * (1 - cell.moisture)
            cell.temperature = 800
    
    def apply_water(self, x: int, y: int, radius: int = 2):
        #aplica apa intr-o zona, creste water_coverge, moisture, reduce temperatura si fire intensity. poate stinge focul (se raspandeste si in zonele apropiate)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    dist = math.hypot(dx, dy)
                    if dist <= radius:
                        cell = self.cells[nx][ny]
                        effectiveness = config.water_eff * (1 - dist / radius)
                        cell.water_coverage = min(1.0, cell.water_coverage + effectiveness)
                        cell.moisture = min(1.0, cell.moisture + effectiveness * 0.3)
                        cell.temperature *= (1 - effectiveness)
                        
                        if cell.state == CellState.BURNING:
                            cell.fire_intensity *= (1 - effectiveness)
                            if cell.fire_intensity < 100:
                                cell.state = CellState.BURNED
    
    def create_firebreak(self, x: int, y: int):
        #creaza zone moarte care nu pot fi trecute de foc
        for dx in range(-config.firebreak_width, config.firebreak_width + 1):
            for dy in range(-config.firebreak_width, config.firebreak_width + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    cell = self.cells[nx][ny]
                    if cell.state == CellState.UNBURNED:
                        cell.state = CellState.FIREBREAK
                        cell.fuel_load = 0.0
    
    def spawn_firefighter(self, x: int, y: int):
        self.firefighters.append(FirefightingUnit(x, y))
    
    def update_wind(self, vx: float, vy: float):
        #introduce haos controlat, rafala aleatorie
        self.wind_gust_timer += 1
        base_x, base_y = vx, vy
        
        if self.wind_gust_timer % 5 == 0:
            gust_x = random.uniform(-config.wind_gust_variation, config.wind_gust_variation)
            gust_y = random.uniform(-config.wind_gust_variation, config.wind_gust_variation)
            self.wind_x = base_x + gust_x
            self.wind_y = base_y + gust_y
        else:
            self.wind_x = base_x
            self.wind_y = base_y
    
    def calculate_spread_probability(self, from_cell: Cell, to_cell: Cell, dx: int, dy: int) -> float:
        #Rothermel model  
        if to_cell.state != CellState.UNBURNED:
            return 0.0
        if to_cell.state == CellState.FIREBREAK:
            return 0.0                 
        if to_cell.water_coverage > 0.5:
            return 0.0
        #factorii de propagare: tipul vegetatie, penalizare exponentiala, combustibil, pre-incalzire, directie+intesitate, focul urca, apa, si diagonal < ortogonal                 
        veg_factor = self.tree_data[self.veg_types[self.current_veg]]['veg_factor']
                                        
        moisture_factor = math.exp(-3 * to_cell.moisture)
                            
        fuel_factor = to_cell.fuel_load
                            
        temp_factor = 1.0
        if to_cell.temperature > 200:
            temp_factor = 1.0 + (to_cell.temperature - 200) / 600
        
                                 
        wind_factor = 1.0
        if abs(self.wind_x) > 0.01 or abs(self.wind_y) > 0.01:
            dir_x = dx / max(abs(dx), abs(dy), 1)
            dir_y = dy / max(abs(dx), abs(dy), 1)
            
            wind_mag = math.hypot(self.wind_x, self.wind_y)
            if wind_mag > 0:
                wind_norm_x = self.wind_x / wind_mag
                wind_norm_y = self.wind_y / wind_mag
                dot = dir_x * wind_norm_x + dir_y * wind_norm_y
                
                if dot > 0:
                    wind_factor = 1 + dot * (config.wind_max_boost - 1) * wind_mag
                else:
                    wind_factor = 1 + dot * (1 - config.wind_max_reduce)
        
                                                      
        elev_diff = to_cell.elevation - from_cell.elevation
        slope_factor = 1.0 + (elev_diff / 10) * 0.3
        slope_factor = max(0.5, min(slope_factor, 2.0))
                                                 
        distance_factor = 1.0 if (dx == 0 or dy == 0) else 0.7
                                      
        water_reduction = 1.0 - (to_cell.water_coverage * 0.8)
                        
        probability = (config.base_spread * 
                      veg_factor * 
                      moisture_factor * 
                      fuel_factor * 
                      temp_factor * 
                      wind_factor * 
                      slope_factor * 
                      distance_factor * 
                      water_reduction)
        #P = BASE * toti_factorii
        return min(probability, 0.95)
    
    def simulate_embers(self):
        #genereaza scantei, sunt influentate de vant, si pot aprinde focul la distanta
        burning_cells = [(x, y) for x in range(self.grid_size) 
                        for y in range(self.grid_size) 
                        if self.cells[x][y].state == CellState.BURNING]
        
        for x, y in burning_cells:
            if random.random() < config.ember_prob:
                dist = random.randint(1, config.ember_max_dist)
                angle = random.uniform(0, 2 * math.pi)
                
                                               
                wind_influence = 0.7
                if abs(self.wind_x) > 0.01 or abs(self.wind_y) > 0.01:
                    wind_angle = math.atan2(self.wind_y, self.wind_x)
                    angle = wind_angle + random.uniform(-0.5, 0.5)
                
                ex = int(x + dist * math.cos(angle))
                ey = int(y + dist * math.sin(angle))
                
                if 0 <= ex < self.grid_size and 0 <= ey < self.grid_size:
                    target = self.cells[ex][ey]
                    if target.state == CellState.UNBURNED and target.water_coverage < 0.3:
                        if random.random() < 0.3 * (1 - target.moisture):
                            self.ignite_cell(ex, ey)
    
    def update_firefighters(self):
        # dacă are țintă -> se deplasează
        # dacă ajunge -> aplică apă
        # dacă nu are țintă -> caută cel mai apropiat foc
        # dacă nu mai are apă -> dispare
        for ff in self.firefighters[:]:
            if not ff.deployed:
                continue
            
            if ff.target:
                tx, ty = ff.target
                ff.move_towards(tx, ty)
                
                                    
                if math.hypot(ff.x - tx, ff.y - ty) < 1.5:
                    if ff.water_remaining > 0:
                        self.apply_water(tx, ty, config.water_drop_rad)
                        ff.water_remaining -= 1
                    ff.target = None
            else:
                                            
                min_dist = float('inf')
                closest = None
                
                for x in range(self.grid_size):
                    for y in range(self.grid_size):
                        if self.cells[x][y].state == CellState.BURNING:
                            dist = math.hypot(ff.x - x, ff.y - y)
                            if dist < min_dist:
                                min_dist = dist
                                closest = (x, y)
                
                if closest and min_dist < 15:
                    ff.target = closest
            
                                           
            if ff.water_remaining == 0:
                self.firefighters.remove(ff)
    
    def step(self):
        #creste timpul
        #actualizeaza pompierii
        #calc temperatura
        #evaporeaza apa
        #pt foc -> arde, scade burn time, aprinde vecini
        #finalizeaza celulele arse
        #scantei decl la fiecare 3 pasi
        self.time_step += 1
        
                         
        self.update_firefighters()
        
                       
        new_ignitions = []
        total_intensity = 0
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell = self.cells[x][y]
                
                                               
                neighbor_heat = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            n_cell = self.cells[nx][ny]
                            if n_cell.state == CellState.BURNING:
                                neighbor_heat += n_cell.temperature / 8
                
                cell.update_temperature(neighbor_heat, self.ambient_temp)
                
                                         
                if cell.water_coverage > 0:
                    cell.water_coverage *= 0.92
                
                             
                if cell.state == CellState.BURNING:
                    total_intensity += cell.fire_intensity
                    cell.burn_time_remaining -= 1
                    
                    if cell.burn_time_remaining <= 0:
                        cell.state = CellState.BURNED
                        cell.fire_intensity = 0
                        cell.ash_age = 0
                        cell.fuel_load = 0
                    else:
                                   
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                                    target = self.cells[nx][ny]
                                    prob = self.calculate_spread_probability(cell, target, dx, dy)
                                    
                                    if random.random() < prob:
                                        new_ignitions.append((nx, ny))
                
                elif cell.state == CellState.BURNED:
                    cell.ash_age += 1
        
                            
        for x, y in new_ignitions:
            self.ignite_cell(x, y)
        
                           
        if self.time_step % 3 == 0:
            self.simulate_embers()
        
                    
        burned_count = sum(1 for x in range(self.grid_size) 
                          for y in range(self.grid_size) 
                          if self.cells[x][y].state == CellState.BURNED)
        
        cell_area_ac = (config.cell_size ** 2) / 4046.86
        burned_area = burned_count * cell_area_ac
        
        self.burned_area_history.append(burned_area)
        self.fire_intensity_history.append(total_intensity)
        
        return burned_area, total_intensity
    
    def is_active(self) -> bool:
        #verifica foc activ
        return any(self.cells[x][y].state == CellState.BURNING 
                  for x in range(self.grid_size) 
                  for y in range(self.grid_size))

                                                              
                  
                                                              

def draw_simulation(screen, sim: WildfireSimulation, show_intensity: bool = False):
    for x in range(sim.grid_size):
        for y in range(sim.grid_size):
            cell = sim.cells[x][y]
            
            if show_intensity and cell.state == CellState.BURNING:
                                     
                intensity_norm = min(cell.fire_intensity / 2000, 1.0)
                r = int(255 * intensity_norm)
                g = int(50 * (1 - intensity_norm))
                b = 0
                color = (r, g, b)
            else:
                                
                if cell.state == CellState.UNBURNED:
                    green_val = int(100 + 100 * cell.fuel_load)
                    color = (34, green_val, 34)
                    if cell.water_coverage > 0.1:
                        blue_mix = int(cell.water_coverage * 150)
                        color = (34, green_val, 34 + blue_mix)
                elif cell.state == CellState.BURNING:
                    color = (255, int(100 + 100 * random.random()), 0)
                elif cell.state == CellState.BURNED:
                    age_factor = min(cell.ash_age / 30, 1.0)
                    gray = int(60 * (1 - age_factor) + 30 * age_factor)
                    color = (gray, gray, gray)
                elif cell.state == CellState.WET:
                    color = (50, 100, 200)
                elif cell.state == CellState.FIREBREAK:
                    color = (139, 69, 19)
            
            pygame.draw.rect(screen, color, 
                           (x * config.cell_draw_size, 
                            y * config.cell_draw_size,
                            config.cell_draw_size, 
                            config.cell_draw_size))
                        
    for ff in sim.firefighters:
        if ff.deployed:
            px = int(ff.x * config.cell_draw_size + config.cell_draw_size // 2)
            py = int(ff.y * config.cell_draw_size + config.cell_draw_size // 2)
            
                                
            pygame.draw.circle(screen, (0, 100, 255), (px, py), 4)
            pygame.draw.circle(screen, (255, 255, 0), (px, py), 2)
            
                               
            if ff.target:
                tx = ff.target[0] * config.cell_draw_size + config.cell_draw_size // 2
                ty = ff.target[1] * config.cell_draw_size + config.cell_draw_size // 2
                pygame.draw.line(screen, (0, 200, 255), (px, py), (tx, ty), 1)

                                                              
def main():
                       
    WIDTH = HEIGHT = config.grid_size * config.cell_draw_size
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("wildfire sim")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 14)
                        
    sim = WildfireSimulation("tree_data1.csv", "mtbs_stats.csv")
              
    paused = False
    show_intensity = False
    mouse_mode = "none"                                       
    sim_speed = 10

    running = True
    while running:
        clock.tick(sim_speed)
                  
        keys = pygame.key.get_pressed()
        vx = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
        vy = keys[pygame.K_DOWN] - keys[pygame.K_UP]
        sim.update_wind(vx, vy)
             
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    sim.reset()                
                elif event.key == pygame.K_1:
                    sim.current_veg = 0
                    sim.reset()
                elif event.key == pygame.K_2:
                    sim.current_veg = 1
                    sim.reset()
                elif event.key == pygame.K_3:
                    sim.current_veg = 2
                    sim.reset()
                elif event.key == pygame.K_4:
                    sim.current_veg = 3
                    sim.reset()
                elif event.key == pygame.K_5:
                    sim.current_veg = 4
                    sim.reset()    
                elif event.key == pygame.K_e:
                    sim.ambient_moisture = min(1.0, sim.ambient_moisture + 0.05)
                elif event.key == pygame.K_d:
                    sim.ambient_moisture = max(0.0, sim.ambient_moisture - 0.05)           
                elif event.key == pygame.K_w:
                    mouse_mode = "water"
                elif event.key == pygame.K_f:
                    mouse_mode = "firebreak"
                elif event.key == pygame.K_p:
                    mouse_mode = "firefighter"
                elif event.key == pygame.K_n:
                    mouse_mode = "none"                          
                elif event.key == pygame.K_i:
                    show_intensity = not show_intensity
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    sim_speed = min(60, sim_speed + 5)
                elif event.key == pygame.K_MINUS:
                    sim_speed = max(1, sim_speed - 5)

                    
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                gx = mx // config.cell_draw_size
                gy = my // config.cell_draw_size
                
                if 0 <= gx < sim.grid_size and 0 <= gy < sim.grid_size:
                    if mouse_mode == "water":
                        sim.apply_water(gx, gy, config.water_drop_rad)
                    elif mouse_mode == "firebreak":
                        sim.create_firebreak(gx, gy)
                    elif mouse_mode == "firefighter":
                        sim.spawn_firefighter(gx, gy)
                        sim.firefighters[-1].deployed = True
                      
        if not paused and sim.is_active():
            burned_area, intensity = sim.step()
         
        screen.fill((0, 0, 0))
        draw_simulation(screen, sim, show_intensity)
                
        y_offset = 5
        texts = [
            f"Step: {sim.time_step} | Mode: {mouse_mode.upper()}",
            f"Species: {sim.veg_types[sim.current_veg]} (1-5)",
            f"Moisture: {sim.ambient_moisture:.2f} (E/D)",
            f"Wind: ({sim.wind_x:.1f}, {sim.wind_y:.1f}) (Arrows)",
            f"Firefighters: {len(sim.firefighters)} (P to place)",
            f"Burned: {sim.burned_area_history[-1] if sim.burned_area_history else 0:.1f} ac",
            f"Speed: {sim_speed} FPS (+/- to adjust)",
            "",
            "W=Water | F=Firebreak | N=None | I=Intensity",
            "SPACE=Pause | R=Reset | ESC=Quit"
        ]
        
        for text in texts:
            surface = font.render(text, True, (255, 255, 255))
            screen.blit(surface, (5, y_offset))
            y_offset += 18
        
        pygame.display.flip()
    
    pygame.quit()
    
    if sim.burned_area_history:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        ax1.plot(sim.burned_area_history, linewidth=2, color='orange')
        mean_area = sim.mtbs_df["BurnBndAc"].mean()
        ax1.axhline(mean_area, linestyle="--", color='red', label='Mean historical')
        ax1.set_xlabel("Time step")
        ax1.set_ylabel("Burned Area (acres)")
        ax1.set_title(f"Wildfire Progression - {sim.veg_types[sim.current_veg]}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(sim.fire_intensity_history, linewidth=2, color='red')
        ax2.set_xlabel("Time step")
        ax2.set_ylabel("Total Fire Intensity (kW/m)")
        ax2.set_title("Fire Intensity Over Time")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()