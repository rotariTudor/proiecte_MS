import matplotlib.pyplot as plt
import pygame
import random

pygame.init()

WIDTH, HEIGHT = 1200, 800
BACKGROUND_COLOR = (30, 30, 30)
PREY_COLOR = (255, 165, 0)
PREDATOR_COLOR = (255, 0, 0)
FOOD_COLOR = (0, 255, 0)
TEXT_COLOR = (200, 200, 200)
FPS = 60

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Predator-Prey Simulation")
clock = pygame.time.Clock()
FONT = pygame.font.SysFont(None, 24)


class Obstacle:
    def __init__(self, x, y, radius=20):
        self.position = pygame.math.Vector2(x, y)
        self.radius = radius

    def draw(self):
        pygame.draw.circle(screen, (100, 100, 100), (int(self.position.x), int(self.position.y)), self.radius)



class Food:
    def __init__(self, obstacles):
        while True:

            aux_pos = pygame.math.Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
            if all(aux_pos.distance_to(obs.position) > obs.radius + 5 for obs in obstacles):
                self.position = aux_pos
                break
        self.color = FOOD_COLOR

    def draw(self):
        pygame.draw.circle(screen, self.color, (int(self.position.x), int(self.position.y)), 3)


class Agent:
    def __init__(self, position=None, velocity=None, speed=2, color=PREY_COLOR):
        self.position = position or pygame.math.Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
        self.velocity = velocity or pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize()
        self.speed = speed
        self.color = color
        self.trail = []
        self.max_trail_length = 10
        self.energy = 50
        self.mating = False
        self.mating_timer = 0
        self.mating_duration = 30
        self.repro_threshold = 90
        self.partner = None

    def avoid_obstacles(self, obstacles):
        for obs in obstacles:
            dist = self.position.distance_to(obs.position)
            if dist < obs.radius + 40:
                away = (self.position - obs.position).normalize()
                self.velocity = (self.velocity + away * 1.5).normalize()

    def update_position(self):
        self.energy -= 0.1
        self.position += self.velocity * self.speed
        if self.position.x < 0 or self.position.x > WIDTH:
            self.velocity.x *= -1
        if self.position.y < 0 or self.position.y > HEIGHT:
            self.velocity.y *= -1
        self.position.x = max(0, min(self.position.x, WIDTH))
        self.position.y = max(0, min(self.position.y, HEIGHT))
        self.trail.append(self.position.copy())
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)

    def draw_trail(self):
        if len(self.trail) > 1:
            pygame.draw.lines(screen, self.color, False, [(int(p.x), int(p.y)) for p in self.trail], 1)

    def draw(self):
        raise NotImplementedError

    def ensure_nonzero_velocity(self):
        if self.velocity.length_squared() < 0.01:
            self.velocity = pygame.math.Vector2(random.uniform(-1,1), random.uniform(-1,1)).normalize()



class Prey(Agent):
    def __init__(self):
        super().__init__(speed=2, color=PREY_COLOR)
        self.vision_radius = 50

    def update(self, predators, prey_list, sim):
        self.ensure_nonzero_velocity()
        self.energy += 0.5
        if self.mating:
            self.mating_timer -= 1
            if self.mating_timer <= 0:
                self.mating = False
                if self.partner:
                    self.partner.mating = False
                    self.partner.partner = None
                    self.partner = None
            return


        if self.energy >= sim.prey_repro_treshold and len(prey_list) < 100:
            partner = None
            min_dist = 20
            for other in prey_list:
                if other is self or other.mating:
                    continue
                if other.energy < other.repro_threshold:
                    continue
                d = self.position.distance_to(other.position)
                if d < min_dist:
                    partner = other
                    min_dist = d
            if partner:
                self.mating = True
                partner.mating = True
                self.mating_timer = self.mating_duration
                partner.mating_timer = partner.mating_duration
                self.partner = partner
                partner.partner = self
                baby = Prey()
                baby.position = (self.position + partner.position) / 2
                prey_list.append(baby)
                sim.current_prey_births += 1
                self.energy = 30
                partner.energy = 30
                return

        nearest_predator = self._find_nearest_predator(predators)
        nearest_food = self._find_nearest_food(sim.food_list)

        if nearest_predator:
            self.flee_from(nearest_predator)

        elif nearest_food in sim.food_list:
            self.move_toward(nearest_food)
            self.eat(nearest_food, sim)

        else:
            if sim.flocking_enabled:
                self.flocking(prey_list)

        self.avoid_obstacles(sim.obstacle_list)
        self.update_position()

    def _find_nearest_predator(self, predators):
        nearest = None
        min_distance = self.vision_radius
        for predator in predators:
            distance = self.position.distance_to(predator.position)
            if distance < min_distance:
                min_distance = distance
                nearest = predator
        return nearest
    
    def move_toward(self, target):
        direction = target.position - self.position
        if direction.length_squared() > 0.1:
            self.velocity = direction.normalize()


    def get_neighbors(self, prey_list, radius=60):
        neighbors = []
        for other in prey_list:
            if other is self:
                continue
            if self.position.distance_to(other.position) < radius and len(neighbors)<30:
                neighbors.append(other)
        return neighbors


    def separation(self, neighbors, desired_distance=20):
        steer = pygame.Vector2(0, 0)
        count = 0

        for other in neighbors:
            d = self.position.distance_to(other.position)
            if d < desired_distance and d > 0:
                steer += (self.position - other.position).normalize() / d
                count += 1

        if count > 0:
            steer /= count

        return steer


    def alignment(self, neighbors):
        if not neighbors:
            return pygame.Vector2(0, 0)

        avg_vel = pygame.Vector2(0, 0)

        for other in neighbors:
            avg_vel += other.velocity

        avg_vel /= len(neighbors)
        return (avg_vel - self.velocity)


    def cohesion(self, neighbors):
        if not neighbors:
            return pygame.Vector2(0, 0)

        center = pygame.Vector2(0, 0)

        for other in neighbors:
            center += other.position

        center /= len(neighbors)
        return (center - self.position)

    def _find_nearest_food(self, food_list):
        return min(food_list, key=lambda prey: self.position.distance_to(prey.position), default=None)

    def eat(self, food, sim):
        if self.position.distance_to(food.position) < 10:
            self.energy += 20
            self.energy = min(self.energy, 150)
            sim.food_list.remove(food)
            return True
        return False

    def flocking(self, prey_list):
        neighbors = self.get_neighbors(prey_list, radius=60)

        if not neighbors:
            self.velocity.rotate_ip(random.uniform(-15, 15))
            return


        sep = self.separation(neighbors) * 1.8
        ali = self.alignment(neighbors) * 1.0
        coh = self.cohesion(neighbors) * 0.8

        self.velocity += sep + ali + coh

        if self.velocity.length() > 0:
            self.velocity = self.velocity.normalize() * self.speed

        scaling = 0.15
        new_speed = 2 + len(neighbors) * scaling
        self.speed = min(new_speed, 4) 


    def flee_from(self, predator):
        self.velocity = (self.position - predator.position).normalize()

    def draw(self):
        pygame.draw.circle(screen, self.color, (int(self.position.x), int(self.position.y)), 4)
        self.draw_trail()


class Predator(Agent):
    def __init__(self):
        super().__init__(speed=3, color=PREDATOR_COLOR)

    def update(self, prey_list, predator_list):
        self.energy -= 0.05
        if self.mating:
            self.mating_timer -= 1
            if self.mating_timer <= 0:
                self.mating = False
                if self.partner:
                    self.partner.mating = False
                    self.partner.partner = None
                    self.partner = None
            return

        if self.energy >= sim.predator_repro_treshold and len(predator_list) < 5:
            partner = None
            min_dist = 20
            for other in predator_list:
                if other is self or other.mating:
                    continue
                if other.energy < other.repro_threshold:
                    continue
                d = self.position.distance_to(other.position)
                if d < min_dist:
                    partner = other
                    min_dist = d
            if partner:
                self.mating = True
                partner.mating = True
                self.mating_timer = self.mating_duration
                partner.mating_timer = partner.mating_duration
                self.partner = partner
                partner.partner = self
                baby = Predator()
                baby.position = (self.position + partner.position) / 2
                predator_list.append(baby)
                sim.current_pred_births += 1
                self.energy = 30
                partner.energy = 30
                return
        if prey_list:
            nearest_prey = self._find_nearest_prey(prey_list)
            count=0
            if nearest_prey and count<10:
                self.hunt(nearest_prey)
                count+=1
            else:
                self.avoid_obstacles(sim.obstacle_list)
                self.update_position()
                count=0

    def _find_nearest_prey(self, prey_list):
        return min(prey_list, key=lambda prey: self.position.distance_to(prey.position), default=None)

    def hunt(self, prey):
        self.velocity = (prey.position - self.position).normalize()
        self.update_position()

    def draw(self):
        angle = self.velocity.angle_to(pygame.math.Vector2(1, 0))
        points = [pygame.math.Vector2(10, 0), pygame.math.Vector2(-5, -5), pygame.math.Vector2(-5, 5)]
        rotated_points = [self.position + p.rotate(-angle) for p in points]
        pygame.draw.polygon(screen, self.color, rotated_points)
        self.draw_trail()


class Simulation:
    def __init__(self, num_prey=50, num_predators=3, num_food=50):
        self.flocking_enabled = True
        self.prey_repro_treshold = 90
        self.predator_repro_treshold = 90
        
        self.obstacle_list = [
            Obstacle(200, 200, 25),
            Obstacle(500, 350, 40),
            Obstacle(650, 120, 30)
        ]  
        self.prey_list = [Prey() for _ in range(num_prey)]
        self.predator_list = [Predator() for _ in range(num_predators)]
        self.food_list = [Food(self.obstacle_list) for _ in range(num_food)]
        self.running = True
        self.food_spawn_timer = 0
        self.food_spawn_interval = 15   
        #pt tracking
        self.history_time = []
        self.history_prey = []
        self.history_pred = []
        self.history_food = []

        self.births_prey = []
        self.births_pred = []

        self.current_prey_births = 0
        self.current_pred_births = 0

        self.time_step = 0
      

    def record_history(self):
        self.history_time.append(self.time_step)
        self.history_prey.append(len(self.prey_list))
        self.history_pred.append(len(self.predator_list))
        self.history_food.append(len(self.food_list))

        
        self.births_prey.append(self.current_prey_births)
        self.births_pred.append(self.current_pred_births)

        
        self.current_prey_births = 0
        self.current_pred_births = 0

        self.time_step += 1

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.add_prey()
                elif event.key == pygame.K_o:
                    self.add_predator()
                elif event.key == pygame.K_f:
                    self.add_food()
                elif event.key == pygame.K_i:
                    self.flocking_enabled = not self.flocking_enabled
                elif event.key == pygame.K_UP:
                    self.prey_repro_treshold += 5
                elif event.key == pygame.K_DOWN:
                    self.prey_repro_treshold = max(10, self.prey_repro_treshold - 5)
                elif event.key == pygame.K_RIGHT:
                    self.predator_repro_treshold += 5
                elif event.key == pygame.K_LEFT:
                    self.predator_repro_treshold = max(10, self.predator_repro_treshold - 5)


    def run(self):
        while self.running:
            clock.tick(FPS)
            self.handle_events()
            self.update_agents()
            self.handle_collisions()
            self.render()
        pygame.quit()

    def add_prey(self):
        self.prey_list.append(Prey())

    def add_predator(self):
        self.predator_list.append(Predator())

    def add_food(self):
        if len(self.food_list) < 100:
            self.food_list.append(Food(self.obstacle_list))


    def update_agents(self):
        for prey in self.prey_list:
            prey.update(self.predator_list, self.prey_list, self)
        for predator in self.predator_list:
            predator.update(self.prey_list, self.predator_list)
        self.prey_list = [prey for prey in self.prey_list if prey.energy > 0]
        self.predator_list = [pred for pred in self.predator_list if pred.energy > 0]

        self.food_spawn_timer += 1
        if self.food_spawn_timer >= self.food_spawn_interval:
            self.add_food()
            self.food_spawn_timer = 0
        self.record_history()

    def handle_collisions(self):
        for predator in self.predator_list:
            for prey in self.prey_list[:]:
                if predator.position.distance_to(prey.position) < 6:
                    self.prey_list.remove(prey)
                    predator.energy += 30
                    predator.energy = min(predator.energy, 150)

    def render(self):
        screen.fill(BACKGROUND_COLOR)
        self.draw_legend()
        self.draw_stats()
        for obs in self.obstacle_list:
            obs.draw()
        for food in self.food_list:
            food.draw()
        for prey in self.prey_list:
            prey.draw()
        for predator in self.predator_list:
            predator.draw()
        pygame.display.flip()


    def generate_graphs(self):
        plt.figure(figsize=(10, 8))
        plt.subplot(2,1,1)
        plt.plot(self.history_time, self.history_prey, label="Populatie prada")
        plt.plot(self.history_time, self.history_pred, label="Populatie pradatori")
        plt.xlabel("Timpul")
        plt.ylabel("Marimea populatiei")
        plt.legend()
        plt.title("Population Over Time")

        plt.subplot(2,1,2)
        plt.plot(self.history_time, self.births_prey, label="Nasterea prazii in timp")
        plt.plot(self.history_time, self.births_pred, label="Nasterea pradatorilor in timp")
        plt.xlabel("Timpul")
        plt.ylabel("Nasteri")
        plt.legend()
        plt.title("Birth rates over time")
        plt.show()


    def draw_legend(self):
        screen.blit(FONT.render('Add prey (Orange Circle) - P add', True, PREY_COLOR), (10, 10))
        screen.blit(FONT.render('Add predator (Red Triangle) - O add', True, PREDATOR_COLOR), (10, 30))
        screen.blit(FONT.render('Add food (Small Green) - F add', True, FOOD_COLOR), (10, 50))

    def draw_stats(self):
        x = 10  # left side
        y_start = 70
        y_offset = 20

        screen.blit(FONT.render(f'Prey Count: {len(self.prey_list)}', True, TEXT_COLOR), (x, y_start))
        screen.blit(FONT.render(f'Predator Count: {len(self.predator_list)}', True, TEXT_COLOR), (x, y_start + y_offset))
        screen.blit(FONT.render(f'Food Count: {len(self.food_list)}', True, TEXT_COLOR), (x, y_start + 2 * y_offset))
        screen.blit(FONT.render(f'Flocking Toggle - I: {"ON" if self.flocking_enabled else "OFF"}', True, TEXT_COLOR), (x, y_start + 3 * y_offset))
        screen.blit(FONT.render(f'Prey Repro Treshold Adjustment - UP/DOWN: {self.prey_repro_treshold}', True, TEXT_COLOR), (x, y_start + 4 * y_offset))
        screen.blit(FONT.render(f'Predator Repro Treshold Adjustment - LEFT/RIGHT: {self.predator_repro_treshold}', True, TEXT_COLOR), (x, y_start + 5 * y_offset))


if __name__ == "__main__":
    sim = Simulation()
    sim.run()

    sim.generate_graphs()
