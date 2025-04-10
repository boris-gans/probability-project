import simpy
import numpy as np
import random2 as random
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

# <><><><><><><><><><><><><><><><><><><><><><><><><><<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><<><><><><><><><><><><><><>
INITIAL_PEOPLE_ENTRANCE = 0
INITIAL_PEOPLE_METRO = 0
INITIAL_PEOPLE_CERCANIAS = 0

# Capacity limits
METRO_CAPACITY = 1128
# ~22k people per day / 19.5 operational hours = 1128 days
CERCANIAS_CAPACITY = 750 

# Operating hours
START_HOUR = 6.0    # 6:00 AM
END_HOUR = 25.5     # 1:30 AM (next day)

# Flow probabilities between areas
# From Metro
METRO_TO_METRO_PROB = 0.45     # Stay in Metro
METRO_TO_CERCANIAS_PROB = 0.25  # Go to Cercanias
METRO_TO_ENTRANCE_PROB = 0.30   # Go to Entrance

# From Entrance
ENTRANCE_TO_EXIT_PROB = 1.00   # Only for people that have the intention of leaving, not for people generated here.
ENTRANCE_TO_ENTRANCE_PROB = 0.05   # Stay in Entrance
ENTRANCE_TO_METRO_PROB = 0.60     # Go to Metro
ENTRANCE_TO_CERCANIAS_PROB = 0.35 # Go to Cercanias

# From Cercanias
CERCANIAS_TO_CERCANIAS_PROB = 0.02    # Stay in Cercanias
CERCANIAS_TO_METRO_PROB = 0.588       # Go to Metro
CERCANIAS_TO_ENTRANCE_PROB = 0.392    # Go to Entrance

# Mu and Llambda for each area 
# (people per hour / 60 minutes = people per minute)
METRO_1_LAM = 734 / 60 #~14k people per day (both directions)
METRO_2_LAM = 435 / 60 #~8.5k people per day (both directions)
METRO_3_LAM = 350 / 60 #~7k people per day (both directions)
CERCANIAS_LAM = 657 / 60 #~12k people per day (both directions)
ENTRANCE_LAM = 292 / 60 #~6k people per day 

METRO_1_MU = 1 / METRO_1_LAM
METRO_2_MU = 1 / METRO_2_LAM
METRO_3_MU = 1 / METRO_3_LAM
CERCANIAS_MU = 1 / CERCANIAS_LAM
ENTRANCE_MU = 1 / ENTRANCE_LAM

# Capacity for each area (people per minute)
METRO_1_CAPACITY = 250 #We assume that metro's arrive every 4 minutes and can take 300 people for two directions
METRO_2_CAPACITY = 250 #We assume that metro's arrive every 4 minutes and can take 300 people for two directions
METRO_3_CAPACITY = 250 #We assume that metro's arrive every 4 minutes and can take 300 people for two directions
CERCANIAS_CAPACITY = 220 #We assume that cercanias arrive every 10 minutes and can take 600 people for two directions
ENTRANCE_CAPACITY = 100 #We know that their are a total of 25 turnstiles to enter the station (10 for metro, 15 for cercanias)

# Simulation time step (in minutes)
STEP_INTERVAL = 10  # Minutes between each flow calculation

# Visualization parameters
DISPLAY_INTERVAL = 60  # Minutes between status updates
PLOT_TARGET_HOURS = [6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 25.5]  # Hours to analyze in charts

class Person:
    """Represents a person moving through the station"""
    def __init__(self, env, id, source="Generated", destination=None, leaving=False):
        self.env = env
        self.id = id
        self.entry_time = env.now
        self.source = source
        self.destination = destination
        self.current_area = None
        self.wait_start = None
        self.wait_time = 0  # Total wait time in minutes
        self.leaving = leaving #Flag denoting if someone is on their way out through the Entrance 
        
    def start_waiting(self):
        """Mark the start of waiting time"""
        self.wait_start = self.env.now
        
    def stop_waiting(self):
        """Record the waiting time for an area"""
        if self.wait_start is not None:
            self.wait_time += (self.env.now - self.wait_start)
            self.wait_start = None

class Area:
    """Base class for areas in the station"""
    def __init__(self, env, name, arrival_rate, service_rate, capacity, station):
        self.env = env
        self.name = name
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.server = simpy.Resource(env, capacity=capacity)
        self.queue_length = []
        self.queue_times = []
        self.people_count = 0
        self.served_count = 0
        self.station = station
        self.server_busy_time = 0
        self.last_state_change = 0
        self.utilization_data = []
        self.people_data = []
        self.generation_data = 0
        
        # Start the arrival process
        if self.arrival_rate > 0:
            self.env.process(self.generate_arrivals())
            
        # Start the metrics collection process
        self.env.process(self.collect_metrics())
        
    def generate_arrivals(self):
        """Generate new people arriving to this area"""
        person_id = 0
        while True:
            # Wait for next arrival
            interarrival_time = random.expovariate(self.arrival_rate)
            yield self.env.timeout(interarrival_time)
            
            # Create new person
            person_id += 1
            person = Person(self.env, f"{self.name}_{person_id}")
            person.current_area = self.name
            
            # Send to service process
            self.people_count += 1
            self.generation_data += 1
            self.env.process(self.serve_person(person))
            
    def serve_person(self, person):
        """Process a person through this area"""
        arrival_time = self.env.now
        
        # Record when person starts waiting
        person.start_waiting()
        
        # Record queue length upon arrival
        queue_len = len(self.server.queue) + 1
        self.queue_length.append(queue_len)
        
        # Request service
        with self.server.request() as request:
            yield request
            
            # Mark end of waiting
            person.stop_waiting()
            
            # Record wait time for this area
            wait_time = self.env.now - arrival_time
            self.queue_times.append(wait_time)
            
            # Service time follows exponential distribution
            service_time = random.expovariate(self.service_rate)
            yield self.env.timeout(service_time)
            
            self.served_count += 1
            
            # Determine next destination
            self.people_count -= 1
            self.decide_next_destination(person)
            
    def decide_next_destination(self, person):
        """Determine where the person goes next"""
        # To be implemented by subclasses
        pass
    
    def collect_metrics(self):
        """Collect metrics on a regular interval"""
        while True:
            yield self.env.timeout(STEP_INTERVAL)
            
            # Record utilization (busy servers / total servers)
            current_hour = self.env.now / 60
            utilization = self.server.count / self.server.capacity
            self.utilization_data.append((current_hour, utilization))
            
            # Record people count
            self.people_data.append((current_hour, self.people_count))
            # self.generation_data.append((self.name, self.people_count))
            # print(f"People in {self.name} at hour {current_hour}: {self.people_count}")

class Metro(Area):
    def __init__(self, env, station):
        self.metro_to_metro_count = 0
        self.total_count = 0

        # Combine all metro arrival rates
        total_metro_arrival_rate = METRO_1_LAM + METRO_2_LAM + METRO_3_LAM
        
        # Use weighted average of service rates
        avg_service_rate = (
            METRO_1_LAM * METRO_1_MU + 
            METRO_2_LAM * METRO_2_MU + 
            METRO_3_LAM * METRO_3_MU
        ) / total_metro_arrival_rate
        
        # Combined capacity
        total_capacity = METRO_1_CAPACITY + METRO_2_CAPACITY + METRO_3_CAPACITY
        
        super().__init__(env, "Metro", total_metro_arrival_rate, avg_service_rate, 
                        total_capacity, station)
    
    def decide_next_destination(self, person):
        """Determine where a person goes after Metro"""
        rand = random.random()
        self.total_count +=1

        if rand < METRO_TO_METRO_PROB:
            # Stay in Metro
            self.metro_to_metro_count +=1
            self.station.metro.people_count +=1
            self.env.process(self.serve_person(person))
        elif rand < METRO_TO_METRO_PROB + METRO_TO_CERCANIAS_PROB:
            # Go to Cercanias
            person.current_area = "Cercanias"
            self.station.cercanias.people_count += 1  # Increment count in new area before starting service
            self.env.process(self.station.cercanias.serve_person(person))
        else:
            # Go to Entrance and set leaving flag
            person.current_area = "Entrance"
            person.leaving = True
            self.station.entrance.people_count += 1  # Increment count in new area before starting service
            self.env.process(self.station.entrance.serve_person(person))
        # print(f"Total: {self.total_count} Metro to Metro: {self.metro_to_metro_count}")

class Cercanias(Area):
    def __init__(self, env, station):
        super().__init__(env, "Cercanias", CERCANIAS_LAM, CERCANIAS_MU, 
                        CERCANIAS_CAPACITY, station)
    
    def decide_next_destination(self, person):
        """Determine where a person goes after Cercanias"""
        rand = random.random()
        
        if rand < CERCANIAS_TO_CERCANIAS_PROB:
            # Stay in Cercanias
            self.station.cercanias.people_count += 1
            self.env.process(self.serve_person(person))
        elif rand < CERCANIAS_TO_CERCANIAS_PROB + CERCANIAS_TO_METRO_PROB:
            # Go to Metro
            person.current_area = "Metro"
            self.station.metro.people_count += 1  # Increment count in new area before starting service
            self.env.process(self.station.metro.serve_person(person))
        else:
            # Go to Entrance and set leaving flag
            person.current_area = "Entrance"
            person.leaving = True
            self.station.entrance.people_count += 1  # Increment count in new area before starting service
            self.env.process(self.station.entrance.serve_person(person))

class Entrance(Area):
    def __init__(self, env, station):
        super().__init__(env, "Entrance", ENTRANCE_LAM, ENTRANCE_MU, 
                        ENTRANCE_CAPACITY, station)
        self.exit_count = 0
    
    def decide_next_destination(self, person):
        """Determine where a person goes after Entrance"""
        # If person is leaving, they exit the system
        if person.leaving:
            self.exit_count += 1
            self.station.sink.enter(person)
            return
        
        # Otherwise determine next destination
        rand = random.random()
        
        if rand < ENTRANCE_TO_ENTRANCE_PROB:
            # Stay in Entrance
            self.station.entrance.people_count += 1
            self.env.process(self.serve_person(person))
        elif rand < ENTRANCE_TO_ENTRANCE_PROB + ENTRANCE_TO_METRO_PROB:
            # Go to Metro
            person.current_area = "Metro"
            self.station.metro.people_count += 1  # Increment count in new area before starting service
            self.env.process(self.station.metro.serve_person(person))
        else:
            # Go to Cercanias
            person.current_area = "Cercanias"
            self.station.cercanias.people_count += 1  # Increment count in new area before starting service
            self.env.process(self.station.cercanias.serve_person(person))

class Sink:
    def __init__(self, env):
        self.env = env
        self.departed = 0
        self.exit_times = []

    def enter(self, person):
        self.departed += 1
        wait_time = person.wait_time
        self.exit_times.append((self.env.now, wait_time))

class StationSimulation:
    def __init__(self, env):
        self.env = env
        self.sink = Sink(env)
        self.hourly_data = defaultdict(lambda: defaultdict(int))

        self.env.process(self.collect_hourly_metrics())
        
        # Create areas
        self.metro = Metro(env, self)
        self.cercanias = Cercanias(env, self)
        self.entrance = Entrance(env, self)
        
        # Create metrics collection
        
        # Data storage
    
    def collect_hourly_metrics(self):
        """Collect metrics every hour"""
        while True:
            yield self.env.timeout(60)  # Every hour
            
            hour = int(self.env.now / 60)
            print(f"Hour {hour}: metro {self.metro.people_count} cerc {self.cercanias.people_count} entrance {self.entrance.people_count}")

            self.hourly_data[hour]['metro_count'] = self.metro.people_count
            self.hourly_data[hour]['cercanias_count'] = self.cercanias.people_count
            self.hourly_data[hour]['entrance_count'] = self.entrance.people_count
            self.hourly_data[hour]['total_count'] = (self.metro.people_count + 
                                                   self.cercanias.people_count + 
                                                   self.entrance.people_count)
            self.hourly_data[hour]['exits'] = self.sink.departed
    
    def calculate_metrics(self):
        """Calculate final metrics after simulation"""
        metrics = {}
        
        # Average queue length
        metrics['avg_queue_length_metro'] = np.mean(self.metro.queue_length) if self.metro.queue_length else 0
        metrics['avg_queue_length_cercanias'] = np.mean(self.cercanias.queue_length) if self.cercanias.queue_length else 0
        metrics['avg_queue_length_entrance'] = np.mean(self.entrance.queue_length) if self.entrance.queue_length else 0
        
        # Average queue waiting time
        metrics['avg_wait_time_metro'] = np.mean(self.metro.queue_times) if self.metro.queue_times else 0
        metrics['avg_wait_time_cercanias'] = np.mean(self.cercanias.queue_times) if self.cercanias.queue_times else 0
        metrics['avg_wait_time_entrance'] = np.mean(self.entrance.queue_times) if self.entrance.queue_times else 0
        
        # Server utilization calculation
        total_time = self.env.now
        metro_busy_time = sum(self.metro.server.count for _ in range(len(self.metro.utilization_data))) / len(self.metro.utilization_data) if self.metro.utilization_data else 0
        cercanias_busy_time = sum(self.cercanias.server.count for _ in range(len(self.cercanias.utilization_data))) / len(self.cercanias.utilization_data) if self.cercanias.utilization_data else 0
        entrance_busy_time = sum(self.entrance.server.count for _ in range(len(self.entrance.utilization_data))) / len(self.entrance.utilization_data) if self.entrance.utilization_data else 0
        
        metrics['utilization_metro'] = metro_busy_time / self.metro.server.capacity if self.metro.server.capacity > 0 else 0
        metrics['utilization_cercanias'] = cercanias_busy_time / self.cercanias.server.capacity if self.cercanias.server.capacity > 0 else 0
        metrics['utilization_entrance'] = entrance_busy_time / self.entrance.server.capacity if self.entrance.server.capacity > 0 else 0
        
        # Total counts
        metrics['total_served_metro'] = self.metro.served_count
        metrics['total_served_cercanias'] = self.cercanias.served_count
        metrics['total_served_entrance'] = self.entrance.served_count
        metrics['total_exits'] = self.sink.departed

        metrics['total_generated_metro'] = self.metro.generation_data
        metrics['total_generated_cercanias'] = self.cercanias.generation_data
        metrics['total_generated_entrance'] = self.entrance.generation_data

        
        return metrics
    
    def plot_results(self):
        """Plot simulation results"""
        metrics = self.calculate_metrics()
        
        # Convert hourly data to DataFrame
        df = pd.DataFrame.from_dict(self.hourly_data, orient='index')
        
        # Plot 1: People count per area over time
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(df.index, df.metro_count, label='Metro')
        plt.plot(df.index, df.cercanias_count, label='Cercanias')
        plt.plot(df.index, df.entrance_count, label='Entrance')
        plt.plot(df.index, df.total_count, label='Total', linestyle='--')
        plt.title('People Count Over Time')
        plt.xlabel('Hour')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Server utilization
        plt.subplot(2, 2, 2)
        utilization_data = {
            'Metro': metrics['utilization_metro'],
            'Cercanias': metrics['utilization_cercanias'],
            'Entrance': metrics['utilization_entrance']
        }
        areas = list(utilization_data.keys())
        values = list(utilization_data.values())
        max_val = max(metrics['utilization_metro'], metrics['utilization_cercanias'], metrics['utilization_entrance'])
        plt.bar(areas, values)
        plt.title('Server Utilization')
        plt.xlabel('Area')
        plt.ylabel('Utilization')
        plt.ylim(0, max_val)
        
        # Plot 3: Average queue length
        plt.subplot(2, 2, 3)
        queue_length_data = {
            'Metro': metrics['avg_queue_length_metro'],
            'Cercanias': metrics['avg_queue_length_cercanias'],
            'Entrance': metrics['avg_queue_length_entrance']
        }
        areas = list(queue_length_data.keys())
        values = list(queue_length_data.values())
        plt.bar(areas, values)
        plt.title('Average Queue Length')
        plt.xlabel('Area')
        plt.ylabel('People')
        
        # Plot 4: Average waiting time
        plt.subplot(2, 2, 4)
        wait_time_data = {
            'Metro': metrics['avg_wait_time_metro'],
            'Cercanias': metrics['avg_wait_time_cercanias'],
            'Entrance': metrics['avg_wait_time_entrance']
        }
        areas = list(wait_time_data.keys())
        values = list(wait_time_data.values())
        plt.bar(areas, values)
        plt.title('Average Wait Time')
        plt.xlabel('Area')
        plt.ylabel('Minutes')
        
        plt.tight_layout()
        plt.savefig('station_simulation_results.png')
        plt.show()
        
        # Print summary metrics
        print("\nSimulation Summary Metrics:")
        print(f"Total people served - Metro: {metrics['total_served_metro']}")
        print(f"Total people served - Cercanias: {metrics['total_served_cercanias']}")
        print(f"Total people served - Entrance: {metrics['total_served_entrance']}")
        print(f"Total exits: {metrics['total_exits']}")
        print(f"Average queue length - Metro: {metrics['avg_queue_length_metro']:.2f}")
        print(f"Average queue length - Cercanias: {metrics['avg_queue_length_cercanias']:.2f}")
        print(f"Average queue length - Entrance: {metrics['avg_queue_length_entrance']:.2f}")
        print(f"Average wait time - Metro: {metrics['avg_wait_time_metro']:.2f} minutes")
        print(f"Average wait time - Cercanias: {metrics['avg_wait_time_cercanias']:.2f} minutes")
        print(f"Average wait time - Entrance: {metrics['avg_wait_time_entrance']:.2f} minutes")
        print(f"Server utilization - Metro: {metrics['utilization_metro']:.2%}")
        print(f"Server utilization - Cercanias: {metrics['utilization_cercanias']:.2%}")
        print(f"Server utilization - Entrance: {metrics['utilization_entrance']:.2%}")
        print(f"Total people generated - Metro: {metrics['total_generated_metro']}")
        print(f"Total people generated - Cercanias: {metrics['total_generated_cercanias']}")
        print(f"Total people generated - Entrance: {metrics['total_generated_entrance']}")


def run_simulation(sim_duration=24*60):
    """Run the simulation for the specified duration in minutes"""
    # Create SimPy environment
    env = simpy.Environment()
    
    # Create and run simulation
    simulation = StationSimulation(env)
    
    # Run for specified duration (minutes)
    env.run(until=sim_duration)
    
    # Plot results
    simulation.plot_results()
    
    return simulation

if __name__ == "__main__":
    # Run for 24 hours (1440 minutes)
    sim = run_simulation(1440)
