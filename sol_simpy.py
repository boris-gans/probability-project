import simpy
import numpy as np
import random2 as random
import matplotlib.pyplot as plt
from scipy import stats
from collection import defaultdict
import pandas as pd

# Simulation parameters (easy to modify)
# --------------------------------------
# Initial values
INITIAL_PEOPLE_ENTRANCE = 100
INITIAL_PEOPLE_METRO = 100
INITIAL_PEOPLE_CERCANIAS = 100

# Capacity limits
METRO_CAPACITY = 500
CERCANIAS_CAPACITY = 400

# Operating hours
START_HOUR = 6.0    # 6:00 AM
END_HOUR = 25.5     # 1:30 AM (next day)

# Time distribution parameters
PEAK_HOUR = 17.0    # 5:00 PM
SKEW_PARAM = 5      # Controls skewness, positive for right-skew
SCALE_PARAM = 3.0   # Scale parameter (spread of distribution)
# IN THE NOTEBOOK LETS VISUALIZE THIS DISTRIBUTION AND COMPARE W GOOGLE MAPS

# Flow probabilities between areas
# From Metro
METRO_TO_METRO_PROB = 0.45
METRO_TO_CERCANIAS_PROB = 0.25
METRO_TO_ENTRANCE_PROB = 0.30

# From Entrance
ENTRANCE_TO_ENTRANCE_PROB = 0.03
ENTRANCE_TO_METRO_PROB = 0.776
ENTRANCE_TO_CERCANIAS_PROB = 0.194

# From Cercanias
CERCANIAS_TO_CERCANIAS_PROB = 0.02
CERCANIAS_TO_METRO_PROB = 0.588
CERCANIAS_TO_ENTRANCE_PROB = 0.392

# Arrival and departure rates
METRO_ARRIVAL_RATE = 40
ENTRANCE_ARRIVAL_RATE = 30
CERCANIAS_ARRIVAL_RATE = 35

METRO_DEPARTURE_RATE = 35
ENTRANCE_DEPARTURE_RATE = 25
CERCANIAS_DEPARTURE_RATE = 30

# Simulation time step (in minutes)
STEP_INTERVAL = 10  # Minutes between each flow calculation

# Visualization parameters
DISPLAY_INTERVAL = 60  # Minutes between status updates
PLOT_TARGET_HOURS = [6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 25.5]  # Hours to analyze in charts

class TimeOfDay:
    """Models time of day and provides factors to adjust flow rates"""
    def __init__(self, start_hour=START_HOUR, end_hour=END_HOUR, peak_hour=PEAK_HOUR, 
                 skew_param=SKEW_PARAM, scale_param=SCALE_PARAM):
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.peak_hour = peak_hour
        self.operating_hours = end_hour - start_hour
        self.skew_param = skew_param
        self.scale_param = scale_param
        
        # Cached distribution values
        self._cached_x = np.linspace(0, 1, 1000)
        pdf = stats.skewnorm.pdf(self._cached_x, skew_param, loc=peak_hour/24, scale=scale_param/24)
        self._cached_pdf = pdf / pdf.max()  # Normalize to max 1.0
    
    def get_time_factor(self, hour):
        """Get the busyness factor based on time of day"""
        if hour < self.start_hour or hour > self.end_hour:
            return 0.0
        
        # Find the closest value in the distribution
        normalized_hour = (hour - self.start_hour) / self.operating_hours
        idx = int(normalized_hour * 1000)
        idx = max(0, min(idx, 999))  # Ensure index is within bounds
        
        return self._cached_pdf[idx]
    
    def format_hour(self, hour):
        """Convert decimal hour to HH:MM format"""
        is_next_day = hour >= 24
        hour_adjusted = hour % 24
        
        hours = int(hour_adjusted)
        minutes = int((hour_adjusted - hours) * 60)
        
        if is_next_day:
            return f"{hours:02d}:{minutes:02d} (next day)"
        else:
            return f"{hours:02d}:{minutes:02d}"

class StationArea:
    """Base class for station areas (Entrance, Metro, Cercanias)"""
    def __init__(self, env, name, initial_people, capacity=float('inf')):
        self.env = env
        self.name = name
        self.resource = simpy.Container(env, init=initial_people, capacity=capacity)
        self.waiting_people = 0
        self.turned_away = 0
        self.queue_times = []
        self.metrics = {
            'queue_length': [],
            'time': [],
            'occupancy': [],
            'turned_away_count': [],
            'entries': defaultdict(int),
            'exits': defaultdict(int),
            'rejected': defaultdict(int)
        }
    
    def log_metrics(self):
        """Record current metrics"""
        current_time = self.env.now / 60.0  # Convert minutes to hours
        self.metrics['queue_length'].append(self.waiting_people)
        self.metrics['time'].append(current_time)
        self.metrics['occupancy'].append(self.resource.level)
        self.metrics['turned_away_count'].append(self.turned_away)
        
    def reset_flow_metrics(self):
        """Reset flow metrics for the next time period"""
        self.turned_away = 0
        self.metrics['entries'] = defaultdict(int)
        self.metrics['exits'] = defaultdict(int)
        self.metrics['rejected'] = defaultdict(int)

class Entrance(StationArea):
    """Entrance area of the station"""
    def __init__(self, env, initial_people=INITIAL_PEOPLE_ENTRANCE):
        super().__init__(env, "Entrance", initial_people)

class Metro(StationArea):
    """Metro area of the station"""
    def __init__(self, env, initial_people=INITIAL_PEOPLE_METRO, capacity=METRO_CAPACITY):
        super().__init__(env, "Metro", initial_people, capacity)

class Cercanias(StationArea):
    """Cercanias area of the station"""
    def __init__(self, env, initial_people=INITIAL_PEOPLE_CERCANIAS, capacity=CERCANIAS_CAPACITY):
        super().__init__(env, "Cercanias", initial_people, capacity)

class Person:
    """Represents a person moving through the station"""
    def __init__(self, env, id, source="Outside"):
        self.env = env
        self.id = id
        self.entry_time = env.now
        self.source = source
        self.destination = None
        self.current_area = None
        self.wait_start = None
        
    def start_waiting(self):
        """Mark the start of waiting time"""
        self.wait_start = self.env.now
        
    def stop_waiting(self, area):
        """Record the waiting time for an area"""
        if self.wait_start is not None:
            wait_time = self.env.now - self.wait_start
            area.queue_times.append(wait_time)
            self.wait_start = None

class StationSimulation:
    """Main simulation class for the Sol Metro station"""
    def __init__(self, env, time_system=None):
        self.env = env
        self.time_system = time_system or TimeOfDay()
        
        # Create station areas
        self.entrance = Entrance(env)
        self.metro = Metro(env)
        self.cercanias = Cercanias(env)
        
        # Keep track of people and timing
        self.person_count = 0
        self.hourly_data = []
        
        # Start processes
        self.processes = [
            # External arrivals
            env.process(self.generate_arrivals(self.metro, METRO_ARRIVAL_RATE, "Outside")),
            env.process(self.generate_arrivals(self.entrance, ENTRANCE_ARRIVAL_RATE, "Outside")),
            env.process(self.generate_arrivals(self.cercanias, CERCANIAS_ARRIVAL_RATE, "Outside")),
            
            # External departures
            env.process(self.generate_departures(self.metro, METRO_DEPARTURE_RATE, "Outside")),
            env.process(self.generate_departures(self.entrance, ENTRANCE_DEPARTURE_RATE, "Outside")),
            env.process(self.generate_departures(self.cercanias, CERCANIAS_DEPARTURE_RATE, "Outside")),
            
            # Internal people flows
            env.process(self.flow_between_areas()),
            
            # Metrics collection
            env.process(self.collect_metrics()),
            env.process(self.display_status())
        ]
        
        # Lists for plotting
        self.time_history = []
        self.metro_history = []
        self.cercanias_history = []
        self.entrance_history = []
    
    def get_current_hour(self):
        """Get the current simulation hour"""
        return self.time_system.start_hour + (self.env.now / 60.0)
    
    def get_time_factor(self):
        """Get time-based activity factor for current hour"""
        current_hour = self.get_current_hour()
        return self.time_system.get_time_factor(current_hour)
    
    def generate_arrivals(self, area, base_rate, source):
        """Generate arrivals from outside the system to an area"""
        while True:
            current_hour = self.get_current_hour()
            
            # Skip if outside operating hours
            if current_hour < self.time_system.start_hour or current_hour > self.time_system.end_hour:
                yield self.env.timeout(STEP_INTERVAL)
                continue
            
            # Calculate arrival rate based on time of day
            time_factor = self.time_system.get_time_factor(current_hour)
            adjusted_rate = base_rate * time_factor * 3.0
            adjusted_rate = max(adjusted_rate, base_rate * 0.2)
            
            # Generate random number of arrivals
            arrivals = random.randint(int(0.7 * adjusted_rate), int(1.3 * adjusted_rate))
            
            # Process arrivals based on capacity
            available_space = area.resource.capacity - area.resource.level
            accepted = min(arrivals, available_space) if area.resource.capacity < float('inf') else arrivals
            rejected = arrivals - accepted
            
            if accepted > 0:
                # Create people and add them to the area
                for _ in range(accepted):
                    person = Person(self.env, self.person_count, source)
                    self.person_count += 1
                    yield area.resource.put(1)
                    area.metrics['entries'][source] += 1
                
            if rejected > 0:
                area.turned_away += rejected
                area.metrics['rejected'][source] += rejected
                
            # Wait until next arrival cycle
            yield self.env.timeout(STEP_INTERVAL)
    
    def generate_departures(self, area, base_rate, destination):
        """Generate departures from an area to outside the system"""
        while True:
            current_hour = self.get_current_hour()
            
            # Skip if outside operating hours
            if current_hour < self.time_system.start_hour or current_hour > self.time_system.end_hour:
                yield self.env.timeout(STEP_INTERVAL)
                continue
            
            # Calculate departure rate based on time of day
            time_factor = self.time_system.get_time_factor(current_hour)
            adjusted_rate = base_rate * time_factor * 3.0
            adjusted_rate = max(adjusted_rate, base_rate * 0.2)
            
            # Generate random number of departures (capped by current population)
            max_departures = min(random.randint(int(0.7 * adjusted_rate), int(1.3 * adjusted_rate)), 
                               area.resource.level)
            
            if max_departures > 0:
                yield area.resource.get(max_departures)
                area.metrics['exits'][destination] += max_departures
            
            # Wait until next departure cycle
            yield self.env.timeout(STEP_INTERVAL)
    
    def flow_between_areas(self):
        """Handle the flow of people between different areas of the station"""
        while True:
            current_hour = self.get_current_hour()
            
            # Skip if outside operating hours
            if current_hour < self.time_system.start_hour or current_hour > self.time_system.end_hour:
                yield self.env.timeout(STEP_INTERVAL)
                continue
            
            # 1. Metro flows
            metro_total = self.metro.resource.level
            if metro_total > 0:
                # Calculate how many people move to each destination
                metro_to_cercanias = int(metro_total * METRO_TO_CERCANIAS_PROB)
                metro_to_entrance = int(metro_total * METRO_TO_ENTRANCE_PROB)
                # Staying in metro is calculated but not needed for movement
                
                # Process Metro → Cercanias
                if metro_to_cercanias > 0:
                    # Remove from Metro
                    yield self.metro.resource.get(metro_to_cercanias)
                    self.metro.metrics['exits']["Cercanias"] += metro_to_cercanias
                    
                    # Add to Cercanias (respecting capacity)
                    available_space = self.cercanias.resource.capacity - self.cercanias.resource.level
                    accepted = min(metro_to_cercanias, available_space)
                    rejected = metro_to_cercanias - accepted
                    
                    if accepted > 0:
                        yield self.cercanias.resource.put(accepted)
                        self.cercanias.metrics['entries']["Metro"] += accepted
                    
                    if rejected > 0:
                        self.cercanias.turned_away += rejected
                        self.cercanias.metrics['rejected']["Metro"] += rejected
                
                # Process Metro → Entrance
                if metro_to_entrance > 0:
                    # Remove from Metro
                    yield self.metro.resource.get(metro_to_entrance)
                    self.metro.metrics['exits']["Entrance"] += metro_to_entrance
                    
                    # Add to Entrance (no capacity limit)
                    yield self.entrance.resource.put(metro_to_entrance)
                    self.entrance.metrics['entries']["Metro"] += metro_to_entrance
            
            # 2. Entrance flows
            entrance_total = self.entrance.resource.level
            if entrance_total > 0:
                # Calculate how many people move to each destination
                entrance_to_metro = int(entrance_total * ENTRANCE_TO_METRO_PROB)
                entrance_to_cercanias = int(entrance_total * ENTRANCE_TO_CERCANIAS_PROB)
                
                # Process Entrance → Metro
                if entrance_to_metro > 0:
                    # Remove from Entrance
                    yield self.entrance.resource.get(entrance_to_metro)
                    self.entrance.metrics['exits']["Metro"] += entrance_to_metro
                    
                    # Add to Metro (respecting capacity)
                    available_space = self.metro.resource.capacity - self.metro.resource.level
                    accepted = min(entrance_to_metro, available_space)
                    rejected = entrance_to_metro - accepted
                    
                    if accepted > 0:
                        yield self.metro.resource.put(accepted)
                        self.metro.metrics['entries']["Entrance"] += accepted
                    
                    if rejected > 0:
                        self.metro.turned_away += rejected
                        self.metro.metrics['rejected']["Entrance"] += rejected
                
                # Process Entrance → Cercanias
                if entrance_to_cercanias > 0:
                    # Remove from Entrance
                    yield self.entrance.resource.get(entrance_to_cercanias)
                    self.entrance.metrics['exits']["Cercanias"] += entrance_to_cercanias
                    
                    # Add to Cercanias (respecting capacity)
                    available_space = self.cercanias.resource.capacity - self.cercanias.resource.level
                    accepted = min(entrance_to_cercanias, available_space)
                    rejected = entrance_to_cercanias - accepted
                    
                    if accepted > 0:
                        yield self.cercanias.resource.put(accepted)
                        self.cercanias.metrics['entries']["Entrance"] += accepted
                    
                    if rejected > 0:
                        self.cercanias.turned_away += rejected
                        self.cercanias.metrics['rejected']["Entrance"] += rejected
            
            # 3. Cercanias flows
            cercanias_total = self.cercanias.resource.level
            if cercanias_total > 0:
                # Calculate how many people move to each destination
                cercanias_to_metro = int(cercanias_total * CERCANIAS_TO_METRO_PROB)
                cercanias_to_entrance = int(cercanias_total * CERCANIAS_TO_ENTRANCE_PROB)
                
                # Process Cercanias → Metro
                if cercanias_to_metro > 0:
                    # Remove from Cercanias
                    yield self.cercanias.resource.get(cercanias_to_metro)
                    self.cercanias.metrics['exits']["Metro"] += cercanias_to_metro
                    
                    # Add to Metro (respecting capacity)
                    available_space = self.metro.resource.capacity - self.metro.resource.level
                    accepted = min(cercanias_to_metro, available_space)
                    rejected = cercanias_to_metro - accepted
                    
                    if accepted > 0:
                        yield self.metro.resource.put(accepted)
                        self.metro.metrics['entries']["Cercanias"] += accepted
                    
                    if rejected > 0:
                        self.metro.turned_away += rejected
                        self.metro.metrics['rejected']["Cercanias"] += rejected
                
                # Process Cercanias → Entrance
                if cercanias_to_entrance > 0:
                    # Remove from Cercanias
                    yield self.cercanias.resource.get(cercanias_to_entrance)
                    self.cercanias.metrics['exits']["Entrance"] += cercanias_to_entrance
                    
                    # Add to Entrance (no capacity limit)
                    yield self.entrance.resource.put(cercanias_to_entrance)
                    self.entrance.metrics['entries']["Cercanias"] += cercanias_to_entrance
            
            # Wait until next flow cycle
            yield self.env.timeout(STEP_INTERVAL)
    
    def collect_metrics(self):
        """Collect metrics every hour"""
        while True:
            # Wait until the next hour
            yield self.env.timeout(60)  # 60 minutes
            
            current_hour = self.get_current_hour()
            
            # Skip if outside operating hours
            if current_hour < self.time_system.start_hour or current_hour > self.time_system.end_hour:
                continue
            
            # Log metrics for each area
            self.metro.log_metrics()
            self.entrance.log_metrics()
            self.cercanias.log_metrics()
            
            # Save population data for plotting
            self.time_history.append(current_hour)
            self.metro_history.append(self.metro.resource.level)
            self.entrance_history.append(self.entrance.resource.level)
            self.cercanias_history.append(self.cercanias.resource.level)
            
            # Save hourly flow data
            hourly_snapshot = {
                "hour": current_hour,
                "formatted_hour": self.time_system.format_hour(current_hour),
                "metro_entries": dict(self.metro.metrics['entries']),
                "entrance_entries": dict(self.entrance.metrics['entries']),
                "cercanias_entries": dict(self.cercanias.metrics['entries']),
                "metro_exits": dict(self.metro.metrics['exits']),
                "entrance_exits": dict(self.entrance.metrics['exits']),
                "cercanias_exits": dict(self.cercanias.metrics['exits']),
                "metro_rejected": dict(self.metro.metrics['rejected']),
                "cercanias_rejected": dict(self.cercanias.metrics['rejected']),
                "metro_queue": self.metro.waiting_people,
                "entrance_queue": self.entrance.waiting_people,
                "cercanias_queue": self.cercanias.waiting_people,
                "metro_wait_time": np.mean(self.metro.queue_times) if self.metro.queue_times else 0,
                "entrance_wait_time": np.mean(self.entrance.queue_times) if self.entrance.queue_times else 0,
                "cercanias_wait_time": np.mean(self.cercanias.queue_times) if self.cercanias.queue_times else 0,
            }
            self.hourly_data.append(hourly_snapshot)
            
            # Reset flow metrics for the next hour
            self.metro.reset_flow_metrics()
            self.entrance.reset_flow_metrics()
            self.cercanias.reset_flow_metrics()
    
    def display_status(self):
        """Display status at regular intervals"""
        while True:
            yield self.env.timeout(DISPLAY_INTERVAL)
            
            current_hour = self.get_current_hour()
            if current_hour < self.time_system.start_hour or current_hour > self.time_system.end_hour:
                continue
                
            formatted_time = self.time_system.format_hour(current_hour)
            
            print(f"\n ========== Simulation Time: {formatted_time} ========== \n")
            print(f"Metro: {self.metro.resource.level} people (capacity: {self.metro.resource.capacity})")
            print(f"Entrance: {self.entrance.resource.level} people")
            print(f"Cercanias: {self.cercanias.resource.level} people (capacity: {self.cercanias.resource.capacity})")
            print(f"Time factor: {self.get_time_factor():.2f}")
            print("===========================================")
    
    def plot_results(self):
        """Plot simulation results"""
        # Create figure for population over time
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Population over time
        ax.plot(self.time_history, self.metro_history, label="Metro", color='blue')
        ax.plot(self.time_history, self.cercanias_history, label="Cercanias", color='green')
        ax.plot(self.time_history, self.entrance_history, label="Entrance", color='orange')
        
        # Format x-axis with time labels
        time_ticks = np.linspace(0, len(self.time_history)-1, min(10, len(self.time_history)), dtype=int)
        if len(time_ticks) > 0:
            time_labels = [self.time_system.format_hour(self.time_history[i]) for i in time_ticks]
            ax.set_xticks([self.time_history[i] for i in time_ticks])
            ax.set_xticklabels(time_labels, rotation=45)
        
        ax.set_title("Population at Sol Station Over Time")
        ax.set_xlabel("Time of Day")
        ax.set_ylabel("Number of People")
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Create plot for waiting times if available
        if any(len(area.queue_times) > 0 for area in [self.metro, self.entrance, self.cercanias]):
            self.plot_waiting_times()
        
        # Create flow analysis charts
        self.plot_flow_charts()
    
    def plot_waiting_times(self):
        """Plot waiting time statistics"""
        # Create figure for waiting times
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract waiting time data
        areas = ["Metro", "Entrance", "Cercanias"]
        wait_data = []
        
        for i, data in enumerate(self.hourly_data):
            wait_data.append({
                'hour': data['hour'],
                'Metro': data['metro_wait_time'],
                'Entrance': data['entrance_wait_time'],
                'Cercanias': data['cercanias_wait_time']
            })
        
        df = pd.DataFrame(wait_data)
        
        # Plot waiting times
        for area in areas:
            ax.plot(df['hour'], df[area], label=area)
        
        # Format x-axis with time labels
        time_ticks = np.linspace(min(df['hour']), max(df['hour']), 10)
        ax.set_xticks(time_ticks)
        ax.set_xticklabels([self.time_system.format_hour(h) for h in time_ticks], rotation=45)
        
        ax.set_title("Average Waiting Times")
        ax.set_xlabel("Time of Day")
        ax.set_ylabel("Average Wait Time (minutes)")
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def plot_flow_charts(self):
        """Plot flow analysis charts for specific hours"""
        # Extract flow data from target hours for comparison
        target_hours = PLOT_TARGET_HOURS
        selected_data = []
        
        for hour in target_hours:
            # Find the closest data point to the target hour
            closest_idx = None
            min_diff = float('inf')
            
            for i, data in enumerate(self.hourly_data):
                diff = abs(data["hour"] - hour)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = i
            
            if closest_idx is not None and min_diff < 1.0:  # Only use if within 1 hour
                selected_data.append(self.hourly_data[closest_idx])
        
        # Create bar charts for each selected hour
        for data in selected_data:
            self.create_entity_flow_chart(data)
    
    def create_entity_flow_chart(self, data):
        """Create flow chart for a specific hour"""
        formatted_hour = data["formatted_hour"]
        
        # Create figure with 3 subplots (one for each entity)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Setup colors for different sources/destinations
        colors = {
            'Outside': 'gray', 
            'Metro': 'blue', 
            'Entrance': 'orange', 
            'Cercanias': 'green',
            'Rejected': 'red'
        }
        
        # Plot for Metro
        self.plot_entity_flows(
            ax1, 
            "Metro", 
            data["metro_entries"], 
            data["metro_exits"], 
            data["metro_rejected"],
            colors
        )
        
        # Plot for Entrance
        self.plot_entity_flows(
            ax2, 
            "Entrance", 
            data["entrance_entries"], 
            data["entrance_exits"], 
            {},  # No capacity limits for entrance
            colors
        )
        
        # Plot for Cercanias
        self.plot_entity_flows(
            ax3, 
            "Cercanias", 
            data["cercanias_entries"], 
            data["cercanias_exits"], 
            data["cercanias_rejected"],
            colors
        )
        
        # Set title for the entire figure
        fig.suptitle(f"People Flow Analysis at {formatted_hour}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)  # Make room for the title
        plt.show()
    
    def plot_entity_flows(self, ax, entity_name, entries, exits, rejected, colors):
        """Plot flow data for one entity"""
        # Bar width
        width = 0.35
        
        # Get the sources and destinations
        sources = list(entries.keys())
        destinations = list(exits.keys())
        
        # Get the values
        entry_values = [entries.get(src, 0) for src in sources]
        exit_values = [exits.get(dest, 0) for dest in destinations]
        
        # Setup x positions
        x_pos_entries = np.arange(len(sources))
        x_pos_exits = np.arange(len(destinations)) + len(sources) + 1  # Add space between groups
        
        # Plot inflows
        entry_bars = ax.bar(x_pos_entries, entry_values, width, label='Entries', color=[colors[src] for src in sources])
        
        # Plot outflows
        exit_bars = ax.bar(x_pos_exits, exit_values, width, label='Exits', color=[colors[dest] for dest in destinations])
        
        # If there are rejected entries, plot them
        if rejected:
            x_pos_rejected = np.arange(len(rejected)) + len(sources) + len(destinations) + 2
            rejected_values = [rejected.get(src, 0) for src in rejected.keys()]
            rejected_bars = ax.bar(x_pos_rejected, rejected_values, width, label='Rejected', color='red')
            
            # Add rejected labels
            ax.set_xticks(np.concatenate([x_pos_entries, x_pos_exits, x_pos_rejected]))
            ax.set_xticklabels(sources + destinations + [f"Rej: {src}" for src in rejected.keys()], rotation=45)
        else:
            # No rejected entries
            ax.set_xticks(np.concatenate([x_pos_entries, x_pos_exits]))
            ax.set_xticklabels(sources + destinations, rotation=45)
        
        # Add labels and title
        ax.set_ylabel('Number of People')
        ax.set_title(f'{entity_name} Flow')
        
        # Add value labels on top of each bar
        for bars in [entry_bars, exit_bars]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{int(height)}', ha='center', va='bottom')
        
        # Add a legend
        ax.legend()

def run_simulation(sim_duration=24*60):
    """Run the simulation for the specified duration in minutes"""
    # Create SimPy environment
    env = simpy.Environment()
    
    # Create time system
    time_system = TimeOfDay()
    
    # Create and run simulation
    simulation = StationSimulation(env, time_system)
    
    # Run for specified duration (minutes)
    env.run(until=sim_duration)
    
    # Plot results
    simulation.plot_results()
    
    return simulation

if __name__ == "__main__":
    # Run for 24 hours (1440 minutes)
    sim = run_simulation(1440)
