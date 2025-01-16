# Group: 28
# Authors: Thieme Brandwagt and Sam Buisman
#=========================================
# Assignment 2 - Aircraft Routing Problem 
#=========================================

# Import necessary packages
from openpyxl import *
from time import *
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#=========================================

# Load airport data into a numpy array
file_path_airports = "DATA/AirportData.xlsx"
airports_df = pd.read_excel(file_path_airports, usecols ="B:U")
airports = airports_df.to_numpy()
print(airports[0][0])
# Create list for all the airports (columns)
cities = airports_df.columns


# Load fleet data into a numpy array
file_path_fleet = "DATA/FleetType.xlsx"
fleet_df = pd.read_excel(file_path_fleet)
fleet_df = fleet_df.set_index("Aircraft Type")
fleet = fleet_df.to_numpy()

# Create list for all the aircraft types (columns)
aircraft_types = fleet_df.columns


# Load demand data into a numpy array
file_path_demand = "DATA/Group28.xlsx"
demand_df = pd.read_excel(file_path_demand, usecols="B:AG", skiprows=4)
demand = demand_df.to_numpy()

# Create a triple dimensional array for demand data (origin x destination x time)
demand_matrix = np.zeros((len(cities), len(cities), demand.shape[1] - 2))
# Populate matrix with all demand from all intervals
for rows in range(len(cities)):
    for columns in range(len(cities)):
        for time_steps in range(int(120/4)):  # 120 hours devided by 4 hour time steps
            demand_matrix[rows][columns][time_steps] = demand[rows * 20 + columns][time_steps + 2]

# Adding the previous two bins to the currect bin to get total demand
def flow(i, j, t):
    return demand_matrix[i,j,t] + 0.2 * (demand_matrix[i,j,t-1]+demand_matrix[i,j,t-2])

# Calculate the distance between two airports
def distance(i, j, R_E=6371):
    if i != j:
        lat_i, long_i = airports[1][i], airports[2][i]
        lat_j, long_j = airports[1][j], airports[2][j]
        delta_lat = lat_j - lat_i
        delta_long = long_j - long_i
        root = math.sin(math.radians(delta_lat)/2)**2 + math.cos(math.radians(lat_i)) \
            * math.cos(math.radians(lat_j)) * math.sin(math.radians(delta_long)/2)**2
        return R_E * 2 * math.asin(math.sqrt(root)) 
    else:
        return 0

# Calculate the profit for each flight leg
def cost_flight_leg(i, j, k):
    if i != j:
        tc = fleet[7][k] * (distance(i, j, R_E=6371) / fleet[0][k])     # formula for the time costs
        fc = ((fleet[8][k] * 1.42) / 1.5) * distance(i, j, R_E=6371)    # formula for the fuel costs
        foc = fleet[6][k]                                               # formula for the fixed operating costs
        return tc + fc + foc
# ---------------------------------------------------------------------------------------------------------------------
# print(cost_flight_leg(0,1,0))

def satisfied_demand(i,j,k,n):
    return min(flow(i,j,math.floor(n / 40)), fleet[1][k])

# Calculate the profit for each flight leg
def profit(i, j, k, n):
    if i != j:
        revenue = 0.26 * distance(i,j) * satisfied_demand(i, j, k, n) / 1000
        costs = cost_flight_leg(i,j,k)
        return revenue - costs
    else:
        return 0
# ---------------------------------------------------------------------------------------------------------------------
# print(flow(0,1,26))
# print(profit(0, 1, 0, 1050))
# print(fleet[1][0])

# Calculate the flight duration for each flight leg
# ONDERSTAANDE TWEE FUNCTIES KUNNEN LATER EVT GEMERGED WORDEN
def get_block_time(i,j,k):
    return distance(i, j, R_E=6371) / fleet[0][k] * 60 + 30

def flight_duration(i,j,k):
    return get_block_time(i,j,k) + fleet[2][k] # Round up to the nearest 6-fold
# print(flight_duration(1,2,1)) # From Paris to Amsterdam

def get_stages_in_flight(i,j,k):
    return math.ceil((flight_duration(i,j,k)) / 6)

# ---------------------------------------------------------------------------------------------------------------------
# print(get_stages_in_flight(1,2,1))


# Create a list of all possible decisions a plane can take at airport i at departure stage n
hub = 3 # Define Frankfurt as a hub

def get_candidates(i,k,n):    
    candidates = []
    if i == hub:                 
        for j in range(len(cities)):                                                        
            if i == j:
                arrival_stage = n+1
                if arrival_stage < 1200:                                                     # make sure no airplane is flying after the fifth day
                    candidates.append([n, arrival_stage, i, j, 0, 0, 0])             # grounded at hub
            else:
                arrival_stage = n+get_stages_in_flight(i,j,k)
                if arrival_stage < 1200:
                    candidates.append([n, arrival_stage, i, j, satisfied_demand(i,j,k,n), get_block_time(i,j,k), profit(i,j,k,n)]) # flight to spoke
        
    else:
        if n+1 < 1200:
            candidates.append([n, n+1 ,i,i, 0, 0, 0])   
        arrival_stage = n+get_stages_in_flight(i,hub,k)                               # grounded at spoke
        if arrival_stage < 1200:
            candidates.append([n, arrival_stage, i, hub, satisfied_demand(i, hub, k, n), get_block_time(i,hub,k), profit(i,hub,k,n)])     # flight to hub
    return candidates

# ---------------------------------------------------------------------------------------------------------------------
# print(get_candidates(hub,0,1100))



memo = {}

def get_routes(departure_airport, k, departure_stage):
    # Check if the result is already cached
    if (departure_airport, k, departure_stage) in memo:
        return memo[(departure_airport, k, departure_stage)]
    
    routes = []
    
    for c in get_candidates(departure_airport, k, departure_stage):
        arrival_stage = c[1]
        arrival_airport = c[3]
        future_routes = get_routes(arrival_airport, k, arrival_stage)
        
        if future_routes:
            for route in future_routes:
                if distance(c[2], c[3]) <= fleet[3][k] and airports[3][c[2]] >= fleet[4][k] and airports[3][c[3]] >= fleet[4][k]:
                    total_profit = c[-1] + route[0]  # Add current profit to future route's profit
                    total_block = c[-2] + route[1]   # Add current block time to future route's block time

                    # Add the total profit to the route and print only the flight actions (not the grounding)
                    if c[2] != c[3]:
                        routes.append([total_profit, total_block, [c[0], c[1], c[2], c[3], c[4]]] + route[2:])
                    else:
                        routes.append([total_profit] + route[1:])
        else:
            total_profit = c[-1]  # Profit for the last segment
            total_block = c[-2]   # Block time for the last segment
            routes.append([total_profit, total_block, [c[0], c[1], c[2], c[3], c[4]]])


# BLOCK TIME CONSTRAINT IS MISSING! Have a total block time of 1800 minutes or more (6 hours a day on average)
    
    # Filter routes that end at the hub 
    valid_routes = [route for route in routes if route[-1][3] == hub]
    # Only keep the route with the highest profit for each departure_airport
    valid_routes = sorted(valid_routes, key=lambda route: route[0], reverse=True)
    
    if valid_routes:
        max_profit_route = valid_routes[0]  # Take the route with the highest profit
        memo[(departure_airport, k, departure_stage)] = [max_profit_route]  # Cache only the highest profit route for the current stage
    else:
        memo[(departure_airport, k, departure_stage)] = []
    
    return memo[(departure_airport, k, departure_stage)]

# Clear the memo dictionary before running to avoid stale data
memo.clear()

# Example call
# print(get_routes(3, 0, 0))
# ---------------------------------------------------------------------------------------------------------------------
# for big_element in get_routes(hub,0,0):
#     for small_element in big_element:
#         print(small_element)


def choose_ac(departure_airport=3, departure_stage=0):
    available_ac = (fleet[9])   # tuple of available aircrafts
    ac_list = [*available_ac]   # list of available aircrafts
    print(ac_list)
    global demand_matrix

    # Keep initiating aircrafts untill all aircrafts are used
    while not all(ac == 0 for ac in ac_list):
        best_profit = 0
        flown_route = []
        ac_best_route = None
        for k in range(len(ac_list)):
            # Check if there are any aircrafts available of type k
            if ac_list[k] >= 1:
                flown_route.append(get_routes(departure_airport, k, departure_stage)[0])
                print(flown_route[k])
                # Check which aircraft type has the highest profit
                if flown_route[k][0] - fleet[5][k]*5 > best_profit:
                    best_profit = flown_route[k][0] - fleet[5][k]*5
                    ac_best_route = k
                # Don't consider aircraft type that can't make a profit in the next loop
                if flown_route[k][0] - fleet[5][k]*5 <= 0:
                    ac_list[k] = 0
        if not all(ac == 0 for ac in ac_list):
            print(ac_best_route, best_profit, flown_route[ac_best_route][1:])
            ac_list[ac_best_route] -= 1
            print(ac_list)
            print(range(len(flown_route[ac_best_route][2:])))

            # Subtract satisfied demand from the demand matrix
            for flight in range(len(flown_route[ac_best_route][2:])):
                # print(demand_matrix[3, 4, int(205/40)])
                flown_demand = [0] * len(flown_route[ac_best_route][2:])
                flown_bin =    [0] * len(flown_route[ac_best_route][2:])
                flown_dep =    [0] * len(flown_route[ac_best_route][2:])
                flown_arr =    [0] * len(flown_route[ac_best_route][2:])
                flown_demand[flight] = flown_route[ac_best_route][2:][flight][-1]
                flown_bin[flight] = int(flown_route[ac_best_route][2:][flight][0] / 40)
                flown_dep[flight] = flown_route[ac_best_route][2:][flight][2]
                flown_arr[flight] = flown_route[ac_best_route][2:][flight][3]
                if flown_demand[flight] < demand_matrix[flown_dep[flight]][flown_arr[flight]][flown_bin[flight]]:
                    demand_matrix[flown_dep[flight]][flown_arr[flight]][flown_bin[flight]] -= flown_demand[flight]
                else:
                    demand_matrix[flown_dep[flight]][flown_arr[flight]][flown_bin[flight]] = 0
                    rest_demand_first = flown_demand[flight] - demand_matrix[flown_dep[flight]][flown_arr[flight]][flown_bin[flight]]
                    if rest_demand_first < demand_matrix[flown_dep[flight]][flown_arr[flight]][flown_bin[flight - 1]]:
                        demand_matrix[flown_dep[flight]][flown_arr[flight]][flown_bin[flight - 1]] -= rest_demand_first
                    else:
                        demand_matrix[flown_dep[flight]][flown_arr[flight]][flown_bin[flight - 1]] = 0
                        rest_demand_second = rest_demand_first - demand_matrix[flown_dep[flight]][flown_arr[flight]][flown_bin[flight - 1]]
                        demand_matrix[flown_dep[flight]][flown_arr[flight]][flown_bin[flight - 2]] -= rest_demand_second

choose_ac()
