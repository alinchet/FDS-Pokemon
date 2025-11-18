import pandas as pd
from tqdm.notebook import tqdm
import numpy as np


def make_pokemons_db(data: list[dict]) -> dict:
    """
    Return a Pokémons database indexed by Pokémon name.
    The db is made out of all the Pokémons seen in all of the battles in the training data.
    Test data has NOT been used.
    
    Parameters:
        data_list (list[dict]): List of battle data dictionaries.
        level_threshold (int): Only include Pokémon at this level (default is 100).

    Returns:
        dict: Dictionary with Pokémon names as keys and their stats/types as values.
    """
    
    BASE_STATS = [
        'base_atk', 'base_def', 'base_hp', 'base_spa', 'base_spd', 'base_spe', 'level', 'types'
    ]

    pokemons_db = {}
    for battle in data:
        for pokemon_in_battle in battle['p1_team_details']:
            pokemon_name = pokemon_in_battle['name']
            if pokemon_name not in pokemons_db:
                pokemons_db[pokemon_name] = {}
                for stat in BASE_STATS:
                    pokemons_db[pokemon_name][stat] = pokemon_in_battle[stat]
        
        pokemon_in_battle = battle['p2_lead_details']
        pokemon_name = pokemon_in_battle['name']
        if pokemon_name not in pokemons_db:
            pokemons_db[pokemon_name] = {}
            for stat in BASE_STATS:
                pokemons_db[pokemon_name][stat] = pokemon_in_battle[stat]
    
    return pokemons_db


def make_pokemons_db_alt(data: list[dict], level_threshold: int = 100):
    """
    Creates a dictionary mapping Pokémon names to their base stats and types,
    using only Pokémon at the specified level threshold from the provided data list.

    Parameters:
        data_list (list[dict]): List of battle data dictionaries.
        level_threshold (int): Only include Pokémon at this level (default is 100).

    Returns:
        dict: Dictionary with Pokémon names as keys and their stats/types as values.
    """
    pokemons_db = {}

    for battle in data:
        # Check P1 team details
        pokemon_list = battle.get('p1_team_details', [])
        for pokemon in pokemon_list:
            name = pokemon.get('name')
            level = pokemon.get('level')
            if name and name not in pokemons_db and level == level_threshold:
                # Extract all base stats
                stats = {k: v for k, v in pokemon.items() if k.startswith('base_')}

                # Add the types
                stats['types'] = pokemon.get('types', ['notype', 'notype'])

                if stats:
                    pokemons_db[name] = stats
    return pokemons_db


def gather_pokemon_stats(battle, pokemons_db):
    """
    Iterates through the timeline of a battle to track the stats of each Pokémon
    (mainly surviving ones).
    """
    # Initialize the data structure
    team1_survivors = {
        pokemon.get('name', f'p1_unknown_{i}'): {
            'hp': 1.00,
            'status': 'nostatus'
        } for i, pokemon in enumerate(battle.get('p1_team_details', []))
    }

    team2_survivors = {}
    
    team2_survivors[battle.get('p2_lead_details', {}).get('name')] = {
        'hp': 1.00,
        'status': 'nostatus'
    }
    
    # Fill the values with the latest conditions shown in the timeline
    for turn in battle.get('battle_timeline', []):
        team1_survivors[turn.get('p1_pokemon_state', {}).get('name')] = {
            'hp': turn.get('p1_pokemon_state', {}).get('hp_pct'),
            'status': turn.get('p1_pokemon_state', {}).get('status')
        }
        team2_survivors[turn.get('p2_pokemon_state', {}).get('name')] = {
            'hp': turn.get('p2_pokemon_state', {}).get('hp_pct'),
            'status': turn.get('p2_pokemon_state', {}).get('status')
        }

    # Compute the number of pokemon changes for each player (indicator of strategy)
    team1_player_swaps = 0
    team2_player_swaps = 0
    for turn in battle.get('battle_timeline', []):
        p1_current_state = turn.get('p1_pokemon_state', {}).get('name')
        p2_current_state = turn.get('p2_pokemon_state', {}).get('name')
        if turn != battle.get('battle_timeline', [])[0]:
            if p1_prev_state != p1_current_state:
                team1_player_swaps += 1
            if p2_prev_state != p2_current_state:
                team2_player_swaps += 1
        else:
            p1_prev_state = p1_current_state
            p2_prev_state = p2_current_state

    # Compute the number of effects working on the last round and weight them
    p1_effects = len(battle.get('battle_timeline', [])[-1].get('p1_pokemon_state', {}).get('effects', [])) * 0.4
    p2_effects = len(battle.get('battle_timeline', [])[-1].get('p2_pokemon_state', {}).get('effects', [])) * 0.4


    #winningmost_pokemons = ['starmie', 'alakazam', 'tauros', 'exeggutor', 'snorlax', 'chansey']
    #loser_pokemons = ['charizard', 'persian', 'articuno', 'dragonite', 'victreebel']
    winningmost_pokemons = ['exeggutor', 'snorlax', 'chansey']
    loser_pokemons = ['charizard']
    # All the pokemons seen in the battle (indexed by their name).
    p1_names_state, p2_names_state = {}, {}
    for turn in battle['battle_timeline']:
        p1_state = turn['p1_pokemon_state']
        p2_state = turn['p2_pokemon_state']
        
        p1_names_state[p1_state['name']] = {
            'hp_pct': p1_state['hp_pct'],
            'status': p1_state['status']
        }
        p2_names_state[p2_state['name']] = {
            'hp_pct': p2_state['hp_pct'],
            'status': p2_state['status']
        }
    
    # Winningmost Pokemons combined stats
    
    # Team 1 survivors
    spd = spe = spa = atk = deff = level = 0
    for n, cond in p1_names_state.items():
        if cond['hp_pct'] > 0 and cond['status'] == 'nostatus':
            spd += pokemons_db[n]['base_spd'] * cond['hp_pct']
            spe += pokemons_db[n]['base_spe'] * cond['hp_pct']
            spa += pokemons_db[n]['base_spa'] * cond['hp_pct']
            atk += pokemons_db[n]['base_atk'] * cond['hp_pct']
            deff += pokemons_db[n]['base_def'] * cond['hp_pct']
            level += pokemons_db[n]['level'] * cond['hp_pct']
    winning_stat_1 = {'c_spd': spd, 'c_spe': spe, 'c_spa': spa, 'c_atk': atk, 'c_def': deff, 'c_level': level}
    
    # Team 2 survivors
    spd = spe = spa = atk = deff = level = 0
    for n, cond in p2_names_state.items():
        if cond['hp_pct'] > 0 and cond['status'] == 'nostatus':
            spd += pokemons_db[n]['base_spd'] * cond['hp_pct']
            spe += pokemons_db[n]['base_spe'] * cond['hp_pct']
            spa += pokemons_db[n]['base_spa'] * cond['hp_pct']
            atk += pokemons_db[n]['base_atk'] * cond['hp_pct']
            deff += pokemons_db[n]['base_def'] * cond['hp_pct']
            level += pokemons_db[n]['level'] * cond['hp_pct']
    winning_stat_2 = {'c_spd': spd, 'c_spe': spe, 'c_spa': spa, 'c_atk': atk, 'c_def': deff, 'c_level': level}


    # Surviving Pokemons after 30 turns with a critical status (psn or brn).
    t1_critical_status_num = sum([p1_names_state[n]['status'] in ['psn', 'brn'] for n in p1_names_state if p1_names_state[n]['hp_pct'] > 0])
    t2_critical_status_num = sum([p2_names_state[n]['status'] in ['psn', 'brn'] for n in p2_names_state if p2_names_state[n]['hp_pct'] > 0])
    
    # Surviving psychic Pokemons per team
    t1_psychic_num = sum(['psychic' in pokemons_db[n]['types'] for n in p1_names_state if p1_names_state[n]['hp_pct'] > 0])
    t2_psychic_num = sum(['psychic' in pokemons_db[n]['types'] for n in p2_names_state if p2_names_state[n]['hp_pct'] > 0])

    # Surviving winningmost Pokemons per team
    t1_winningmost_num = sum([n in winningmost_pokemons for n in p1_names_state if p1_names_state[n]['hp_pct'] > 0])
    t2_winningmost_num = sum([n in winningmost_pokemons for n in p2_names_state if p2_names_state[n]['hp_pct'] > 0])

    # Surviving most loser Pokemons per team
    t1_loser_num = sum([n in loser_pokemons for n in p1_names_state if p1_names_state[n]['hp_pct'] > 0])
    t2_loser_num = sum([n in loser_pokemons for n in p2_names_state if p2_names_state[n]['hp_pct'] > 0])
    
    # Add the slots corresponding to the unseen pokemons of player #2
    for i in range(len(team2_survivors), 6):
        team2_survivors[f'p2_unknown_{i}'] = {
            'hp': 1.00,
            'status': 'nostatus'
        }

    return [
        team1_player_swaps, p1_effects, team1_survivors, team2_player_swaps, p2_effects, team2_survivors, t1_critical_status_num, t2_critical_status_num, t1_psychic_num, t2_psychic_num, 
        t1_winningmost_num, t2_winningmost_num, t1_loser_num, t2_loser_num, winning_stat_1, winning_stat_2
    ]


def compute_base_stats_calculations(p1_pok_cond, p2_pok_cond, pokemon_db):
    t1_total_speed = t2_total_speed = 0
    t1_total_attack = t2_total_attack = 0
    t1_total_defense = t2_total_defense = 0
    t1_total_sp_attack = t2_total_sp_attack = 0
    t1_total_sp_defense = t2_total_sp_defense = 0
    t1_total_hp = t2_total_hp = 0
    
    for pokemon_name in p1_pok_cond.keys():
        if pokemon_name in pokemon_db:
            t1_total_speed += pokemon_db[pokemon_name]['base_spe']
            t1_total_attack += pokemon_db[pokemon_name]['base_atk']
            t1_total_defense += pokemon_db[pokemon_name]['base_def']
            t1_total_sp_attack += pokemon_db[pokemon_name]['base_spa']
            t1_total_sp_defense += pokemon_db[pokemon_name]['base_spd']
            t1_total_hp += pokemon_db[pokemon_name]['base_hp']

    for pokemon_name in p2_pok_cond.keys():
        if pokemon_name in pokemon_db:
            t2_total_speed += pokemon_db[pokemon_name]['base_spe']
            t2_total_attack += pokemon_db[pokemon_name]['base_atk']
            t2_total_defense += pokemon_db[pokemon_name]['base_def']
            t2_total_sp_attack += pokemon_db[pokemon_name]['base_spa']
            t2_total_sp_defense += pokemon_db[pokemon_name]['base_spd']
            t2_total_hp += pokemon_db[pokemon_name]['base_hp']
    
    speed_diff = t1_total_speed - t2_total_speed
    defense_diff = t1_total_defense - t2_total_defense
    attack_diff = t1_total_attack - t2_total_attack
    sp_attack_diff = t1_total_sp_attack - t2_total_sp_attack
    sp_defense_diff = t1_total_sp_defense - t2_total_sp_defense
    hp_diff = t1_total_hp - t2_total_hp

    speed_ratio = t1_total_speed / t2_total_speed
    defense_ratio = t1_total_defense / t2_total_defense
    attack_ratio = t1_total_attack / t2_total_attack
    sp_attack_ratio = t1_total_sp_attack / t2_total_sp_attack
    sp_defense_ratio = t1_total_sp_defense / t2_total_sp_defense
    hp_ratio = t1_total_hp / t2_total_hp

    return [
        speed_diff, defense_diff, attack_diff, sp_attack_diff, sp_defense_diff, hp_diff,
        speed_ratio, defense_ratio, attack_ratio, sp_attack_ratio, sp_defense_ratio, hp_ratio
    ]


def build_features(data: list[dict]) -> pd.DataFrame:
    feature_list = []

    # Creating a dictionary of pokemons along with stats in the dataset
    pokemon_dict = make_pokemons_db_alt(data)

    # our own version of Pokemon db
    pokemons_db = make_pokemons_db(data)

    # For each battle
    for battle in tqdm(data, desc="Extracting features"):
        features = {}
        timeline = battle.get('battle_timeline', [])

        # Track the conditions of teams at the end of the timeline
        # Track the number of changes of each trainer
        # Track the number of effects
        (t1_player_swaps, t1_effects, t1_survivors, t2_player_swaps, t2_effects, t2_survivors,
         t1_critical_status_num, t2_critical_status_num, t1_psychic_num, t2_psychic_num,
         t1_winningmost_num, t2_winningmost_num, t1_loser_num, t2_loser_num, winning_stat_1, winning_stat_2) = \
            gather_pokemon_stats(battle, pokemons_db)

        features |= winning_stat_1
        features |= winning_stat_2
        
        for c_stat in winning_stat_1.keys():
            features[f'{c_stat}_diff'] = winning_stat_1[c_stat] - winning_stat_2[c_stat]
            features[f'{c_stat}_adv'] = int(winning_stat_1[c_stat] > winning_stat_2[c_stat])
            features[f'{c_stat}_ratio'] = winning_stat_1[c_stat] / (winning_stat_2[c_stat] or 1)
        
        
        features['t1_winningmost_num'] = t1_winningmost_num
        features['t2_winningmost_num'] = t2_winningmost_num
        #features['winningmost_ratio'] = t1_winningmost_num / (t2_winningmost_num or 1)
        #features['t1_loser_num'] = t1_loser_num
        #features['t2_loser_num'] = t2_loser_num
        features['t1_winning_loser_ratio'] = (t1_winningmost_num or 1) / (t2_loser_num or 1)
        features['t2_winning_loser_ratio'] = (t2_winningmost_num or 1) / (t1_loser_num or 1)

        features['t1_critical_status_num'] = t1_critical_status_num
        features['t2_critical_status_num'] = t2_critical_status_num
        features['critical_status_num_ratio'] = (t1_critical_status_num or 1) / (t2_critical_status_num or 1)
        
        features['t1_psychic_num'] = t1_psychic_num
        features['t2_psychic_num'] = t2_psychic_num
        features['psychic_num_ratio'] = (t1_psychic_num or 1) / (t2_psychic_num or 1)

        # Add to the features the mean of the percentage of HP for each team
        team1_mean_hp_pct = np.mean([info['hp'] for info in t1_survivors.values()])
        team2_mean_hp_pct = np.mean([info['hp'] for info in t2_survivors.values()])
        features['team1_mean_pc_hp'] = team1_mean_hp_pct
        features['team2_mean_pc_hp'] = team2_mean_hp_pct

        # Add to the features the number of surviving pokemon for each team
        team1_surviving_pokemon = sum(1 for info in t1_survivors.values() if info["hp"] > 0)
        team2_surviving_pokemon = sum(1 for info in t2_survivors.values() if info["hp"] > 0)
        features['team1_surviving_pokemon'] = team1_surviving_pokemon
        features['team2_surviving_pokemon'] = team2_surviving_pokemon

        # Add to the features the number of pokemon affected by status and an effect index for each team
        team1_status_score = sum(1 for i in t1_survivors.values() if i['hp'] > 0 and i['status'] != 'nostatus') + t1_effects
        team2_status_score = sum(1 for i in t2_survivors.values() if i['hp'] > 0 and i['status'] != 'nostatus') + t2_effects
        features['team1_status_score'] = team1_status_score
        features['team2_status_score'] = team2_status_score

        # Add to the features not the mean but simply the difference
        # Also some of them could not be chosen because there is redundance in the pattern of stats distribution
        speed_diff, defense_diff, attack_diff, sp_attack_diff, sp_defense_diff, hp_diff, speed_ratio, defense_ratio, attack_ratio, sp_attack_ratio, sp_defense_ratio, hp_ratio = \
            compute_base_stats_calculations(t1_survivors, t2_survivors, pokemon_dict)
                
        features['total_speed_diff'] = speed_diff
        features['total_attack_diff'] = attack_diff
        features['total_defense_diff'] = defense_diff
        features['total_sp_attack_diff'] = sp_attack_diff
        features['total_sp_defense_diff'] = sp_defense_diff
        features['total_hp_diff'] = hp_diff
        
        features['total_speed_ratio'] = speed_ratio
        features['total_attack_ratio'] = attack_ratio
        features['total_defense_ratio'] = defense_ratio
        features['total_sp_attack_ratio'] = sp_attack_ratio
        features['total_sp_defense_ratio'] = sp_defense_ratio
        features['total_hp_ratio'] = hp_ratio

        # Add to the features the number of pokemon changes along the timeline for each team (indicator of strategy)
        features['team1_player_swaps'] = t1_player_swaps
        features['team2_player_swaps'] = t2_player_swaps

        # Add to the features the battle id and the true outcome of the battle
        features['battle_id'] = battle.get('battle_id')
        # Include target variable if in data
        if 'player_won' in battle:
            features['player_won'] = int(battle['player_won'])

        # Append all features to the list
        feature_list.append(features)

    # Convert to DataFrame and handle missing values.
    return pd.DataFrame(feature_list).fillna(0)
