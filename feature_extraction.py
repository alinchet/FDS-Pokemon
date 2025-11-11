# Different extract* functions: they differentiate on the number of features they extract.
# Only one of them must be called (otherwise the last called one overwrites the features dataframe).
import pandas as pd
import numpy as np

POKEMON_STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']
BOOST_STATS = ['atk', 'def', 'spa', 'spd', 'spe']

LOW_hp_pct_THRESHOLD = 0.3

def extract_full_features(data: list[dict], progress_bar=None) -> pd.DataFrame:
    """
    This is the most complete features extractor.
    """
    feature_list = []
    data_lines = tqdm(data, desc="Extracting features") if progress_bar else data
    
    for battle in data_lines:
        battle_features = {}

        # We also need the ID and the target variable (if they exist)
        battle_features['battle_id'] = battle.get('battle_id')
        battle_features['player_won'] = int(battle['player_won'])
        
        # --- Player 1 Team Features ---
        #p1_team = battle.get('p1_team_details', [])
        p1_team = battle['p1_team_details']
        for stat in POKEMON_STATS:
            battle_features[f'p1_mean_{stat}'] = np.mean([p[f'base_{stat}'] for p in p1_team])
            #battle_features[f'p1_min_{stat}'] = np.min([p.get(f'base_{stat}', 0) for p in p1_team])
            #battle_features[f'p1_max_{stat}'] = np.max([p.get(f'base_{stat}', 0) for p in p1_team])
            #battle_features[f'p1_std_{stat}'] = np.std([p.get(f'base_{stat}', 0) for p in p1_team])
        
        i = 1
        for p in p1_team:
            battle_features[f'p1_name_{i}'] = p['name']
            battle_features[f'p1_level_{i}'] = p['level']
            battle_features[f'p1_types_1_{i}'] = p['types'][0]
            battle_features[f'p1_types_2_{i}'] = p['types'][1]
            i += 1            
        
        battle_features['p1_mean_level'] = np.mean([p.get('level', 0) for p in p1_team])
        
        '''
        for stat in BOOST_STATS:
            battle_features[f'p1_boosts_{stat}'] = np.mean([move['p1_pokemon_state']['boosts'][stat] for move in battle['battle_timeline']])
            battle_features[f'p1_boosts_{stat}'] = np.mean([move['p2_pokemon_state']['boosts'][stat] for move in battle['battle_timeline']])
        '''
        
        # --- Player 2 Lead battle_features ---
        p2_lead = battle['p2_lead_details']
        #if p2_lead:
        # Player 2's lead Pokémon's stats
        for stat in POKEMON_STATS:
            battle_features[f'p2_lead_{stat}'] = p2_lead[f'base_{stat}']
            
        battle_features['p2_lead_name'] = p2_lead['name']
        battle_features['p2_lead_level'] = p2_lead['level']
        battle_features['p2_lead_types_1'] = p2_lead['types'][0]
        battle_features['p2_lead_types_2'] = p2_lead['types'][1]

        for stat in ['spd']:
            base_stat = f'base_{stat}'
            battle_features[f'1st_player_{stat}_advantage'] = battle['p1_team_details'][0][base_stat] / (p2_lead[base_stat] or 1)
        

        # Number of Moves
        # Always 30: no information!
        #battle_features['moves'] = len(battle['battle_timeline'])

        battle_features['p1_mean_hp_pct'] = np.mean([move['p1_pokemon_state']['hp_pct'] for move in battle['battle_timeline']])
        battle_features['p2_mean_hp_pct'] = np.mean([move['p2_pokemon_state']['hp_pct'] for move in battle['battle_timeline']])
  
        battle_features['hp_last_advantage'] = \
            battle['battle_timeline'][-1]['p1_pokemon_state']['hp_pct'] >= \
            battle['battle_timeline'][-1]['p2_pokemon_state']['hp_pct']
    
        battle_features['p1_hp_pct_shortage'] = sum(
            [move['p1_pokemon_state']['hp_pct'] < LOW_hp_pct_THRESHOLD for move in battle['battle_timeline']]
        )
        battle_features['p2_hp_pct_shortage'] = sum(
            [move['p2_pokemon_state']['hp_pct'] < LOW_hp_pct_THRESHOLD for move in battle['battle_timeline']]
        )

        # Very Good!
        battle_features['p1_hp_pct_zero'] = sum(
            [move['p1_pokemon_state']['hp_pct'] == 0 for move in battle['battle_timeline']]
        )
        battle_features['p2_hp_pct_zero'] = sum(
            [move['p2_pokemon_state']['hp_pct'] == 0 for move in battle['battle_timeline']]
        )
        #battle_features['p1_hp_pct_zero_advantage'] = battle_features['p1_hp_pct_zero'] < battle_features['p2_hp_pct_zero']
        battle_features['p1_hp_pct_zero_diff'] = battle_features['p1_hp_pct_zero'] - battle_features['p2_hp_pct_zero']
        
        battle_features['p1_mean_accuracy'] = \
            np.mean([move['p1_move_details']['accuracy'] for move in battle['battle_timeline'] if move['p1_move_details']])
        battle_features['p2_mean_accuracy'] = \
            np.mean([move['p2_move_details']['accuracy'] for move in battle['battle_timeline'] if move['p2_move_details']])

        # This feature doesn't seem to give any advantage.
        '''        
        battle_features['p1_mean_base_power'] = \
            np.mean([move['p1_move_details']['base_power'] for move in battle['battle_timeline'] if move['p1_move_details']])
        battle_features['p2_mean_base_power'] = \
            np.mean([move['p2_move_details']['base_power'] for move in battle['battle_timeline'] if move['p2_move_details']])
        '''

        battle_features['power_advantage'] = \
            sum([move['p1_move_details']['base_power'] <= move['p2_move_details']['base_power'] for move in battle['battle_timeline'] \
                 if move['p1_move_details'] and move['p2_move_details']])
        
        '''        
        battle_features['accuracy_advantage'] = \
            sum([move['p1_move_details']['accuracy'] > move['p2_move_details']['accuracy'] for move in battle['battle_timeline'] \
                 if move['p1_move_details'] and move['p2_move_details']])
        '''
        
        battle_features['p1_mean_priority'] = \
            np.mean([move['p1_move_details']['priority'] for move in battle['battle_timeline'] if move['p1_move_details']])
        battle_features['p2_mean_priority'] = \
            np.mean([move['p2_move_details']['priority'] for move in battle['battle_timeline'] if move['p2_move_details']])

        battle_features['priority_advantage'] = \
            np.mean([move['p1_move_details']['priority'] - move['p2_move_details']['priority'] for move in battle['battle_timeline'] \
                 if move['p1_move_details'] and move['p2_move_details']])

        battle_features['hp_pct_advantage'] = \
            sum([move['p1_pokemon_state']['hp_pct'] > move['p2_pokemon_state']['hp_pct'] for move in battle['battle_timeline'] \
                 if move['p1_move_details'] and move['p2_move_details']])

        for move in battle['battle_timeline']:
            if move['p1_move_details'] and move['p2_move_details']:
                p1_name = move['p1_pokemon_state']['name']
                p2_name = move['p2_pokemon_state']['name']
                names_comb = f'{p1_name}_{p2_name}'
                battle_features[names_comb] = battle_features.get(names_comb, 0) + 1

                # Move types, counted by pair
                '''
                p1_move_type = move['p1_move_details']['type']
                p2_move_type = move['p2_move_details']['type']
                move_type_comb = f'{p1_move_type}_{p2_move_type}'
                battle_features[move_type_comb] = battle_features.get(move_type_comb, 0) + 1
                '''
                
                # How many times (moves) a Pokemon appears in battle
                battle_features[p1_name] = battle_features.get(p1_name, 0) + 1 
                battle_features[p2_name] = battle_features.get(p2_name, 0) + 1

        
        # STATUSES
        player_status_count = {}
        players_cross_status_count = {}
        for move in battle['battle_timeline']:
            status_p1 = f'{move['p1_pokemon_state']['status']}'
            status_p2 = f'{move['p2_pokemon_state']['status']}'
            
            if move['p1_move_details'] and move['p2_move_details']:
                # Count how many times each status appears in each team in a battle
                player_status_count[f'p1_{status_p1}'] = player_status_count.get(f'p1_{status_p1}', 0) + 1
                player_status_count[f'p2_{status_p2}'] = player_status_count.get(f'p2_{status_p2}', 0) + 1

                players_cross_status_count[f'{status_p1}_{status_p2}'] = players_cross_status_count.get(f'{status_p1}_{status_p2}', 0) + 1
        
        #battle_features['nostatus_advantage'] = player_status_count.get('nostatus_p0', 0) > player_status_count.get('nostatus_p1', 0)
        for status in ['nostatus', 'frz', 'slp', 'fnt']:
            #battle_features[f'{status}_advantage'] = player_status_count.get(f'{status}_p0', 0) > player_status_count.get(f'{status}_p1', 0)
            battle_features[f'{status}_diff'] = player_status_count.get(f'p1_{status}', 0) - player_status_count.get(f'p2_{status}', 0)

            battle_features[f'nostatus_{status}'] = players_cross_status_count.get(f'nostatus_{status}', 0)
            battle_features[f'nostatus_{status}_diff'] = \
                players_cross_status_count.get(f'nostatus_{status}', 0) - players_cross_status_count.get(f'{status}_nostatus', 0)
        
        '''
        for move in battle['battle_timeline']:
            if move['p1_move_details'] and move['p2_move_details']:
            # Last hp_pct per pokemon per battle
                den_p1 = move['p1_move_details']['base_power'] if move['p1_move_details']['base_power'] else 1
                den_p2 = move['p2_move_details']['base_power'] if move['p2_move_details']['base_power'] else 1
                battle_features['t1_power_advantage'] = battle_features.get('t1_power_advantage', 0) + \
                    move['p1_move_details']['base_power'] / den_p2
                battle_features['t2_power_advantage'] = battle_features.get('t2_power_advantage', 0) + \
                    move['p2_move_details']['base_power'] / den_p1
        '''
        
        for move in battle['battle_timeline']:
            # Last hp_pct per pokemon per battle
            battle_features[f'p1_{move['p1_pokemon_state']['name']}_hp_pct'] = move['p1_pokemon_state']['hp_pct']
            battle_features[f'p2_{move['p2_pokemon_state']['name']}_hp_pct'] = move['p2_pokemon_state']['hp_pct']
            
        p1_residual_hp_pct = p2_residual_hp_pct = 0
        p1_players_with_low_hp_pct = p2_players_with_low_hp_pct = 0
        for key, value in battle_features.items():
            if key.endswith('_hp_pct'):
                if key.startswith('p1_'):
                    p1_residual_hp_pct += value
                    if value < LOW_hp_pct_THRESHOLD:
                        p1_players_with_low_hp_pct += 1
                elif key.startswith('p2_'):
                    p2_residual_hp_pct += value
                    if value < LOW_hp_pct_THRESHOLD:
                        p2_players_with_low_hp_pct += 1

        battle_features['p1_residual_hp_pct'] = p1_residual_hp_pct
        battle_features['p2_residual_hp_pct'] = p2_residual_hp_pct
        battle_features['p1_players_with_low_hp_pct'] = p1_players_with_low_hp_pct
        battle_features['p2_players_with_low_hp_pct'] = p2_players_with_low_hp_pct        
        battle_features['residual_hp_pct_advantage'] = p1_residual_hp_pct - p2_residual_hp_pct
            
        feature_list.append(battle_features)
        
    return pd.DataFrame(feature_list).fillna(0)


# Alternative extract_features()
def extract_features_only_from_battles(data: list[dict], progress_bar=None) -> pd.DataFrame:
    """
    This function extracts features only from the battle moves
    """
    feature_list = []
    data_lines = tqdm(data, desc="Extracting features") if progress_bar else data
    
    for battle in data_lines:
        battle_features, battle_features_tmp = {}, {}

        # We also need the ID and the target variable (if they exist)
        battle_features['battle_id'] = battle.get('battle_id')
        battle_features['player_won'] = int(battle['player_won'])

        battle_features['p1_mean_hp_pct'] = np.mean([move['p1_pokemon_state']['hp_pct'] for move in battle['battle_timeline']])
        battle_features['p2_mean_hp_pct'] = np.mean([move['p2_pokemon_state']['hp_pct'] for move in battle['battle_timeline']])
  
        battle_features['hp_last_advantage'] = \
            battle['battle_timeline'][-1]['p1_pokemon_state']['hp_pct'] >= \
            battle['battle_timeline'][-1]['p2_pokemon_state']['hp_pct']
    
        battle_features['p1_hp_pct_shortage'] = sum(
            [move['p1_pokemon_state']['hp_pct'] < LOW_hp_pct_THRESHOLD for move in battle['battle_timeline']]
        )
        battle_features['p2_hp_pct_shortage'] = sum(
            [move['p2_pokemon_state']['hp_pct'] < LOW_hp_pct_THRESHOLD for move in battle['battle_timeline']]
        )

        # Very Good!
        battle_features['p1_hp_pct_zero'] = sum(
            [move['p1_pokemon_state']['hp_pct'] == 0 for move in battle['battle_timeline']]
        )
        battle_features['p2_hp_pct_zero'] = sum(
            [move['p2_pokemon_state']['hp_pct'] == 0 for move in battle['battle_timeline']]
        )
        #battle_features['p1_hp_pct_zero_advantage'] = battle_features['p1_hp_pct_zero'] < battle_features['p2_hp_pct_zero']
        battle_features['p1_hp_pct_zero_diff'] = battle_features['p1_hp_pct_zero'] - battle_features['p2_hp_pct_zero']
        
        battle_features['p1_mean_accuracy'] = \
            np.mean([move['p1_move_details']['accuracy'] for move in battle['battle_timeline'] if move['p1_move_details']])
        battle_features['p2_mean_accuracy'] = \
            np.mean([move['p2_move_details']['accuracy'] for move in battle['battle_timeline'] if move['p2_move_details']])

        # This feature doesn't seem to give any advantage.
        '''        
        battle_features['p1_mean_base_power'] = \
            np.mean([move['p1_move_details']['base_power'] for move in battle['battle_timeline'] if move['p1_move_details']])
        battle_features['p2_mean_base_power'] = \
            np.mean([move['p2_move_details']['base_power'] for move in battle['battle_timeline'] if move['p2_move_details']])
        '''

        battle_features['power_advantage'] = \
            sum([move['p1_move_details']['base_power'] <= move['p2_move_details']['base_power'] for move in battle['battle_timeline'] \
                 if move['p1_move_details'] and move['p2_move_details']])
        
        '''        
        battle_features['accuracy_advantage'] = \
            sum([move['p1_move_details']['accuracy'] > move['p2_move_details']['accuracy'] for move in battle['battle_timeline'] \
                 if move['p1_move_details'] and move['p2_move_details']])
        '''
        
        battle_features['p1_mean_priority'] = \
            np.mean([move['p1_move_details']['priority'] for move in battle['battle_timeline'] if move['p1_move_details']])
        battle_features['p2_mean_priority'] = \
            np.mean([move['p2_move_details']['priority'] for move in battle['battle_timeline'] if move['p2_move_details']])

        battle_features['priority_advantage'] = \
            np.mean([move['p1_move_details']['priority'] - move['p2_move_details']['priority'] for move in battle['battle_timeline'] \
                 if move['p1_move_details'] and move['p2_move_details']])


        battle_features['hp_pct_advantage'] = \
            sum([move['p1_pokemon_state']['hp_pct'] > move['p2_pokemon_state']['hp_pct'] for move in battle['battle_timeline'] \
                 if move['p1_move_details'] and move['p2_move_details']])


        for move in battle['battle_timeline']:
            if move['p1_move_details'] and move['p2_move_details']:
                p1_name = move['p1_pokemon_state']['name']
                p2_name = move['p2_pokemon_state']['name']
                names_comb = f'{p1_name}_{p2_name}'
                battle_features[names_comb] = battle_features.get(names_comb, 0) + 1

                # Move types, counted by pair
                '''
                p1_move_type = move['p1_move_details']['type']
                p2_move_type = move['p2_move_details']['type']
                move_type_comb = f'{p1_move_type}_{p2_move_type}'
                battle_features[move_type_comb] = battle_features.get(move_type_comb, 0) + 1
                '''
                
                # How many times (moves) a Pokemon appears in battle
                battle_features[p1_name] = battle_features.get(p1_name, 0) + 1 
                battle_features[p2_name] = battle_features.get(p2_name, 0) + 1

        
        # STATUSES
        player_status_count = {}
        players_cross_status_count = {}
        for move in battle['battle_timeline']:
            status_p1 = f'{move['p1_pokemon_state']['status']}'
            status_p2 = f'{move['p2_pokemon_state']['status']}'
            
            if move['p1_move_details'] and move['p2_move_details']:
                # Count how many times each status appears in each team in a battle
                player_status_count[f'p1_{status_p1}'] = player_status_count.get(f'p1_{status_p1}', 0) + 1
                player_status_count[f'p2_{status_p2}'] = player_status_count.get(f'p2_{status_p2}', 0) + 1
    
                players_cross_status_count[f'{status_p1}_{status_p2}'] = players_cross_status_count.get(f'{status_p1}_{status_p2}', 0) + 1
        
        #battle_features['nostatus_advantage'] = player_status_count.get('nostatus_p0', 0) > player_status_count.get('nostatus_p1', 0)
        for status in ['nostatus', 'frz', 'slp', 'fnt']:
            #battle_features[f'{status}_advantage'] = player_status_count.get(f'{status}_p0', 0) > player_status_count.get(f'{status}_p1', 0)
            battle_features[f'{status}_diff'] = player_status_count.get(f'p1_{status}', 0) - player_status_count.get(f'p2_{status}', 0)

            battle_features[f'nostatus_{status}'] = players_cross_status_count.get(f'nostatus_{status}', 0)
            battle_features[f'nostatus_{status}_diff'] = \
                players_cross_status_count.get(f'nostatus_{status}', 0) - players_cross_status_count.get(f'{status}_nostatus', 0)
        
        '''
        for move in battle['battle_timeline']:
            if move['p1_move_details'] and move['p2_move_details']:
            # Last hp_pct per pokemon per battle
                den_p1 = move['p1_move_details']['base_power'] if move['p1_move_details']['base_power'] else 1
                den_p2 = move['p2_move_details']['base_power'] if move['p2_move_details']['base_power'] else 1
                battle_features['t1_power_advantage'] = battle_features.get('t1_power_advantage', 0) + \
                    move['p1_move_details']['base_power'] / den_p2
                battle_features['t2_power_advantage'] = battle_features.get('t2_power_advantage', 0) + \
                    move['p2_move_details']['base_power'] / den_p1
        '''
        
        for move in battle['battle_timeline']:
            # Last hp_pct per pokemon per battle
            battle_features[f'p1_{move['p1_pokemon_state']['name']}_hp_pct'] = move['p1_pokemon_state']['hp_pct']
            battle_features[f'p2_{move['p2_pokemon_state']['name']}_hp_pct'] = move['p2_pokemon_state']['hp_pct']
            
        p1_residual_hp_pct = p2_residual_hp_pct = 0
        p1_players_with_low_hp_pct = p2_players_with_low_hp_pct = 0
        for key, value in battle_features.items():
            if key.endswith('_hp_pct'):
                if key.startswith('p1_'):
                    p1_residual_hp_pct += value
                    if value < LOW_hp_pct_THRESHOLD:
                        p1_players_with_low_hp_pct += 1
                elif key.startswith('p2_'):
                    p2_residual_hp_pct += value
                    if value < LOW_hp_pct_THRESHOLD:
                        p2_players_with_low_hp_pct += 1

        battle_features['p1_residual_hp_pct'] = p1_residual_hp_pct
        battle_features['p2_residual_hp_pct'] = p2_residual_hp_pct
        battle_features['p1_players_with_low_hp_pct'] = p1_players_with_low_hp_pct
        battle_features['p2_players_with_low_hp_pct'] = p2_players_with_low_hp_pct        
        battle_features['residual_hp_pct_advantage'] = int(p1_residual_hp_pct > p2_residual_hp_pct)
            
        feature_list.append(battle_features)
        
    return pd.DataFrame(feature_list).fillna(0)


# Minimal extract_features()
def extract_features_minimal(data: list[dict], progress_bar=None) -> pd.DataFrame:
    """
    This function extracts only the most basic features only from the battle moves
    """
    feature_list = []
    data_lines = tqdm(data, desc="Extracting features") if progress_bar else data
    
    for battle in data_lines:
        battle_features, battle_features_tmp = {}, {}

        # We also need the ID and the target variable (if they exist)
        battle_features['battle_id'] = battle.get('battle_id')
        if 'player_won' in battle:
            battle_features['player_won'] = int(battle['player_won'])

        battle_features['p1_mean_hp_pct'] = np.mean([move['p1_pokemon_state']['hp_pct'] for move in battle['battle_timeline']])
        battle_features['p2_mean_hp_pct'] = np.mean([move['p2_pokemon_state']['hp_pct'] for move in battle['battle_timeline']])
      
        battle_features['p1_hp_pct_shortage'] = sum(
            [move['p1_pokemon_state']['hp_pct'] < LOW_hp_pct_THRESHOLD for move in battle['battle_timeline']]
        )
        battle_features['p2_hp_pct_shortage'] = sum(
            [move['p2_pokemon_state']['hp_pct'] < LOW_hp_pct_THRESHOLD for move in battle['battle_timeline']]
        )

               
        # Very Good!
        battle_features['p1_hp_pct_zero'] = sum(
            [move['p1_pokemon_state']['hp_pct'] == 0 for move in battle['battle_timeline']]
        )
        battle_features['p2_hp_pct_zero'] = sum(
            [move['p2_pokemon_state']['hp_pct'] == 0 for move in battle['battle_timeline']]
        )
        battle_features['p1_hp_pct_zero_advantage'] = battle_features['p1_hp_pct_zero'] < battle_features['p2_hp_pct_zero']
        #battle_features['p1_hp_pct_zero_diff'] = battle_features['p1_hp_pct_zero'] - battle_features['p2_hp_pct_zero']

        battle_features['hp_pct_advantage'] = \
            sum([move['p1_pokemon_state']['hp_pct'] > move['p2_pokemon_state']['hp_pct'] for move in battle['battle_timeline'] \
                 if move['p1_move_details'] and move['p2_move_details']])

        battle_features['base_power_advantage'] = \
            sum([move['p1_move_details']['base_power'] > move['p2_move_details']['base_power'] for move in battle['battle_timeline'] \
                 if move['p1_move_details'] and move['p2_move_details']])

        
        # Comparison between mean of the p1 team stats and p2 leader stats
        for c in ['base_atk', 'base_def', 'base_hp', 'base_spa', 'base_spd', 'base_spe', 'level']:
            p1_mean_stat = np.mean([p[c] for p in battle['p1_team_details']])
            p2_lead_stat = battle['p2_lead_details'][c]
            #battle_features[f'p1_{c}_advantage'] = int(p1_mean_stat > p2_lead_stat)
            battle_features[f'p1_{c}_diff'] = p1_mean_stat - p2_lead_stat

        
        # STATUSES
        player_status_count = {}
        players_cross_status_count = {}
        for move in battle['battle_timeline']:
            status_p1 = f'{move['p1_pokemon_state']['status']}'
            status_p2 = f'{move['p2_pokemon_state']['status']}'
            
            if move['p1_move_details'] and move['p2_move_details']:
                # Count how many times each status appears in each team in a battle
                player_status_count[f'p1_{status_p1}'] = player_status_count.get(f'p1_{status_p1}', 0) + 1
                player_status_count[f'p2_{status_p2}'] = player_status_count.get(f'p2_{status_p2}', 0) + 1
    
                players_cross_status_count[f'{status_p1}_{status_p2}'] = players_cross_status_count.get(f'{status_p1}_{status_p2}', 0) + 1
        
        #battle_features['nostatus_advantage'] = player_status_count.get('nostatus_p0', 0) > player_status_count.get('nostatus_p1', 0)
        for status in ['nostatus', 'frz', 'par', 'slp', 'fnt']: # 'psn' status seems to not give any advantage
            #battle_features[f'{status}_advantage'] = player_status_count.get(f'{status}_p0', 0) > player_status_count.get(f'{status}_p1', 0)
            battle_features[f'{status}_diff'] = player_status_count.get(f'p1_{status}', 0) - player_status_count.get(f'p2_{status}', 0)

            battle_features[f'nostatus_{status}'] = players_cross_status_count.get(f'nostatus_{status}', 0)
            battle_features[f'nostatus_{status}_diff'] = \
                players_cross_status_count.get(f'nostatus_{status}', 0) - players_cross_status_count.get(f'{status}_nostatus', 0)
                
        for move in battle['battle_timeline']:
            # Last hp_pct per pokemon per battle
            battle_features[f'p1_{move['p1_pokemon_state']['name']}_hp_pct'] = move['p1_pokemon_state']['hp_pct']
            battle_features[f'p2_{move['p2_pokemon_state']['name']}_hp_pct'] = move['p2_pokemon_state']['hp_pct']
            
        '''
        p1_residual_hp_pct = p2_residual_hp_pct = 0
        p1_players_with_low_hp_pct = p2_players_with_low_hp_pct = 0
        for key, value in battle_features.items():
            if key.endswith('_hp_pct'):
                if key.startswith('p1_'):
                    p1_residual_hp_pct += value
                    if value < LOW_hp_pct_THRESHOLD:
                        p1_players_with_low_hp_pct += 1
                elif key.startswith('p2_'):
                    p2_residual_hp_pct += value
                    if value < LOW_hp_pct_THRESHOLD:
                        p2_players_with_low_hp_pct += 1

        battle_features['p1_residual_hp_pct'] = p1_residual_hp_pct
        battle_features['p2_residual_hp_pct'] = p2_residual_hp_pct
        battle_features['p1_players_with_low_hp_pct'] = p1_players_with_low_hp_pct
        battle_features['p2_players_with_low_hp_pct'] = p2_players_with_low_hp_pct        
        battle_features['residual_hp_pct_adv'] = int(p1_residual_hp_pct > p2_residual_hp_pct)
        battle_features['residual_hp_pct_diff'] = p1_residual_hp_pct - p2_residual_hp_pct
        battle_features['residual_hp_pct_ratio'] = p1_residual_hp_pct / (p2_residual_hp_pct or 1)
        '''

        # SUM of the hp_pct of the Pokemons for each team after 30 moves and
        # COUNT of the surviving pokemons for each team
        p1_names_state, p2_names_state = {}, {}
        for move in battle['battle_timeline']:
            p1_state = move['p1_pokemon_state']
            p2_state = move['p2_pokemon_state']
            p1_names_state[p1_state['name']] = {}
            p2_names_state[p2_state['name']] = {}
            p1_names_state[p1_state['name']]['hp_pct'] = p1_state['hp_pct']
            p2_names_state[p2_state['name']]['hp_pct'] = p2_state['hp_pct']    

        t1_hp_pct_sum = sum([p1_names_state[n]['hp_pct'] for n in p1_names_state.keys()])
        t2_hp_pct_sum = sum([p2_names_state[n]['hp_pct'] for n in p2_names_state.keys()])

        # Surviving Pokemons for each team
        t1_alive_pokemons = sum([p1_names_state[n]['hp_pct'] > 0 for n in p1_names_state.keys()])
        t2_alive_pokemons = sum([p2_names_state[n]['hp_pct'] > 0 for n in p2_names_state.keys()])

        # Pokemons with low hp_pct for each team
        t1_low_pokemons = sum([p1_names_state[n]['hp_pct'] > LOW_hp_pct_THRESHOLD for n in p1_names_state.keys()])
        t2_low_pokemons = sum([p2_names_state[n]['hp_pct'] > LOW_hp_pct_THRESHOLD for n in p2_names_state.keys()])

        battle_features['hp_pct_sum_teams_adv'] = t2_hp_pct_sum > t1_hp_pct_sum
        battle_features['hp_pct_sum_teams_diff'] = t2_hp_pct_sum - t1_hp_pct_sum
        battle_features['hp_pct_sum_teams_ratio'] = t2_hp_pct_sum / (t1_hp_pct_sum or 1)
        
        battle_features['surviving_pokemons_adv'] = t2_alive_pokemons > t1_alive_pokemons
        battle_features['surviving_pokemons_diff'] = t2_alive_pokemons - t1_alive_pokemons
        battle_features['surviving_pokemons_ratio'] = t2_alive_pokemons / (t1_alive_pokemons or 1)

        battle_features['low_pokemons_adv'] = t2_low_pokemons > t1_low_pokemons
        battle_features['low_pokemons_diff'] = t2_low_pokemons - t1_low_pokemons
        battle_features['low_pokemons_ratio'] = t2_low_pokemons / (t1_low_pokemons or 1)

            
        feature_list.append(battle_features)
        
    return pd.DataFrame(feature_list).fillna(0)



# --- Sealed functions ---
# Functions that proved to give a certain accuracy, sealed so that
# they don't get lost because of changes.

# 82+% accuracy with HistGradientBoostingClassifier
def extract_full_features_SEALED_82(data: list[dict], progress_bar=None) -> pd.DataFrame:
    """
    This is the most complete features extractor.
    """
    feature_list = []
    data_lines = tqdm(data, desc="Extracting features") if progress_bar else data
    
    for battle in data_lines:
        battle_features = {}
        
        # --- Player 1 Team Features ---
        #p1_team = battle.get('p1_team_details', [])
        p1_team = battle['p1_team_details']
        #if p1_team:
        for stat in POKEMON_STATS:
            battle_features[f'p1_mean_{stat}'] = np.mean([p[f'base_{stat}'] for p in p1_team])
            #battle_features[f'p1_min_{stat}'] = np.min([p.get(f'base_{stat}', 0) for p in p1_team])
            #battle_features[f'p1_max_{stat}'] = np.max([p.get(f'base_{stat}', 0) for p in p1_team])
            #battle_features[f'p1_std_{stat}'] = np.std([p.get(f'base_{stat}', 0) for p in p1_team])
        
        i = 1
        for p in p1_team:
            battle_features[f'p1_name_{i}'] = p['name']
            battle_features[f'p1_level_{i}'] = p['level']
            battle_features[f'p1_types_1_{i}'] = p['types'][0]
            battle_features[f'p1_types_2_{i}'] = p['types'][1]
            i += 1            
        
        battle_features['p1_mean_level'] = np.mean([p.get('level', 0) for p in p1_team])
        
        '''
        for stat in BOOST_STATS:
            battle_features[f'p1_boosts_{stat}'] = np.mean([move['p1_pokemon_state']['boosts'][stat] for move in battle['battle_timeline']])
            battle_features[f'p1_boosts_{stat}'] = np.mean([move['p2_pokemon_state']['boosts'][stat] for move in battle['battle_timeline']])
        '''
        
        # --- Player 2 Lead battle_features ---
        p2_lead = battle['p2_lead_details']
        #if p2_lead:
        # Player 2's lead Pokémon's stats
        for stat in POKEMON_STATS:
            battle_features[f'p2_lead_{stat}'] = p2_lead[f'base_{stat}']
            
        battle_features['p2_lead_name'] = p2_lead['name']
        battle_features['p2_lead_level'] = p2_lead['level']
        battle_features['p2_lead_types_1'] = p2_lead['types'][0]
        battle_features['p2_lead_types_2'] = p2_lead['types'][1]

        for stat in ['spd']:
            base_stat = f'base_{stat}'
            battle_features[f'1st_player_{stat}_advantage'] = battle['p1_team_details'][0][base_stat] / p2_lead[base_stat] if p2_lead[base_stat] else 1
        
        # We also need the ID and the target variable (if they exist)
        battle_features['battle_id'] = battle.get('battle_id')
        battle_features['player_won'] = int(battle['player_won'])

        # Number of Moves
        # Always 30: no information!
        #battle_features['moves'] = len(battle['battle_timeline'])

        battle_features['p1_mean_hp_pct'] = np.mean([move['p1_pokemon_state']['hp_pct'] for move in battle['battle_timeline']])
        battle_features['p2_mean_hp_pct'] = np.mean([move['p2_pokemon_state']['hp_pct'] for move in battle['battle_timeline']])
  
        battle_features['hp_last_advantage'] = \
            battle['battle_timeline'][-1]['p1_pokemon_state']['hp_pct'] >= \
            battle['battle_timeline'][-1]['p2_pokemon_state']['hp_pct']
    
        battle_features['p1_hp_pct_shortage'] = sum(
            [move['p1_pokemon_state']['hp_pct'] < 0.1 for move in battle['battle_timeline']]
        )
        battle_features['p2_hp_pct_shortage'] = sum(
            [move['p2_pokemon_state']['hp_pct'] < 0.1 for move in battle['battle_timeline']]
        )

        # Very Good!
        battle_features['p1_hp_pct_zero'] = sum(
            [move['p1_pokemon_state']['hp_pct'] == 0 for move in battle['battle_timeline']]
        )
        battle_features['p2_hp_pct_zero'] = sum(
            [move['p2_pokemon_state']['hp_pct'] == 0 for move in battle['battle_timeline']]
        )
        battle_features['p1_hp_pct_zero_advantage'] = battle_features['p1_hp_pct_zero'] < battle_features['p2_hp_pct_zero']
        
        battle_features['p1_mean_accuracy'] = \
            np.mean([move['p1_move_details']['accuracy'] for move in battle['battle_timeline'] if move['p1_move_details']])
        battle_features['p2_mean_accuracy'] = \
            np.mean([move['p2_move_details']['accuracy'] for move in battle['battle_timeline'] if move['p2_move_details']])

        # This feature doesn't seem to give any advantage.
        '''        
        battle_features['p1_mean_base_power'] = \
            np.mean([move['p1_move_details']['base_power'] for move in battle['battle_timeline'] if move['p1_move_details']])
        battle_features['p2_mean_base_power'] = \
            np.mean([move['p2_move_details']['base_power'] for move in battle['battle_timeline'] if move['p2_move_details']])
        '''

        battle_features['power_advantage'] = \
            sum([move['p1_move_details']['base_power'] <= move['p2_move_details']['base_power'] for move in battle['battle_timeline'] \
                 if move['p1_move_details'] and move['p2_move_details']])
        
        '''        
        battle_features['accuracy_advantage'] = \
            sum([move['p1_move_details']['accuracy'] > move['p2_move_details']['accuracy'] for move in battle['battle_timeline'] \
                 if move['p1_move_details'] and move['p2_move_details']])
        '''
        
        battle_features['p1_mean_priority'] = \
            np.mean([move['p1_move_details']['priority'] for move in battle['battle_timeline'] if move['p1_move_details']])
        battle_features['p2_mean_priority'] = \
            np.mean([move['p2_move_details']['priority'] for move in battle['battle_timeline'] if move['p2_move_details']])

        battle_features['priority_advantage'] = \
            sum([move['p1_move_details']['priority'] > move['p2_move_details']['priority'] for move in battle['battle_timeline'] \
                 if move['p1_move_details'] and move['p2_move_details']])


        battle_features['hp_pct_advantage'] = \
            sum([move['p1_pokemon_state']['hp_pct'] > move['p2_pokemon_state']['hp_pct'] for move in battle['battle_timeline'] \
                 if move['p1_move_details'] and move['p2_move_details']])

        
        for move in battle['battle_timeline']:
            if move['p1_move_details'] and move['p2_move_details']:
                p1_name = move['p1_pokemon_state']['name']
                p2_name = move['p2_pokemon_state']['name']
                names_comb = f'{p1_name}_{p2_name}'
                battle_features[names_comb] = battle_features.get(names_comb, 0) + 1

                # Move types, counted by pair
                '''
                p1_move_type = move['p1_move_details']['type']
                p2_move_type = move['p2_move_details']['type']
                move_type_comb = f'{p1_move_type}_{p2_move_type}'
                battle_features[move_type_comb] = battle_features.get(move_type_comb, 0) + 1
                '''
                
                # How many times (moves) a Pokemon appears in battle
                battle_features[p1_name] = battle_features.get(p1_name, 0) + 1 
                battle_features[p2_name] = battle_features.get(p2_name, 0) + 1
        
        '''
        for move in battle['battle_timeline']:
            if move['p1_move_details'] and move['p2_move_details']:
            # Last hp_pct per pokemon per battle
                den_p1 = move['p1_move_details']['base_power'] if move['p1_move_details']['base_power'] else 1
                den_p2 = move['p2_move_details']['base_power'] if move['p2_move_details']['base_power'] else 1
                battle_features['t1_power_advantage'] = battle_features.get('t1_power_advantage', 0) + \
                    move['p1_move_details']['base_power'] / den_p2
                battle_features['t2_power_advantage'] = battle_features.get('t2_power_advantage', 0) + \
                    move['p2_move_details']['base_power'] / den_p1
        '''
        
        for move in battle['battle_timeline']:
            # Last hp_pct per pokemon per battle
            battle_features[f'p1_{move['p1_pokemon_state']['name']}_hp_pct'] = move['p1_pokemon_state']['hp_pct']
            battle_features[f'p2_{move['p2_pokemon_state']['name']}_hp_pct'] = move['p2_pokemon_state']['hp_pct']
            
        p1_residual_hp_pct = p2_residual_hp_pct = 0
        p1_players_with_low_hp_pct = p2_players_with_low_hp_pct = 0
        for key, value in battle_features.items():
            if key.endswith('_hp_pct'):
                if key.startswith('p1_'):
                    p1_residual_hp_pct += value
                    if value < LOW_hp_pct_THRESHOLD:
                        p1_players_with_low_hp_pct += 1
                elif key.startswith('p2_'):
                    p2_residual_hp_pct += value
                    if value < LOW_hp_pct_THRESHOLD:
                        p2_players_with_low_hp_pct += 1

        battle_features['p1_residual_hp_pct'] = p1_residual_hp_pct
        battle_features['p2_residual_hp_pct'] = p2_residual_hp_pct
        battle_features['p1_players_with_low_hp_pct'] = p2_players_with_low_hp_pct
        battle_features['p2_players_with_low_hp_pct'] = p2_players_with_low_hp_pct        
        battle_features['residual_hp_pct_advantage'] = int(p1_residual_hp_pct > p2_residual_hp_pct)
            
        feature_list.append(battle_features)
        
    return pd.DataFrame(feature_list).fillna(0)


# SEALED 84.8%
def extract_full_features_SEALED_848(data: list[dict], progress_bar=None) -> pd.DataFrame:
    """
    This is the most complete features extractor.
    """
    feature_list = []
    data_lines = tqdm(data, desc="Extracting features") if progress_bar else data
    
    for battle in data_lines:
        battle_features = {}

        # We also need the ID and the target variable (if they exist)
        battle_features['battle_id'] = battle.get('battle_id')
        battle_features['player_won'] = int(battle['player_won'])
        
        # --- Player 1 Team Features ---
        #p1_team = battle.get('p1_team_details', [])
        p1_team = battle['p1_team_details']
        for stat in POKEMON_STATS:
            battle_features[f'p1_mean_{stat}'] = np.mean([p[f'base_{stat}'] for p in p1_team])
            #battle_features[f'p1_min_{stat}'] = np.min([p.get(f'base_{stat}', 0) for p in p1_team])
            #battle_features[f'p1_max_{stat}'] = np.max([p.get(f'base_{stat}', 0) for p in p1_team])
            #battle_features[f'p1_std_{stat}'] = np.std([p.get(f'base_{stat}', 0) for p in p1_team])
        
        i = 1
        for p in p1_team:
            battle_features[f'p1_name_{i}'] = p['name']
            battle_features[f'p1_level_{i}'] = p['level']
            battle_features[f'p1_types_1_{i}'] = p['types'][0]
            battle_features[f'p1_types_2_{i}'] = p['types'][1]
            i += 1            
        
        battle_features['p1_mean_level'] = np.mean([p.get('level', 0) for p in p1_team])
        
        '''
        for stat in BOOST_STATS:
            battle_features[f'p1_boosts_{stat}'] = np.mean([move['p1_pokemon_state']['boosts'][stat] for move in battle['battle_timeline']])
            battle_features[f'p1_boosts_{stat}'] = np.mean([move['p2_pokemon_state']['boosts'][stat] for move in battle['battle_timeline']])
        '''
        
        # --- Player 2 Lead battle_features ---
        p2_lead = battle['p2_lead_details']
        #if p2_lead:
        # Player 2's lead Pokémon's stats
        for stat in POKEMON_STATS:
            battle_features[f'p2_lead_{stat}'] = p2_lead[f'base_{stat}']
            
        battle_features['p2_lead_name'] = p2_lead['name']
        battle_features['p2_lead_level'] = p2_lead['level']
        battle_features['p2_lead_types_1'] = p2_lead['types'][0]
        battle_features['p2_lead_types_2'] = p2_lead['types'][1]

        for stat in ['spd']:
            base_stat = f'base_{stat}'
            battle_features[f'1st_player_{stat}_advantage'] = battle['p1_team_details'][0][base_stat] / (p2_lead[base_stat] or 1)
        

        # Number of Moves
        # Always 30: no information!
        #battle_features['moves'] = len(battle['battle_timeline'])

        battle_features['p1_mean_hp_pct'] = np.mean([move['p1_pokemon_state']['hp_pct'] for move in battle['battle_timeline']])
        battle_features['p2_mean_hp_pct'] = np.mean([move['p2_pokemon_state']['hp_pct'] for move in battle['battle_timeline']])
  
        battle_features['hp_last_advantage'] = \
            battle['battle_timeline'][-1]['p1_pokemon_state']['hp_pct'] >= \
            battle['battle_timeline'][-1]['p2_pokemon_state']['hp_pct']
    
        battle_features['p1_hp_pct_shortage'] = sum(
            [move['p1_pokemon_state']['hp_pct'] < 0.1 for move in battle['battle_timeline']]
        )
        battle_features['p2_hp_pct_shortage'] = sum(
            [move['p2_pokemon_state']['hp_pct'] < 0.1 for move in battle['battle_timeline']]
        )

        # Very Good!
        battle_features['p1_hp_pct_zero'] = sum(
            [move['p1_pokemon_state']['hp_pct'] == 0 for move in battle['battle_timeline']]
        )
        battle_features['p2_hp_pct_zero'] = sum(
            [move['p2_pokemon_state']['hp_pct'] == 0 for move in battle['battle_timeline']]
        )
        #battle_features['p1_hp_pct_zero_advantage'] = battle_features['p1_hp_pct_zero'] < battle_features['p2_hp_pct_zero']
        battle_features['p1_hp_pct_zero_diff'] = battle_features['p1_hp_pct_zero'] - battle_features['p2_hp_pct_zero']
        
        battle_features['p1_mean_accuracy'] = \
            np.mean([move['p1_move_details']['accuracy'] for move in battle['battle_timeline'] if move['p1_move_details']])
        battle_features['p2_mean_accuracy'] = \
            np.mean([move['p2_move_details']['accuracy'] for move in battle['battle_timeline'] if move['p2_move_details']])

        # This feature doesn't seem to give any advantage.
        '''        
        battle_features['p1_mean_base_power'] = \
            np.mean([move['p1_move_details']['base_power'] for move in battle['battle_timeline'] if move['p1_move_details']])
        battle_features['p2_mean_base_power'] = \
            np.mean([move['p2_move_details']['base_power'] for move in battle['battle_timeline'] if move['p2_move_details']])
        '''

        battle_features['power_advantage'] = \
            sum([move['p1_move_details']['base_power'] <= move['p2_move_details']['base_power'] for move in battle['battle_timeline'] \
                 if move['p1_move_details'] and move['p2_move_details']])
        
        '''        
        battle_features['accuracy_advantage'] = \
            sum([move['p1_move_details']['accuracy'] > move['p2_move_details']['accuracy'] for move in battle['battle_timeline'] \
                 if move['p1_move_details'] and move['p2_move_details']])
        '''
        
        battle_features['p1_mean_priority'] = \
            np.mean([move['p1_move_details']['priority'] for move in battle['battle_timeline'] if move['p1_move_details']])
        battle_features['p2_mean_priority'] = \
            np.mean([move['p2_move_details']['priority'] for move in battle['battle_timeline'] if move['p2_move_details']])

        battle_features['priority_advantage'] = \
            np.mean([move['p1_move_details']['priority'] - move['p2_move_details']['priority'] for move in battle['battle_timeline'] \
                 if move['p1_move_details'] and move['p2_move_details']])


        battle_features['hp_pct_advantage'] = \
            sum([move['p1_pokemon_state']['hp_pct'] > move['p2_pokemon_state']['hp_pct'] for move in battle['battle_timeline'] \
                 if move['p1_move_details'] and move['p2_move_details']])


        for move in battle['battle_timeline']:
            if move['p1_move_details'] and move['p2_move_details']:
                p1_name = move['p1_pokemon_state']['name']
                p2_name = move['p2_pokemon_state']['name']
                names_comb = f'{p1_name}_{p2_name}'
                battle_features[names_comb] = battle_features.get(names_comb, 0) + 1

                # Move types, counted by pair
                '''
                p1_move_type = move['p1_move_details']['type']
                p2_move_type = move['p2_move_details']['type']
                move_type_comb = f'{p1_move_type}_{p2_move_type}'
                battle_features[move_type_comb] = battle_features.get(move_type_comb, 0) + 1
                '''
                
                # How many times (moves) a Pokemon appears in battle
                battle_features[p1_name] = battle_features.get(p1_name, 0) + 1 
                battle_features[p2_name] = battle_features.get(p2_name, 0) + 1

        
        # STATUSES
        player_status_count = {}
        for move in battle['battle_timeline']:
            if move['p1_move_details'] and move['p2_move_details']:
                # Count how many times each status appears in each team in a battle
                player_status_count[f'{move['p1_pokemon_state']['status']}_p0'] = \
                    player_status_count.get(f'{move['p1_pokemon_state']['status']}_p0', 0) + 1
                player_status_count[f'{move['p2_pokemon_state']['status']}_p1'] = \
                    player_status_count.get(f'{move['p2_pokemon_state']['status']}_p1', 0) + 1
        # Statuses advantage
        #battle_features['nostatus_advantage'] = player_status_count.get('nostatus_p0', 0) > player_status_count.get('nostatus_p1', 0)
        for status in ['nostatus', 'frz', 'slp', 'fnt', 'tox', 'psn', 'brn']:
            #battle_features[f'{status}_advantage'] = player_status_count.get(f'{status}_p0', 0) > player_status_count.get(f'{status}_p1', 0)
            battle_features[f'{status}_diff'] = player_status_count.get(f'{status}_p0', 0) - player_status_count.get(f'{status}_p1', 0)

        
        '''
        for move in battle['battle_timeline']:
            if move['p1_move_details'] and move['p2_move_details']:
            # Last hp_pct per pokemon per battle
                den_p1 = move['p1_move_details']['base_power'] if move['p1_move_details']['base_power'] else 1
                den_p2 = move['p2_move_details']['base_power'] if move['p2_move_details']['base_power'] else 1
                battle_features['t1_power_advantage'] = battle_features.get('t1_power_advantage', 0) + \
                    move['p1_move_details']['base_power'] / den_p2
                battle_features['t2_power_advantage'] = battle_features.get('t2_power_advantage', 0) + \
                    move['p2_move_details']['base_power'] / den_p1
        '''
        
        for move in battle['battle_timeline']:
            # Last hp_pct per pokemon per battle
            battle_features[f'p1_{move['p1_pokemon_state']['name']}_hp_pct'] = move['p1_pokemon_state']['hp_pct']
            battle_features[f'p2_{move['p2_pokemon_state']['name']}_hp_pct'] = move['p2_pokemon_state']['hp_pct']
            
        p1_residual_hp_pct = p2_residual_hp_pct = 0
        p1_players_with_low_hp_pct = p2_players_with_low_hp_pct = 0
        for key, value in battle_features.items():
            if key.endswith('_hp_pct'):
                if key.startswith('p1_'):
                    p1_residual_hp_pct += value
                    if value < LOW_hp_pct_THRESHOLD:
                        p1_players_with_low_hp_pct += 1
                elif key.startswith('p2_'):
                    p2_residual_hp_pct += value
                    if value < LOW_hp_pct_THRESHOLD:
                        p2_players_with_low_hp_pct += 1

        battle_features['p1_residual_hp_pct'] = p1_residual_hp_pct
        battle_features['p2_residual_hp_pct'] = p2_residual_hp_pct
        battle_features['p1_players_with_low_hp_pct'] = p1_players_with_low_hp_pct
        battle_features['p2_players_with_low_hp_pct'] = p2_players_with_low_hp_pct        
        battle_features['residual_hp_pct_advantage'] = int(p1_residual_hp_pct > p2_residual_hp_pct)
            
        feature_list.append(battle_features)
        
    return pd.DataFrame(feature_list).fillna(0)
