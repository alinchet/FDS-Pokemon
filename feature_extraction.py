# Different extract* functions: they differentiate on the number of features they extract.
# Only one of them must be called (otherwise the last called one overwrites the features dataframe).
import pandas as pd
import numpy as np

POKEMON_STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']
LOW_hp_pct_THRESHOLD = 0.35

def extract_full_features(data: list[dict], progress_bar=None) -> pd.DataFrame:
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

        #battle_features['1st_player_spd_advantage'] = battle['p1_team_details'][0]['base_spd'] - p2_lead['base_spd']
        battle_features['1st_player_spe_advantage'] = battle['p1_team_details'][0]['base_spe'] - p2_lead['base_spe']
        
        # We also need the ID and the target variable (if they exist)
        battle_features['battle_id'] = battle.get('battle_id')
        battle_features['player_won'] = int(battle['player_won'])

        # Moves
        #battle_features['moves'] = len(battle['battle_timeline'])

        battle_features['p1_mean_hp_pct'] = np.mean([move['p1_pokemon_state']['hp_pct'] for move in battle['battle_timeline']])
        battle_features['p2_mean_hp_pct'] = np.mean([move['p2_pokemon_state']['hp_pct'] for move in battle['battle_timeline']])

        '''
        battle_features['hp_pct_advantage'] = \
            sum([move['p1_pokemon_state']['hp_pct'] >= move['p2_pokemon_state']['hp_pct'] for move in battle['battle_timeline']])
        '''
  
        battle_features['hp_last_advantage'] = \
        battle['battle_timeline'][-1]['p1_pokemon_state']['hp_pct'] >= \
        battle['battle_timeline'][-1]['p2_pokemon_state']['hp_pct']

        
        battle_features['p1_hp_pct_shortage'] = sum(
            [move['p1_pokemon_state']['hp_pct'] < 0.1 for move in battle['battle_timeline']]
        )
        battle_features['p2_hp_pct_shortage'] = sum(
            [move['p2_pokemon_state']['hp_pct'] < 0.1 for move in battle['battle_timeline']]
        )

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

        for move in battle['battle_timeline']:
            if move['p1_move_details'] and move['p2_move_details']:
                p1_name = move['p1_pokemon_state']['name']
                p2_name = move['p2_pokemon_state']['name']
                names_comb = f'{p1_name}_{p2_name}'
                battle_features[names_comb] = battle_features.get(names_comb, 0) + 1

                #p1_move_type = move['p1_move_details']['type']
                #p2_move_type = move['p2_move_details']['type']
                #move_type_comb = f'{p1_move_type}_{p2_move_type}'
                #battle_features[move_type_comb] = battle_features.get(move_type_comb, 0) + 1
                
                #battle_features[p1_name] = battle_features.get(p1_name, 0) + 1 
                #battle_features[p2_name] = battle_features.get(p2_name, 0) + 1

        
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
        battle_features['hp_pct_advantage'] = int(p1_residual_hp_pct > p2_residual_hp_pct)
            
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

        battle_features_tmp['p1_hp_shortage'] = sum(
            [move['p1_pokemon_state']['hp_pct'] < 0.02 for move in battle['battle_timeline']]
        )
        battle_features_tmp['p2_hp_shortage'] = sum(
            [move['p2_pokemon_state']['hp_pct'] < 0.02 for move in battle['battle_timeline']]
        )

        battle_features['p1_hp_shortage_advantage'] = int(battle_features_tmp['p1_hp_shortage'] < battle_features_tmp['p2_hp_shortage'])

        battle_features['p1_hp_last_advantage'] = int(
            battle['battle_timeline'][-1]['p1_pokemon_state']['hp_pct'] >= \
            battle['battle_timeline'][-1]['p2_pokemon_state']['hp_pct']
        )

        battle_features['p1_mean_accuracy'] = \
            np.mean([move['p1_move_details']['accuracy'] for move in battle['battle_timeline'] if move['p1_move_details']])
        battle_features['p2_mean_accuracy'] = \
            np.mean([move['p2_move_details']['accuracy'] for move in battle['battle_timeline'] if move['p2_move_details']])

        battle_features['p1_mean_priority'] = \
            np.mean([move['p1_move_details']['priority'] for move in battle['battle_timeline'] if move['p1_move_details']])
        battle_features['p2_mean_priority'] = \
            np.mean([move['p2_move_details']['priority'] for move in battle['battle_timeline'] if move['p2_move_details']])

        for move in battle['battle_timeline']:
            # Last hp_pct per pokemon per team per battle
            battle_features[f'p1_{move['p1_pokemon_state']['name']}_hp_pct'] = move['p1_pokemon_state']['hp_pct']
            battle_features[f'p2_{move['p2_pokemon_state']['name']}_hp_pct'] = move['p2_pokemon_state']['hp_pct']

            #if move['p1_move_details'] and move['p2_move_details']:
            #    move_type_pair = f"{move['p1_move_details']['type']}_{move['p2_move_details']['type']}"
            #    battle_features[move_type_pair] = battle_features.get(move_type_pair, 0) + 1
            
        
        feature_list.append(battle_features)
        
    return pd.DataFrame(feature_list).fillna(0)