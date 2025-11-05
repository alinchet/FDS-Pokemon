import numpy as np
import pandas as pd

# ==============================
# CONSTANTES
# ==============================
POKEMON_STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']
LOW_hp_pct_THRESHOLD = 0.35

# Table simplifiée d’efficacité des types (Gen 1)
# Valeurs : 2 = super efficace, 0.5 = pas très efficace, 1 = neutre, 0 = inefficace
TYPE_EFFECTIVENESS = {
    'Fire': {'Grass': 2, 'Water': 0.5, 'Fire': 0.5, 'Ice': 2},
    'Water': {'Fire': 2, 'Grass': 0.5, 'Ground': 2, 'Rock': 2},
    'Grass': {'Water': 2, 'Fire': 0.5, 'Rock': 2, 'Ground': 2},
    'Electric': {'Water': 2, 'Ground': 0, 'Flying': 2},
    'Ground': {'Fire': 2, 'Electric': 2, 'Flying': 0},
    'Rock': {'Fire': 2, 'Ice': 2, 'Flying': 2, 'Bug': 2},
    'Ice': {'Grass': 2, 'Ground': 2, 'Flying': 2, 'Water': 0.5, 'Fire': 0.5},
    'Normal': {'Rock': 0.5, 'Ghost': 0},
    'Fighting': {'Normal': 2, 'Flying': 0.5, 'Rock': 2, 'Bug': 0.5, 'Ghost': 0},
    'Psychic': {'Fighting': 2, 'Poison': 2},
    # ...
}


# ==============================
# FONCTIONS AUXILIAIRES
# ==============================

def type_effectiveness(move_type, defender_types):
    """
    Calcule un multiplicateur d'efficacité de type basé sur une table simplifiée.
    """
    eff = 1.0
    if move_type is None:
        return eff
    if move_type not in TYPE_EFFECTIVENESS:
        return eff
    for d_type in defender_types:
        if d_type in TYPE_EFFECTIVENESS[move_type]:
            eff *= TYPE_EFFECTIVENESS[move_type][d_type]
    return eff


def compute_expected_damage(move, attacker_types, defender_types):
    """
    Approximation du dommage attendu :
        base_power * accuracy * STAB * type_effectiveness
    """
    if not move:
        return 0.0
    base_power = move.get('base_power', 0) or 0
    accuracy = move.get('accuracy', 100) / 100
    move_type = move.get('type')
    
    # STAB (Same Type Attack Bonus)
    stab = 1.5 if move_type in attacker_types else 1.0

    # Efficacité du type
    type_eff = type_effectiveness(move_type, defender_types)

    return base_power * accuracy * stab * type_eff


# ==============================
# EXTRACTION DES FEATURES
# ==============================

def extract_full_features(data, progress_bar=None):
    feature_list = []

    for battle in (progress_bar or data):
        battle_features = {}

        # --- Player 1: team-level static features
        p1_team = battle['p1_team_details']
        for stat in POKEMON_STATS:
            battle_features[f'p1_mean_{stat}'] = np.mean([p[f'base_{stat}'] for p in p1_team])

        # Ajout des informations sur le premier Pokémon (lead)
        p1_lead = p1_team[0]
        p1_types = [p1_lead.get('types_1'), p1_lead.get('types_2')]
        battle_features['p1_mean_level'] = np.mean([p['level'] for p in p1_team])

        # --- Player 2: lead static features
        p2_lead = battle['p2_lead_details']
        p2_types = [p2_lead.get('types_1'), p2_lead.get('types_2')]
        for stat in POKEMON_STATS:
            battle_features[f'p2_{stat}'] = p2_lead[f'base_{stat}']
        battle_features['p2_level'] = p2_lead['level']

        # --- Basic features
        battle_features['battle_id'] = battle.get('battle_id', -1)
        battle_features['player_won'] = int(battle.get('player_won', 0))

        # --- Dynamic timeline analysis
        timeline = battle.get('battle_timeline', [])

        # HP metrics
        p1_hp, p2_hp = [], []
        p1_expected, p2_expected = [], []

        for move in timeline:
            if move.get('p1_hp_pct') is not None and move.get('p2_hp_pct') is not None:
                p1_hp.append(move['p1_hp_pct'])
                p2_hp.append(move['p2_hp_pct'])

            # Expected damage computation
            if move.get('p1_move_details') and move.get('p2_move_details'):
                dmg1 = compute_expected_damage(move['p1_move_details'], p1_types, p2_types)
                dmg2 = compute_expected_damage(move['p2_move_details'], p2_types, p1_types)
                p1_expected.append(dmg1)
                p2_expected.append(dmg2)

        battle_features['p1_mean_hp_pct'] = np.mean(p1_hp) if p1_hp else 0
        battle_features['p2_mean_hp_pct'] = np.mean(p2_hp) if p2_hp else 0
        battle_features['hp_last_advantage'] = int((p1_hp[-1] if p1_hp else 0) > (p2_hp[-1] if p2_hp else 0))

        # --- NEW: Expected damage advantage
        battle_features['p1_expected_damage'] = np.mean(p1_expected) if p1_expected else 0
        battle_features['p2_expected_damage'] = np.mean(p2_expected) if p2_expected else 0
        battle_features['expected_damage_advantage'] = int(battle_features['p1_expected_damage'] >
                                                           battle_features['p2_expected_damage'])

        # --- NEW: Type advantage (based on STAB × effectiveness)
        p1_type_score = np.mean([type_effectiveness(t, p2_types) for t in p1_types if t])
        p2_type_score = np.mean([type_effectiveness(t, p1_types) for t in p2_types if t])
        battle_features['type_advantage_score'] = p1_type_score - p2_type_score

        # --- Low HP counts (corrected bug)
        p1_low_hp = np.sum(np.array(p1_hp) < LOW_hp_pct_THRESHOLD) if p1_hp else 0
        p2_low_hp = np.sum(np.array(p2_hp) < LOW_hp_pct_THRESHOLD) if p2_hp else 0
        battle_features['p1_players_with_low_hp_pct'] = p1_low_hp
        battle_features['p2_players_with_low_hp_pct'] = p2_low_hp

        # --- Power advantage (corrected logic)
        moves_with_power = [m for m in timeline if m.get('p1_move_details') and m.get('p2_move_details')]
        battle_features['power_advantage'] = sum([
            m['p1_move_details']['base_power'] > m['p2_move_details']['base_power']
            for m in moves_with_power
        ])

        feature_list.append(battle_features)

    return pd.DataFrame(feature_list).fillna(0)
