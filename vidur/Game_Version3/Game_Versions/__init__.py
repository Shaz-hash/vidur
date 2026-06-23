### بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ 

from __future__ import annotations

def resolve_game_version(game_version: str | None):

    from .game_version_1 import GameVersion1Rules
    from .game_version_2 import GameVersion2Rules
    
    if game_version is None:
        return GameVersion1Rules()
    else :
        key_name = game_version.strip().lower()
        if key_name == "game_version_1" or key_name == "v1" or key_name == "version_1" or key_name == "1":
            return GameVersion1Rules()
        elif key_name == "game_version_2" or key_name == "v2" or key_name == "version_2" or key_name == "2":
            return GameVersion2Rules()
        raise ValueError(f"Unknown game version: {game_version}")




