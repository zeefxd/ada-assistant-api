import re
import logging
from typing import Dict, Any, Optional, Tuple
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CommandType(Enum):
    MUSIC = "music"
    UNKNOWN = "unknown"


class CommandDetector:

    def __init__(self):
        self.music_patterns = [
            # Play commands
            r"^(?:puść|włącz|odtwórz|zagraj)\s+(?:utwór|piosenkę|muzykę|kawałek)?\s*(.+?)$",
            
            # Playback control commands
            r"^(?:zatrzymaj|wstrzymaj|pauza|stop|pause)(?:\s+(?:utwór|piosenkę|muzykę|odtwarzanie|to))?.*$",
            r"^(?:wznów|kontynuuj|start|play|resume)(?:\s+(?:utwór|piosenkę|muzykę|odtwarzanie|to))?.*$",
            r"^(?:następny|następna|kolejny|kolejna|dalej|next)(?:\s+(?:utwór|piosenka|kawałek|numer))?.*$",
            r"^(?:poprzedni|poprzednia|wróć|cofnij)(?:\s+(?:utwór|piosenka|kawałek|numer))?.*$",
            
            # Volume commands
            r"^(?:podgłośnij|zwiększ\s+głośność|głośniej)(?:\s+(?:utwór|piosenkę|muzykę|to))?(?:.*?(\d+))?.*$",
            r"^(?:ścisz|zmniejsz\s+głośność|ciszej)(?:\s+(?:utwór|piosenkę|muzykę|to))?(?:.*?(\d+))?.*$",
            r"^(?:ustaw|zmień)\s+głośność\s+na\s+(\d+).*$",
            
            # Current track
            r"^(?:jaki|jaka)(?:\s+(?:aktualny|aktualna|obecny|obecna))?\s+(?:utwór|piosenka)(?:\s+(?:jest|jest teraz|teraz))?\s+(?:odtwarzany|odtwarzana|grany|grana|leci)?.*$",
            r"^co(?:\s+(?:teraz|aktualnie))?\s+(?:leci|gra|jest\s+odtwarzane).*$",
            r"^(?:co\s+to\s+za|jaka\s+to)\s+(?:piosenka|utwór|muzyka).*$"
        ]

    def detect_command(
        self, text: str
    ) -> Tuple[bool, Optional[CommandType], Dict[str, Any]]:
        """
        Detects if the text contains a music command.

        Args:
            text (str): Text to analyze

        Returns:
            Tuple[bool, Optional[CommandType], Dict[str, Any]]:
            - Whether a command was detected
            - Command type
            - Parameters extracted from the command
        """
        if not text:
            return False, None, {}

        text = text.lower().strip()

        for pattern in self.music_patterns:
            match = re.search(pattern, text)
            if match:
                logger.info(f"Wykryto polecenie muzyczne: {text}")
                return True, CommandType.MUSIC, self._extract_music_params(text, pattern, match)

        return False, None, {}

    def _extract_music_params(self, text: str, pattern: str, match) -> Dict[str, Any]:
        """
        Extracts parameters from the music command.

        Args:
            text (str): Command text
            pattern (str): Pattern that matched
            match: Regex match object

        Returns:
            Dict[str, Any]: Parameters of the music command
        """
        params = {}
        
        spotify_suffix_match = re.search(r'(?:na|w)\s+spotify\s*$', text.lower())
        params["targetPlatform"] = "Spotify" if spotify_suffix_match else "none"

        if re.search(r"puść|włącz|odtwórz|zagraj", pattern):
            if match.group(1):
                params["action"] = "play"
                
                query = match.group(1).strip()
                if spotify_suffix_match:
                    query = re.sub(r'\s+(?:na|w)\s+spotify\s*$', '', query, flags=re.IGNORECASE)
                    
                params["query"] = query
            return params

        if re.search(r"zatrzymaj|wstrzymaj|pauza|stop|pause", pattern):
            params["action"] = "pause"
            return params

        if re.search(r"wznów|kontynuuj|start|play|resume", pattern):
            params["action"] = "resume"
            return params

        if re.search(r"następny|następna|kolejny|kolejna|dalej|next", pattern):
            params["action"] = "next"
            return params

        if re.search(r"poprzedni|poprzednia|wróć|cofnij", pattern):
            params["action"] = "previous"
            return params

        if re.search(r"podgłośnij|zwiększ\s+głośność|głośniej", pattern):
            params["action"] = "volume_up"
            volume_match = re.search(r'(\d+)', text)
            params["value"] = int(volume_match.group(1)) if volume_match else 10
            return params

        if re.search(r"ścisz|zmniejsz\s+głośność|ciszej", pattern):
            params["action"] = "volume_down"
            volume_match = re.search(r'(\d+)', text)
            params["value"] = int(volume_match.group(1)) if volume_match else 10
            return params

        if re.search(r"ustaw.*głośność.*na", pattern):
            params["action"] = "volume_set"
            volume_match = re.search(r'(\d+)', text)
            params["value"] = int(volume_match.group(1)) if volume_match else 50
            return params

        if re.search(r"jaki|jaka|co.*leci|co.*gra|co.*odtwarzane|co.*to.*za", pattern):
            params["action"] = "current_track"
            return params

        return {"action": "unknown", "raw_text": text, "targetPlatform": "none"}

    def execute_command(
        self, cmd_type: CommandType, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepares information about the command to be executed.

        Args:
            cmd_type (CommandType): Type of the command
            params (Dict[str, Any]): Command parameters

        Returns:
            Dict[str, Any]: Information about the command
        """
        if cmd_type != CommandType.MUSIC:
            return {
                "command_type": "unknown",
                "success": False,
                "user_message": "Nierozpoznane polecenie",
            }

        action = params.get("action", "")
        platform = params.get("targetPlatform", "none")
        platform_msg = f" na {platform}" if platform != "none" else ""

        if action == "play" and "query" in params:
            message = f"Odtwarzam {params['query']}{platform_msg}"
        elif action == "pause":
            message = f"Wstrzymuję odtwarzanie{platform_msg}"
        elif action == "resume":
            message = f"Wznawiam odtwarzanie{platform_msg}"
        elif action == "next":
            message = f"Przechodzę do następnego utworu{platform_msg}"
        elif action == "previous":
            message = f"Przechodzę do poprzedniego utworu{platform_msg}"
        elif action == "volume_up":
            message = f"Zwiększam głośność o {params.get('value', 10)}%{platform_msg}"
        elif action == "volume_down":
            message = f"Zmniejszam głośność o {params.get('value', 10)}%{platform_msg}"
        elif action == "volume_set":
            message = f"Ustawiam głośność na {params.get('value', 50)}%{platform_msg}"
        elif action == "current_track":
            message = f"Sprawdzam aktualnie odtwarzany utwór{platform_msg}"
        else:
            message = f"Wykonuję polecenie muzyczne{platform_msg}"

        return {
            "type": "music",
            "params": params,
            "user_message": message,
            "requires_spotify_token": platform.lower() == "spotify",
        }
