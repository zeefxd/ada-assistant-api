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
    """A class for detecting and processing Spotify music commands"""

    def __init__(self):
        self.music_patterns = [
            # Polecenia odtwarzania
            r"(?:puść|włącz|odtwórz|zagraj|graj|puśc|zagrj|odtwarz|pusć|odtworz|włacz)\s+(.+)",
            r"(?:chcę usłyszeć|chciałbym usłyszeć|chciałabym usłyszeć|chciałbym posłuchać|chciałabym posłuchać|zagraj mi)\s+(.+)",
            r"(?:poproszę|proszę o|proszę|daj|daj mi)\s+(.+?)(?:\s+(?:piosenkę|utwór|muzykę|piosenka))?",
            r"(?:słuchaj|słuchać|słuchanie|muzyka|posłuchajmy|posłuchamy)\s+(.+)",
            r"(?:zagraj|puść)(?:\s+piosenkę|\s+utwór|\s+coś)?\s+(?:od|przez|z repertuaru|autorstwa|wykonawcy)?\s+(.+)",
            r"(?:znasz|znasz może|masz|masz może)(?:\s+piosenkę|\s+utwór)?\s+(.+)",
            # Polecenia kontroli odtwarzania
            r"(?:zatrzymaj|wstrzymaj|pauza|stop|pause|zatszymaj|wsztymaj|zróbpauę|pałza|zastanów|zaczekaj)(?:\s+(?:muzykę|odtwarzanie|piosenkę|to))?",
            r"(?:wznów|kontynuuj|start|play|resume|dalej|graj dalej|kontynułj|wznuf|odtwarzaj dalej)(?:\s+(?:muzykę|odtwarzanie|piosenkę|to))?",
            r"(?:następny|następna|kolejny|kolejna|dalej|next|skip|pomiń|następ|nastempny|nexxt|skipnij|idź dalej)(?:\s+(?:utwór|piosenka|kawałek|numer|track))?",
            r"(?:poprzedni|poprzednia|wróć|cofnij|back|previous|wstecz|popszedni|poprszedni|wróc|poprzednia piosenka)(?:\s+(?:utwór|piosenka|kawałek|numer|track))?",
            # Polecenia głośności
            r"(?:podgłośnij|zwiększ głośność|głośniej|pogłośnij|daj głośniej|podkręć|mocniej|głośno|zwiększ|zrób głośniej|głośnieeej)(?:\s+o\s+(\d+))?(?:\s+(?:procent|%|punktów))?",
            r"(?:ścisz|zmniejsz głośność|ciszej|przycisz|zrób ciszej|ściż|zciż|ciszej|ścisz to|przycisz głośność)(?:\s+o\s+(\d+))?(?:\s+(?:procent|%|punktów))?",
            r"(?:ustaw|zmień|daj|zrób)(?:\s+(?:głośność|volume))?\s+na\s+(\d+)(?:\s+(?:procent|%|punktów))?",
            r"(?:ustaw|zmień|daj)(?:\s+(?:poziom głośności|głośność|poziom))?\s+(\d+)(?:\s+(?:procent|%|punktów))?",
            # Pytania o aktualny utwór
            r"(?:co\s+(?:to\s+za|jest\s+to\s+za|to\s+jest\s+za))?\s+(?:piosenka|utwór|kawałek|muzyka|numer|track)(?:\s+gra)?",
            r"(?:jaka|jaki|co|kto)(?:\s+teraz)?\s+(?:leci|gra|jest włączone|jest odtwarzane|słychać)",
            r"(?:kto|jaki artysta|jaka artystka|jaki wykonawca|jaka wykonawczyni)(?:\s+to)?\s+(?:śpiewa|gra|wykonuje|jest)",
            # Polecenia dotyczące playlist/albumów
            r"(?:puść|włącz|odtwórz|zagraj|graj)(?:\s+(?:playlistę|album|płytę))?\s+(.+)",
            r"(?:ustaw|włącz)(?:\s+tryb)?\s+(?:shuffle|losowy|losowania|losowo|losowe|przypadkowy|przypadkowe)",
            r"(?:wyłącz)(?:\s+tryb)?\s+(?:shuffle|losowy|losowania|losowo|losowe|przypadkowy|przypadkowe)",
            # Bardziej ogólne polecenia
            r"(?:spotify|muzyka|audio)(?:\s+(.+))?",
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
                return True, CommandType.MUSIC, self._extract_music_params(text)

        return False, None, {}

    def _extract_music_params(self, text: str) -> Dict[str, Any]:
        """
        Extracts parameters from the music command.

        Args:
            text (str): Command text

        Returns:
            Dict[str, Any]: Parameters of the music command
        """
        text = text.lower().strip()
        params = {}

        play_match = re.search(r"(?:puść|włącz|odtwórz|zagraj|graj)\s+(.+)", text)
        if play_match:
            query = play_match.group(1).strip()
            params["action"] = "play"
            params["query"] = query
            return params

        if re.search(
            r"(?:zatrzymaj|wstrzymaj|pauza|stop|pause)(?:\s+muzykę|\s+odtwarzanie)?",
            text,
        ):
            params["action"] = "pause"
            return params

        if re.search(
            r"(?:wznów|kontynuuj|start|play|resume)(?:\s+muzykę|\s+odtwarzanie)?", text
        ):
            params["action"] = "resume"
            return params

        if re.search(
            r"(?:następny|następna|kolejny|kolejna|dalej|next)(?:\s+utwór|\s+piosenka)?",
            text,
        ):
            params["action"] = "next"
            return params

        if re.search(
            r"(?:poprzedni|poprzednia|wróć|previous)(?:\s+utwór|\s+piosenka)?", text
        ):
            params["action"] = "previous"
            return params

        volume_up_match = re.search(
            r"(?:podgłośnij|zwiększ głośność|głośniej)(?:\s+o\s+(\d+))?", text
        )
        if volume_up_match:
            params["action"] = "volume_up"
            params["value"] = (
                int(volume_up_match.group(1)) if volume_up_match.group(1) else 10
            )
            return params

        volume_down_match = re.search(
            r"(?:ścisz|zmniejsz głośność|ciszej)(?:\s+o\s+(\d+))?", text
        )
        if volume_down_match:
            params["action"] = "volume_down"
            params["value"] = (
                int(volume_down_match.group(1)) if volume_down_match.group(1) else 10
            )
            return params

        volume_set_match = re.search(
            r"(?:ustaw|zmień)(?:\s+głośność)?\s+na\s+(\d+)", text
        )
        if volume_set_match:
            params["action"] = "volume_set"
            params["value"] = int(volume_set_match.group(1))
            return params

        return {"action": "unknown", "raw_text": text}

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

        if action == "play" and "query" in params:
            message = f"Odtwarzam {params['query']}"
        elif action == "pause":
            message = "Wstrzymuję odtwarzanie"
        elif action == "resume":
            message = "Wznawiam odtwarzanie"
        elif action == "next":
            message = "Przechodzę do następnego utworu"
        elif action == "previous":
            message = "Przechodzę do poprzedniego utworu"
        elif action == "volume_up":
            message = f"Zwiększam głośność o {params.get('value', 10)}%"
        elif action == "volume_down":
            message = f"Zmniejszam głośność o {params.get('value', 10)}%"
        elif action == "volume_set":
            message = f"Ustawiam głośność na {params.get('value', 50)}%"
        else:
            message = "Wykonuję polecenie muzyczne"

        return {
            "command_type": "music",
            "params": params,
            "user_message": message,
            "requires_spotify_token": True,
        }
