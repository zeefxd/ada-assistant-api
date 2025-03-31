import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class CommandType(Enum):
    CALENDAR = "calendar"
    REMINDER = "reminder"
    WEATHER = "weather"
    EMAIL = "email"
    UNKNOWN = "unknown"
    MUSIC = "music"

class CommandDetector:
    """Klasa do wykrywania i przetwarzania poleceń w zapytaniach użytkownika."""
    
    def __init__(self):
        self.command_patterns = {
            CommandType.CALENDAR: [
                r"(?:utwórz|dodaj|zaplanuj)(?:\s+(?:nowe|nowy))?\s+(?:wydarzenie|spotkanie)(?:\s+w\s+(?:kalendarzu|google\s+calendar))?",
                r"zapisz(?:\s+to)?\s+w\s+(?:kalendarzu|google\s+calendar)",
            ],
            CommandType.REMINDER: [
                r"(?:utwórz|dodaj|ustaw)(?:\s+(?:nowe|nowy|nową))?\s+(?:przypom(?:n|)ienie|alert|powiadomienie)",
                r"przypomnij\s+(?:mi|nam)?\s+(?:o|że|aby)",
                r"zapisz\s+przypom(?:n|)ienie",
                r"ustaw(?:\s+(?:nowe|nowy|nową))?\s+(?:przypom(?:n|)ienie|alert|powiadomienie)",
            ],
            CommandType.WEATHER: [
                r"(?:jaka|jak)(?:\s+jest|będzie)?\s+pogoda(?:\s+w)?",
                r"sprawdź(?:\s+(?:aktualną|dzisiejszą|jutrzejszą))?\s+pogodę(?:\s+w)?",
            ],
            CommandType.EMAIL: [
                r"(?:wyślij|napisz)(?:\s+(?:nowy|nowego))?\s+(?:email|mail|wiadomość)(?:\s+do)?",
                r"(?:stwórz|utwórz)(?:\s+(?:nowy|nowego))?\s+(?:email|mail|wiadomość)(?:\s+do)?",
            ],
            CommandType.MUSIC: [
            r"(?:zatrzymaj|pauza|wstrzymaj|stop)(?:\s+(?:muzykę|odtwarzanie|spotify|utwór|piosenkę))?",
            r"(?:wznów|kontynuuj|odtwórz|play|graj)(?:\s+(?:muzykę|odtwarzanie|spotify|utwór|piosenkę))?",
            r"(?:następn(?:y|a)|kolejn(?:y|a))(?:\s+(?:utwór|piosenk(?:a|ę)|utwor))?",
            r"(?:poprzedni(?:y|a)|wcześniejsz(?:y|a))(?:\s+(?:utwór|piosenk(?:a|ę)|utwor))?",

            r"(?:zwiększ|podgłośnij|podnieś)(?:\s+(?:głośność|volume))?",
            r"(?:zmniejsz|ścisz|zmniejsz)(?:\s+(?:głośność|volume))?",
            r"(?:ustaw)(?:\s+(?:głośność|volume))(?:\s+na)?(?:\s+(\d{1,3})(?:\s*%|\s+procent)?)?",

            r"(?:włącz|odtwórz|graj)(?:\s+piosenkę|utwór)?(?:\s+pod\s+tytułem)?(?:\s+(.+))?(?:\s+(?:na|w)\s+spotify)?",
            r"(?:włącz|odtwórz|graj)(?:\s+(?:muzykę|utwory|piosenki))(?:\s+(?:wykonawcy|artysty))?(?:\s+(.+))?(?:\s+(?:na|w)\s+spotify)?",
            r"(?:włącz|odtwórz|graj)(?:\s+(?:album|płytę)(?:\s+pod\s+tytułem)?(?:\s+(.+)))?(?:\s+(?:na|w)\s+spotify)?",
            r"(?:włącz|odtwórz|graj)(?:\s+(?:playlistę|playlist)(?:\s+pod\s+nazwą)?(?:\s+(.+)))?(?:\s+(?:na|w)\s+spotify)?",
        ],
        }
        
    def detect_command(self, text: str) -> Tuple[bool, Optional[CommandType], Dict[str, Any]]:
        """ Wykrywa polecenia w tekście użytkownika. """
        text = text.lower().strip()
        
        logger.info(f"Analizowanie tekstu pod kątem poleceń: '{text}'")
        
        for cmd_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    logger.info(f"Wykryto polecenie typu: {cmd_type.value} z wzorcem: {pattern}")
                    params = self._extract_command_params(text, cmd_type)
                    logger.info(f"Wyodrębnione parametry: {params}")
                    return True, cmd_type, params
        
        logger.info("Nie wykryto żadnego polecenia")
        return False, None, {}
    
    def _extract_command_params(self, text: str, cmd_type: CommandType) -> Dict[str, Any]:
        """Ekstrahuje parametry z tekstu polecenia w zależności od typu."""
        params = {}
        
        if cmd_type == CommandType.CALENDAR or cmd_type == CommandType.REMINDER:
            # Próba wyodrębnienia daty i godziny
            date_patterns = [
                r'(\d{1,2}(?:\s+|\.|\-)\d{1,2}(?:\s+|\.|\-)\d{2,4})',
                r'(\d{1,2}(?:\s+|\.|\-)\d{1,2})',
                r'(jutro|pojutrze|dzisiaj|dziś|dziś|za tydzień|za\s+\d+\s+dni)', 
                r'(jutrzejs(?:zy|ki)\s+dzie(?:ń|n))',
                r'(pojutrzejs(?:zy|ki)\s+dzie(?:ń|n))',
                r'na\s+(jutro|pojutrze|dziś|dzisiaj)',
                r'w\s+(poniedzialek|poniedziałek|wtorek|środę|srode|czwartek|piątek|piatek|sobotę|sobote|niedzielę|niedziele)', # Dni tygodnia
            ]
            
            time_patterns = [
                r'(\d{1,2}:\d{2})',
                r'o\s+(?:godzin(?:ie|e|nie)?)\s+(\d{1,2}(?::\d{2})?)',
                r'o\s+(\d{1,2}(?::\d{2})?)',
                r'na\s+(?:godzin(?:ę|e|nie)?)\s+(\d{1,2}(?::\d{2})?)',
                r'o\s+(\d{1,2})(?::\d{2})?\s+(?:rano|wieczorem|po\s+południu|w\s+nocy)',
                r'(?:na|o)\s+(\d{1,2})',
            ]
            
            # Szukanie daty
            for pattern in date_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    date_value = match.group(1).strip()
                    
                    # Mapowanie na standardowe wartości
                    if re.match(r'jutrzejs(?:zy|ki)\s+dzie(?:ń|n)', date_value, re.IGNORECASE):
                        date_value = "jutro"
                    elif re.match(r'pojutrzejs(?:zy|ki)\s+dzie(?:ń|n)', date_value, re.IGNORECASE):
                        date_value = "pojutrze"
                    elif re.match(r'na\s+(jutro|pojutrze|dziś|dzisiaj)', date_value, re.IGNORECASE):
                        date_value = re.search(r'na\s+(jutro|pojutrze|dziś|dzisiaj)', date_value, re.IGNORECASE).group(1)
                    
                    params['date'] = date_value
                    break
                
            for pattern in time_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    time_value = match.group(1).strip()
                    if ':' not in time_value:
                        time_value += ':00'
                    params['time'] = time_value
                    break
            
            title_patterns = [
                r'(?:temat|tytuł|nazwa|treść|tresc)(?:\s+to)?[:]?\s+[""]?([^""\n]+)[""]?',
                r'(?:przypom(?:n|)ienie|alert|powiadomienie)(?:\s+o)?[:]?\s+[""]?([^""\n]+)[""]?',
                r'(?:przypom(?:n|)ij(?:\s+mi)?)\s+(?:o|że|aby)\s+([^,\.]+)', 
            ]
            
            for pattern in title_patterns:
                title_match = re.search(pattern, text, re.IGNORECASE)
                if title_match:
                    params['title'] = title_match.group(1).strip()
                    break
            
        elif cmd_type == CommandType.WEATHER:
            location_match = re.search(r'pogoda(?:\s+w|dla)?\s+([^\s,\.]+(?:\s+[^\s,\.]+)?)', text)
            if location_match:
                params['location'] = location_match.group(1).strip()
        elif cmd_type == CommandType.MUSIC:
            if re.search(r"(?:zatrzymaj|pauza|wstrzymaj|stop)", text, re.IGNORECASE):
                params["action"] = "pause"
            elif re.search(r"(?:wznów|kontynuuj|odtwórz|play|graj)$", text, re.IGNORECASE):
                params["action"] = "play"
            elif re.search(r"(?:następn(?:y|a)|kolejn(?:y|a))", text, re.IGNORECASE):
                params["action"] = "next"
            elif re.search(r"(?:poprzedni(?:y|a)|wcześniejsz(?:y|a))", text, re.IGNORECASE):
                params["action"] = "previous"
                
            volume_up = re.search(r"(?:zwiększ|podgłośnij|podnieś)", text, re.IGNORECASE)
            volume_down = re.search(r"(?:zmniejsz|ścisz)", text, re.IGNORECASE)
            volume_set = re.search(r"(?:ustaw)(?:\s+(?:głośność|volume))(?:\s+na)?(?:\s+(\d{1,3})(?:\s*%|\s+procent)?)?", text, re.IGNORECASE)
            
            if volume_up:
                params["action"] = "volume_up"
                volume_value = re.search(r"(?:o|do)(?:\s+|)(\d{1,3})(?:\s*%|\s+procent)?", text, re.IGNORECASE)
                if volume_value:
                    params["value"] = volume_value.group(1)
                    
            elif volume_down:
                params["action"] = "volume_down"
                volume_value = re.search(r"(?:o|do)(?:\s+|)(\d{1,3})(?:\s*%|\s+procent)?", text, re.IGNORECASE)
                if volume_value:
                    params["value"] = volume_value.group(1)
                    
            elif volume_set:
                params["action"] = "volume_set"
                if volume_set.group(1):
                    params["value"] = volume_set.group(1)
                else:
                    params["value"] = "50"  # Domyślna wartość głośności
                
            search_patterns = [
                r"(?:włącz|odtwórz|graj)(?:\s+piosenkę|utwór)?(?:\s+pod\s+tytułem)?(?:\s+(.+))(?:\s+(?:na|w)\s+spotify)?",
                r"(?:włącz|odtwórz|graj)(?:\s+(?:muzykę|utwory|piosenki))(?:\s+(?:wykonawcy|artysty))?(?:\s+(.+))(?:\s+(?:na|w)\s+spotify)?",
                r"(?:włącz|odtwórz|graj)(?:\s+(?:album|płytę)(?:\s+pod\s+tytułem)?(?:\s+(.+)))(?:\s+(?:na|w)\s+spotify)?",
                r"(?:włącz|odtwórz|graj)(?:\s+(?:playlistę|playlist)(?:\s+pod\s+nazwą)?(?:\s+(.+)))(?:\s+(?:na|w)\s+spotify)?",
            ]
            
            if "action" not in params and any(re.search(pattern, text, re.IGNORECASE) for pattern in search_patterns):
                params["action"] = "search"
                if re.search(r"(?:piosenkę|utwór)", text, re.IGNORECASE):
                    params["search_type"] = "track"
                elif re.search(r"(?:wykonawcy|artysty)", text, re.IGNORECASE):
                    params["search_type"] = "artist"
                elif re.search(r"(?:album|płytę)", text, re.IGNORECASE):
                    params["search_type"] = "album"
                elif re.search(r"(?:playlistę|playlist)", text, re.IGNORECASE):
                    params["search_type"] = "playlist"
                else:
                    params["search_type"] = "track"
                
                for pattern in search_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match and match.group(1):
                        params["query"] = match.group(1).strip()
                        break
                
        return params
    
    def execute_command(self, cmd_type: CommandType, params: Dict[str, Any]) -> Dict[str, Any]:
        """ Zwraca informacje o wykrytym poleceniu w formacie dla ESP32. """
        
        result = {
            "command_type": cmd_type.value if cmd_type else "unknown",
            "parameters": params,
        }
        
        if cmd_type == CommandType.CALENDAR:
            result["description"] = "Dodanie wydarzenia do kalendarza"
            result["esp32_handler"] = "calendar_handler"
            
        elif cmd_type == CommandType.REMINDER:
            result["description"] = "Ustawienie przypomnienia"
            result["esp32_handler"] = "reminder_handler"
            
        elif cmd_type == CommandType.WEATHER:
            result["description"] = "Sprawdzenie pogody"
            result["esp32_handler"] = "weather_handler"
            
        elif cmd_type == CommandType.EMAIL:
            result["description"] = "Operacja na email"
            result["esp32_handler"] = "email_handler"
        elif cmd_type == CommandType.MUSIC:
            result["description"] = "Kontrola odtwarzania muzyki"
            result["esp32_handler"] = "music_handler"
            
        user_message = self._generate_confirmation_message(cmd_type, params)
        result["user_message"] = user_message
            
        return result
    
    def _generate_confirmation_message(self, cmd_type: CommandType, params: Dict[str, Any]) -> str:
        """Generuje komunikat potwierdzający dla użytkownika."""
        if cmd_type == CommandType.CALENDAR:
            date = params.get('date', 'nieokreślonej dacie')
            time = params.get('time', 'nieokreślonym czasie')
            title = params.get('title', 'Nowe wydarzenie')
            
            return f"Wykryłam polecenie utworzenia wydarzenia '{title}' w kalendarzu na {date} o godzinie {time}."
            
        elif cmd_type == CommandType.REMINDER:
            date = params.get('date', 'dziś')
            time = params.get('time', 'wkrótce')
            title = params.get('title', 'przypomnienie')
            
            # Formatowanie daty
            if date == "jutro":
                date_text = "jutro"
            elif date == "pojutrze":
                date_text = "pojutrze"
            else:
                date_text = f"na {date}"
                
            return f"Wykryłam polecenie ustawienia przypomnienia '{title}' {date_text} o godzinie {time}."
            
        elif cmd_type == CommandType.WEATHER:
            location = params.get('location', 'Twojej lokalizacji')
            
            return f"Wykryłam polecenie sprawdzenia pogody w {location}."
            
        elif cmd_type == CommandType.EMAIL:
            return "Wykryłam polecenie związane z obsługą email."
        
        elif cmd_type == CommandType.MUSIC:
            action = params.get('action', '')
            
            if action == "pause":
                return "Wykryłam polecenie zatrzymania odtwarzania muzyki."
            
            elif action == "play":
                return "Wykryłam polecenie wznowienia odtwarzania muzyki."
            
            elif action == "next":
                return "Wykryłam polecenie przejścia do następnego utworu."
            
            elif action == "previous":
                return "Wykryłam polecenie przejścia do poprzedniego utworu."
            
            elif action == "volume_up":
                value = params.get('value', '')
                if value:
                    return f"Wykryłam polecenie zwiększenia głośności o {value}%."
                else:
                    return "Wykryłam polecenie zwiększenia głośności."
            
            elif action == "volume_down":
                value = params.get('value', '')
                if value:
                    return f"Wykryłam polecenie zmniejszenia głośności o {value}%."
                else:
                    return "Wykryłam polecenie zmniejszenia głośności."
            
            elif action == "volume_set":
                value = params.get('value', '50')
                return f"Wykryłam polecenie ustawienia głośności na {value}%."
            
            elif action == "search":
                search_type = params.get('search_type', 'track')
                query = params.get('query', '')
                
                if search_type == "track":
                    return f"Wykryłam polecenie wyszukania i odtworzenia utworu \"{query}\"."
                elif search_type == "artist":
                    return f"Wykryłam polecenie wyszukania i odtworzenia muzyki artysty \"{query}\"."
                elif search_type == "album":
                    return f"Wykryłam polecenie wyszukania i odtworzenia albumu \"{query}\"."
                elif search_type == "playlist":
                    return f"Wykryłam polecenie wyszukania i odtworzenia playlisty \"{query}\"."
                else:
                    return f"Wykryłam polecenie wyszukania muzyki \"{query}\"."
            
            return "Wykryłam polecenie kontroli muzyki."
        
        return "Wykryłam polecenie, ale nie rozpoznaję jego dokładnego typu."