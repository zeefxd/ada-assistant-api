import logging
import requests
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spotify_handler")

SPOTIFY_API_BASE_URL = "https://api.spotify.com/v1"

def debug_token(token):
    """Debug helper to safely log token information"""
    if not token:
        return "None"
    if len(token) <= 10:
        return token
    return f"{token[:5]}...{token[-5:]}"

class SpotifyHandler:
    """
    Handler for performing operations on the Spotify API.
    """
    
    def __init__(self, access_token: Optional[str] = None):
        self.access_token = access_token
        logger.info(f"SpotifyHandler initialized with token: {debug_token(access_token)}")
    
    async def execute_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a Spotify command based on parameters.
        
        Args:
            params (Dict[str, Any]): Command parameters
            
        Returns:
            Dict[str, Any]: Result of command execution
        """
        if not self.access_token:
            return {
                "success": False, 
                "message": "Brak tokenu Spotify. Połącz swoje konto Spotify w aplikacji."
            }
        
        action = params.get("action")
        if not action:
            return {"success": False, "message": "Brak określonej akcji"}
        
        try:
            if action == "play" and "query" in params:
                return await self.play_track(params["query"])
            elif action == "pause":
                return await self.pause_playback()
            elif action == "resume":
                return await self.resume_playback()
            elif action == "next":
                return await self.skip_to_next()
            elif action == "previous":
                return await self.skip_to_previous()
            elif action == "current_track":
                return await self.get_current_track()
            elif action == "volume_up" or action == "volume_down":
                value = params.get("value", 10)
                value = value if action == "volume_up" else -value
                return await self.adjust_volume(value)
            elif action == "volume_set":
                value = params.get("value", 50)
                return await self.set_volume(value)
            else:
                return {"success": False, "message": f"Nieznana akcja: {action}"}
        except Exception as e:
            logger.error(f"Błąd podczas wykonywania polecenia Spotify: {e}")
            return {"success": False, "message": f"Błąd: {str(e)}"}
    
    async def play_track(self, query: str) -> Dict[str, Any]:
        """
        Plays a track based on the search query.
        
        Args:
            query (str): Search query (artist, title, etc.)
            
        Returns:
            Dict[str, Any]: Result of command execution
        """
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            logger.info(f"Wyszukiwanie utworu: {query}")
            search_response = requests.get(
                f"{SPOTIFY_API_BASE_URL}/search",
                headers=headers,
                params={"q": query, "type": "track", "limit": 5}
            )
            
            if search_response.status_code != 200:
                logger.error(f"Błąd wyszukiwania: {search_response.status_code} - {search_response.text}")
                return {"success": False, "message": "Nie znaleziono utworu"}
            
            search_data = search_response.json()
            
            if not search_data["tracks"]["items"]:
                return {"success": False, "message": f"Nie znaleziono utworu: {query}"}
            
            logger.info("Znalezione utwory:")
            for i, track in enumerate(search_data["tracks"]["items"][:5]):
                logger.info(f"{i+1}. {track['name']} - {track['artists'][0]['name']}")
            
            track = search_data["tracks"]["items"][0]
            track_uri = track["uri"]
            track_name = track["name"]
            artist_name = track["artists"][0]["name"]
            
            logger.info(f"Wybrany utwór do odtworzenia: {track_name} - {artist_name}")
            
            play_response = requests.put(
                f"{SPOTIFY_API_BASE_URL}/me/player/play",
                headers=headers,
                json={"uris": [track_uri]}
            )
            
            if play_response.status_code in [200, 204]:
                return {
                    "success": True,
                    "message": f"Odtwarzam {track_name} wykonawcy {artist_name}",
                    "track": {"name": track_name, "artist": artist_name}
                }
            elif play_response.status_code == 404:
                return {
                    "success": False, 
                    "message": "Nie znaleziono aktywnego urządzenia Spotify. Otwórz aplikację Spotify na swoim urządzeniu."
                }
            else:
                logger.error(f"Błąd odtwarzania: {play_response.status_code} - {play_response.text}")
                return {"success": False, "message": "Nie udało się odtworzyć utworu"}
                
        except Exception as e:
            logger.error(f"Wyjątek podczas odtwarzania utworu: {e}")
            return {"success": False, "message": f"Błąd: {str(e)}"}
    
    async def pause_playback(self) -> Dict[str, Any]:
        """
        Pauses Spotify playback.
        
        Returns:
            Dict[str, Any]: Result of command execution
        """
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            response = requests.put(
                f"{SPOTIFY_API_BASE_URL}/me/player/pause",
                headers=headers
            )
            
            if response.status_code in [200, 204]:
                return {"success": True, "message": "Wstrzymano odtwarzanie"}
            elif response.status_code == 404:
                return {"success": False, "message": "Nie znaleziono aktywnego odtwarzania"}
            else:
                logger.error(f"Błąd pauzowania: {response.status_code} - {response.text}")
                return {"success": False, "message": "Nie udało się wstrzymać odtwarzania"}
        
        except Exception as e:
            logger.error(f"Wyjątek podczas pauzowania: {e}")
            return {"success": False, "message": f"Błąd: {str(e)}"}
    
    async def resume_playback(self) -> Dict[str, Any]:
        """
        Resumes Spotify playback.
        
        Returns:
            Dict[str, Any]: Result of command execution
        """
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            response = requests.put(
                f"{SPOTIFY_API_BASE_URL}/me/player/play",
                headers=headers
            )
            
            if response.status_code in [200, 204]:
                return {"success": True, "message": "Wznowiono odtwarzanie"}
            elif response.status_code == 404:
                return {
                    "success": False, 
                    "message": "Nie znaleziono aktywnego urządzenia Spotify. Otwórz aplikację Spotify na swoim urządzeniu."
                }
            else:
                logger.error(f"Błąd wznawiania: {response.status_code} - {response.text}")
                return {"success": False, "message": "Nie udało się wznowić odtwarzania"}
        
        except Exception as e:
            logger.error(f"Wyjątek podczas wznawiania: {e}")
            return {"success": False, "message": f"Błąd: {str(e)}"}
    
    async def skip_to_next(self) -> Dict[str, Any]:
        """
        Skips to the next track.
        
        Returns:
            Dict[str, Any]: Result of command execution
        """
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            response = requests.post(
                f"{SPOTIFY_API_BASE_URL}/me/player/next",
                headers=headers
            )
            
            if response.status_code in [200, 204]:
                return {"success": True, "message": "Przechodzę do następnego utworu"}
            elif response.status_code == 404:
                return {
                    "success": False, 
                    "message": "Nie znaleziono aktywnego urządzenia Spotify. Otwórz aplikację Spotify na swoim urządzeniu."
                }
            else:
                logger.error(f"Błąd przejścia do następnego: {response.status_code} - {response.text}")
                return {"success": False, "message": "Nie udało się przejść do następnego utworu"}
        
        except Exception as e:
            logger.error(f"Wyjątek podczas przejścia do następnego: {e}")
            return {"success": False, "message": f"Błąd: {str(e)}"}
    
    async def skip_to_previous(self) -> Dict[str, Any]:
        """
        Skips to the previous track.
        
        Returns:
            Dict[str, Any]: Result of command execution
        """
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            response = requests.post(
                f"{SPOTIFY_API_BASE_URL}/me/player/previous",
                headers=headers
            )
            
            if response.status_code in [200, 204]:
                return {"success": True, "message": "Przechodzę do poprzedniego utworu"}
            elif response.status_code == 404:
                return {
                    "success": False, 
                    "message": "Nie znaleziono aktywnego urządzenia Spotify. Otwórz aplikację Spotify na swoim urządzeniu."
                }
            else:
                logger.error(f"Błąd przejścia do poprzedniego: {response.status_code} - {response.text}")
                return {"success": False, "message": "Nie udało się przejść do poprzedniego utworu"}
        
        except Exception as e:
            logger.error(f"Wyjątek podczas przejścia do poprzedniego: {e}")
            return {"success": False, "message": f"Błąd: {str(e)}"}
    
    async def get_current_volume(self) -> Optional[int]:
        """
        Gets the current volume level.
        
        Returns:
            Optional[int]: Current volume level or None if error occurs
        """
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            response = requests.get(
                f"{SPOTIFY_API_BASE_URL}/me/player",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("device", {}).get("volume_percent")
            
            return None
        
        except Exception as e:
            logger.error(f"Wyjątek podczas pobierania głośności: {e}")
            return None
    
    async def adjust_volume(self, change: int) -> Dict[str, Any]:
        """
        Adjusts the volume by a relative value.
        
        Args:
            change (int): Relative volume change
            
        Returns:
            Dict[str, Any]: Result of command execution
        """
        current_volume = await self.get_current_volume()
        
        if current_volume is None:
            current_volume = 50  # Default value if current volume cannot be retrieved
        
        new_volume = max(0, min(100, current_volume + change))
        return await self.set_volume(new_volume)
    
    async def set_volume(self, volume: int) -> Dict[str, Any]:
        """
        Sets the volume to a specific level.
        
        Args:
            volume (int): Volume level (0-100)
            
        Returns:
            Dict[str, Any]: Result of command execution
        """
        volume = max(0, min(100, volume))
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            response = requests.put(
                f"{SPOTIFY_API_BASE_URL}/me/player/volume",
                headers=headers,
                params={"volume_percent": volume}
            )
            
            if response.status_code in [200, 204]:
                return {"success": True, "message": f"Głośność ustawiona na {volume}%"}
            elif response.status_code == 404:
                return {
                    "success": False, 
                    "message": "Nie znaleziono aktywnego urządzenia Spotify. Otwórz aplikację Spotify na swoim urządzeniu."
                }
            else:
                logger.error(f"Błąd ustawiania głośności: {response.status_code} - {response.text}")
                return {"success": False, "message": "Nie udało się zmienić głośności"}
        
        except Exception as e:
            logger.error(f"Wyjątek podczas ustawiania głośności: {e}")
            return {"success": False, "message": f"Błąd: {str(e)}"}
        
    async def get_current_track(self) -> Dict[str, Any]:
        """
        Gets information about the currently playing track.
        
        Returns:
            Dict[str, Any]: Result with current track information
        """
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            response = requests.get(
                f"{SPOTIFY_API_BASE_URL}/me/player/currently-playing",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data and data.get("item"):
                    track = data["item"]
                    track_name = track.get("name", "Nieznany utwór")
                    artist_name = track.get("artists", [{}])[0].get("name", "Nieznany wykonawca")
                    is_playing = data.get("is_playing", False)
                    
                    status = "odtwarzany" if is_playing else "wstrzymany"
                    message = f"Aktualnie {status}: {track_name} wykonawcy {artist_name}"
                    
                    return {
                        "success": True,
                        "message": message,
                        "track": {
                            "name": track_name,
                            "artist": artist_name,
                            "is_playing": is_playing
                        }
                    }
                else:
                    return {"success": False, "message": "Nic nie jest obecnie odtwarzane"}
                    
            elif response.status_code == 204:
                return {"success": False, "message": "Nic nie jest obecnie odtwarzane"}
            elif response.status_code == 404:
                return {
                    "success": False, 
                    "message": "Nie znaleziono aktywnego urządzenia Spotify. Otwórz aplikację Spotify na swoim urządzeniu."
                }
            else:
                logger.error(f"Błąd pobierania aktualnego utworu: {response.status_code} - {response.text}")
                return {"success": False, "message": "Nie udało się pobrać informacji o aktualnym utworze"}
        
        except Exception as e:
            logger.error(f"Wyjątek podczas pobierania aktualnego utworu: {e}")
            return {"success": False, "message": f"Błąd: {str(e)}"}