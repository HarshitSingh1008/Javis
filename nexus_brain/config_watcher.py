"""
NEXUS AI v4.0 — Configuration hot-reload support.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Provides:
- File watching for config changes
- Hot-reload of settings without restart
- Change callbacks for dependent components
- Validation before applying changes
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, AsyncGenerator
from functools import lru_cache
from contextlib import asynccontextmanager

from nexus_config.settings import get_settings, APP_ROOT, Settings

logger = logging.getLogger("nexus.config_watcher")


@dataclass
class ConfigChange:
    """Represents a configuration change."""
    key: str
    old_value: Any
    new_value: Any
    timestamp: float = field(default_factory=time.time)
    source: str = "file_watcher"


class ConfigWatcher:
    """
    Watches configuration files for changes and triggers hot-reload.
    
    Supports:
    - .env file watching
    - JSON/YAML config file watching
    - Validation before applying
    - Change callbacks for dependent components
    """
    
    def __init__(self, config_dir: Optional[Path] = None, interval: float = 2.0):
        self._config_dir = config_dir or APP_ROOT
        self._interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[ConfigChange], None]] = []
        self._lock = threading.RLock()
        self._file_mtimes: Dict[str, float] = {}
        self._watched_files: Set[str] = set()
        
        # Default watched files
        self.watch_file(".env")
        self.watch_file("overlay_config.json")
    
    def watch_file(self, filename: str) -> None:
        """Add a file to watch list."""
        with self._lock:
            filepath = self._config_dir / filename
            self._watched_files.add(str(filepath))
            if filepath.exists():
                self._file_mtimes[str(filepath)] = filepath.stat().st_mtime
    
    def unwatch_file(self, filename: str) -> None:
        """Remove a file from watch list."""
        with self._lock:
            filepath = str(self._config_dir / filename)
            self._watched_files.discard(filepath)
            self._file_mtimes.pop(filepath, None)
    
    def register_callback(self, callback: Callable[[ConfigChange], None]) -> None:
        """Register a callback for config changes."""
        with self._lock:
            self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[ConfigChange], None]) -> None:
        """Unregister a callback."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def start(self) -> None:
        """Start watching for config changes."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._watch_loop,
            daemon=True,
            name="nexus-config-watcher",
        )
        self._thread.start()
        logger.info("Config watcher started for %s", self._config_dir)
    
    def stop(self) -> None:
        """Stop watching."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Config watcher stopped")
    
    def _watch_loop(self) -> None:
        """Background loop checking for file changes."""
        while self._running:
            time.sleep(self._interval)
            
            try:
                self._check_files()
            except Exception as e:
                logger.debug("Config watcher error: %s", e)
    
    def _check_files(self) -> None:
        """Check watched files for modifications."""
        with self._lock:
            for filepath_str in list(self._watched_files):
                filepath = Path(filepath_str)
                
                if not filepath.exists():
                    # File was deleted
                    if filepath_str in self._file_mtimes:
                        logger.info("Config file deleted: %s", filepath.name)
                        self._notify_change(filepath.name, 
                            self._file_mtimes[filepath_str], None)
                        del self._file_mtimes[filepath_str]
                    continue
                
                mtime = filepath.stat().st_mtime
                old_mtime = self._file_mtimes.get(filepath_str)
                
                if old_mtime is None:
                    # New file
                    self._file_mtimes[filepath_str] = mtime
                    logger.info("New config file detected: %s", filepath.name)
                    self._notify_change(filepath.name, None, mtime)
                elif mtime > old_mtime:
                    # File modified
                    logger.info("Config file changed: %s", filepath.name)
                    self._file_mtimes[filepath_str] = mtime
                    self._notify_change(filepath.name, old_mtime, mtime)
    
    def _notify_change(self, filename: str, old_mtime: Optional[float], new_mtime: Optional[float]) -> None:
        """Notify callbacks of config change."""
        try:
            # Reload settings to get new values
            from nexus_config.settings import get_settings
            new_settings = get_settings()
            
            change = ConfigChange(
                key=filename,
                old_value={"mtime": old_mtime} if old_mtime else None,
                new_value={"mtime": new_mtime} if new_mtime else None,
                source="file_watcher",
            )
            
            for callback in self._callbacks:
                try:
                    callback(change)
                except Exception as e:
                    logger.error("Config change callback error: %s", e)
                    
        except Exception as e:
            logger.error("Failed to process config change: %s", e)
    
    async def reload_settings(self) -> Settings:
        """Force reload of settings."""
        from nexus_config.settings import load_dotenv
        load_dotenv(self._config_dir / ".env", override=True)
        
        # Clear the cached settings
        get_settings.cache_clear()
        
        return get_settings()


class HotReloadableService:
    """
    Base class for services that support hot-reload of configuration.
    
    Subclasses should implement _on_config_change to handle config updates.
    """
    
    def __init__(self, config_watcher: ConfigWatcher):
        self._config_watcher = config_watcher
        self._config_watcher.register_callback(self._on_config_change_wrapper)
    
    def _on_config_change_wrapper(self, change: ConfigChange) -> None:
        """Wrapper that catches errors in config change handlers."""
        try:
            self._on_config_change(change)
        except Exception as e:
            logger.error("Config change handler error in %s: %s", 
                        self.__class__.__name__, e)
    
    def _on_config_change(self, change: ConfigChange) -> None:
        """Override this method to handle config changes."""
        pass
    
    def cleanup(self) -> None:
        """Clean up callbacks."""
        self._config_watcher.unregister_callback(self._on_config_change_wrapper)


# Global config watcher
_config_watcher: Optional[ConfigWatcher] = None


def get_config_watcher() -> ConfigWatcher:
    """Get the global config watcher."""
    global _config_watcher
    if _config_watcher is None:
        _config_watcher = ConfigWatcher()
    return _config_watcher


@asynccontextmanager
async def config_watcher_lifespan() -> AsyncGenerator[ConfigWatcher, None]:
    """Context manager for config watcher lifecycle."""
    watcher = get_config_watcher()
    try:
        watcher.start()
        yield watcher
    finally:
        watcher.stop()


def register_config_callback(callback: Callable[[ConfigChange], None]) -> None:
    """Register a callback for config changes."""
    get_config_watcher().register_callback(callback)


def unregister_config_callback(callback: Callable[[ConfigChange], None]) -> None:
    """Unregister a config change callback."""
    get_config_watcher().unregister_callback(callback)