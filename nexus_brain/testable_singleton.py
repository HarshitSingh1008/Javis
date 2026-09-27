"""
NEXUS AI v4.0 — Testable singleton pattern with reset capability.
Hardware: Intel i3 7th Gen · 12GB RAM · Ollama + Groq API

Provides a singleton pattern that can be reset for testing purposes,
unlike @lru_cache which is difficult to clear.
"""

import threading
from typing import TypeVar, Generic, Optional, Type, Callable, Any
from functools import wraps


T = TypeVar('T')


class SingletonMeta(type):
    """
    Metaclass for creating testable singletons.
    
    Features:
    - Thread-safe instantiation
    - Reset capability for testing
    - Explicit instance management
    - Supports multiple inheritance
    """
    
    _instances: dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()
    _init_args: dict[type, tuple] = {}
    _init_kwargs: dict[type, dict] = {}
    
    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                # Store init args for potential reset with same args
                cls._init_args[cls] = args
                cls._init_kwargs[cls] = kwargs
                cls._instances[cls] = super().__call__(*args, **kwargs)
            return cls._instances[cls]
    
    @classmethod
    def get_instance(cls, singleton_class: Type[T]) -> Optional[T]:
        """Get the singleton instance without creating it."""
        return cls._instances.get(singleton_class)
    
    @classmethod
    def set_instance(cls, singleton_class: Type[T], instance: T) -> None:
        """Set a specific instance (useful for testing with mocks)."""
        with cls._lock:
            cls._instances[singleton_class] = instance
    
    @classmethod
    def reset(cls, singleton_class: Type[T]) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instances.pop(singleton_class, None)
            cls._init_args.pop(singleton_class, None)
            cls._init_kwargs.pop(singleton_class, None)
    
    @classmethod
    def reset_all(cls) -> None:
        """Reset all singletons (for testing)."""
        with cls._lock:
            cls._instances.clear()
            cls._init_args.clear()
            cls._init_kwargs.clear()
    
    @classmethod
    def recreate(cls, singleton_class: Type[T], *args, **kwargs) -> T:
        """Force recreation with new arguments."""
        with cls._lock:
            cls._instances.pop(singleton_class, None)
            cls._init_args[singleton_class] = args
            cls._init_kwargs[singleton_class] = kwargs
            cls._instances[singleton_class] = super(SingletonMeta, cls).__call__(
                singleton_class, *args, **kwargs
            )
            return cls._instances[singleton_class]


class TestableSingletonMixin:
    """
    Mixin class for testable singletons.
    
    Usage:
        class MyService(TestableSingletonMixin, ServiceBase):
            def __init__(self):
                super().__init__("my_service")
        
        # Normal usage
        service = MyService()
        
        # Testing
        MyService.reset()
        mock = Mock()
        MyService.set_instance(MyService, mock)
    """
    pass


# For backward compatibility
TestableSingleton = TestableSingletonMixin


def singleton(cls: Type[T]) -> Type[T]:
    """
    Decorator to make a class a testable singleton.
    
    Usage:
        @singleton
        class MyService:
            def __init__(self):
                pass
        
        # Normal usage
        service = MyService()
        
        # Testing
        MyService.reset()
    """
    original_new = cls.__new__
    
    @wraps(cls)
    def __new__(cls, *args, **kwargs):
        return SingletonMeta.__call__(cls, *args, **kwargs)
    
    cls.__new__ = staticmethod(__new__)
    cls._singleton_meta = True
    return cls


def get_singleton(cls: Type[T]) -> T:
    """
    Get singleton instance, creating if necessary.
    
    Equivalent to cls() but more explicit.
    """
    return cls()


def reset_singleton(cls: Type[T]) -> None:
    """Reset a singleton for testing."""
    SingletonMeta.reset(cls)


def set_singleton_instance(cls: Type[T], instance: T) -> None:
    """Set a specific instance (for testing with mocks)."""
    SingletonMeta.set_instance(cls, instance)


# Global registry for manual singleton management
class SingletonRegistry:
    """Registry for managing singletons manually."""
    
    def __init__(self):
        self._instances: dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def register(self, name: str, instance: Any) -> None:
        """Register a singleton instance."""
        with self._lock:
            self._instances[name] = instance
    
    def get(self, name: str) -> Optional[Any]:
        """Get a singleton instance."""
        with self._lock:
            return self._instances.get(name)
    
    def unregister(self, name: str) -> bool:
        """Unregister a singleton."""
        with self._lock:
            if name in self._instances:
                del self._instances[name]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all singletons."""
        with self._lock:
            self._instances.clear()
    
    def get_all(self) -> dict[str, Any]:
        """Get all registered singletons."""
        with self._lock:
            return self._instances.copy()


# Global singleton registry
_global_registry = SingletonRegistry()


def register_singleton(name: str, instance: Any) -> None:
    """Register a singleton in the global registry."""
    _global_registry.register(name, instance)


def get_singleton(name: str) -> Optional[Any]:
    """Get a singleton from the global registry."""
    return _global_registry.get(name)


def unregister_singleton(name: str) -> bool:
    """Unregister a singleton from the global registry."""
    return _global_registry.unregister(name)


def clear_singletons() -> None:
    """Clear all singletons in the global registry."""
    _global_registry.clear()


# Context manager for test isolation
class isolated_singletons:
    """
    Context manager for isolating singleton state during tests.
    
    Usage:
        with isolated_singletons():
            # Singletons are reset on entry
            service = MyService()
            # Test...
        # Singletons restored on exit
    """
    
    def __init__(self, *classes: Type):
        self._classes = classes
        self._saved_instances = {}
    
    def __enter__(self):
        # Save current instances
        for cls in self._classes:
            self._saved_instances[cls] = SingletonMeta.get_instance(cls)
            SingletonMeta.reset(cls)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore saved instances
        for cls, instance in self._saved_instances.items():
            if instance is not None:
                SingletonMeta.set_instance(cls, instance)
            else:
                SingletonMeta.reset(cls)