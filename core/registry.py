"""Component registry for tasks, adapters, parsers, and metrics."""

from typing import Any, Callable, Dict, Type, TypeVar, Optional
import inspect

T = TypeVar('T')


class Registry:
    """
    Generic registry for components.
    
    Usage:
        # Register a component
        @registry.register("parser", "code")
        class CodeParser(Parser):
            pass
        
        # Get a component
        parser_class = registry.get("parser", "code")
        parser = parser_class()
    """
    
    def __init__(self):
        self._components: Dict[str, Dict[str, Type]] = {
            "task": {},
            "adapter": {},
            "parser": {},
            "dataset_loader": {},
            "metric": {},
            "evaluator": {},
        }
    
    def register(self, component_type: str, name: str) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a component.
        
        Args:
            component_type: Type of component (task, adapter, parser, etc.)
            name: Unique name for the component
        """
        def decorator(cls: Type[T]) -> Type[T]:
            if component_type not in self._components:
                self._components[component_type] = {}
            
            if name in self._components[component_type]:
                raise ValueError(
                    f"Component '{name}' already registered for type '{component_type}'"
                )
            
            self._components[component_type][name] = cls
            return cls
        
        return decorator
    
    def get(self, component_type: str, name: str) -> Optional[Type]:
        """
        Get a registered component class.
        
        Args:
            component_type: Type of component
            name: Name of the component
        
        Returns:
            The component class or None if not found
        """
        return self._components.get(component_type, {}).get(name)
    
    def create(
        self,
        component_type: str,
        name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Create an instance of a registered component.
        
        Args:
            component_type: Type of component
            name: Name of the component
            *args, **kwargs: Arguments to pass to component constructor
        
        Returns:
            Instance of the component
        """
        cls = self.get(component_type, name)
        if cls is None:
            raise ValueError(
                f"Unknown {component_type}: '{name}'. "
                f"Available: {list(self._components.get(component_type, {}).keys())}"
            )
        return cls(*args, **kwargs)
    
    def list_components(self, component_type: Optional[str] = None) -> Dict[str, list]:
        """
        List all registered components.
        
        Args:
            component_type: If specified, only list components of this type
        
        Returns:
            Dictionary mapping component types to lists of names
        """
        if component_type:
            return {component_type: list(self._components.get(component_type, {}).keys())}
        
        return {
            ctype: list(components.keys())
            for ctype, components in self._components.items()
        }
    
    def is_registered(self, component_type: str, name: str) -> bool:
        """Check if a component is registered."""
        return name in self._components.get(component_type, {})
    
    def unregister(self, component_type: str, name: str) -> bool:
        """
        Unregister a component.
        
        Returns:
            True if component was found and removed, False otherwise
        """
        if component_type in self._components and name in self._components[component_type]:
            del self._components[component_type][name]
            return True
        return False
    
    def get_component_info(self, component_type: str, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered component.
        
        Returns:
            Dictionary with component information or None if not found
        """
        cls = self.get(component_type, name)
        if cls is None:
            return None
        
        return {
            "name": name,
            "type": component_type,
            "class": cls.__name__,
            "module": cls.__module__,
            "docstring": inspect.getdoc(cls),
        }


# Global registry instance
registry = Registry()

# Convenience functions
def register_task(name: str):
    """Decorator to register a task."""
    return registry.register("task", name)

def register_adapter(name: str):
    """Decorator to register a model adapter."""
    return registry.register("adapter", name)

def register_parser(name: str):
    """Decorator to register a parser."""
    return registry.register("parser", name)

def register_dataset_loader(name: str):
    """Decorator to register a dataset loader."""
    return registry.register("dataset_loader", name)

def register_metric(name: str):
    """Decorator to register a metric."""
    return registry.register("metric", name)

def register_evaluator(name: str):
    """Decorator to register an evaluator."""
    return registry.register("evaluator", name)

# Factory functions
def get_adapter(name: str) -> Optional[Type]:
    """Get an adapter class by name."""
    return registry.get("adapter", name)

def get_parser(name: str) -> Optional[Type]:
    """Get a parser class by name."""
    return registry.get("parser", name)

def get_dataset_loader(name: str) -> Optional[Type]:
    """Get a dataset loader class by name."""
    return registry.get("dataset_loader", name)

def get_metric(name: str) -> Optional[Type]:
    """Get a metric class by name."""
    return registry.get("metric", name)

def create_adapter(name: str, *args, **kwargs) -> Any:
    """Create an adapter instance."""
    return registry.create("adapter", name, *args, **kwargs)

def create_parser(name: str, *args, **kwargs) -> Any:
    """Create a parser instance."""
    return registry.create("parser", name, *args, **kwargs)

def create_dataset_loader(name: str, *args, **kwargs) -> Any:
    """Create a dataset loader instance."""
    return registry.create("dataset_loader", name, *args, **kwargs)

def create_metric(name: str, *args, **kwargs) -> Any:
    """Create a metric instance."""
    return registry.create("metric", name, *args, **kwargs)
