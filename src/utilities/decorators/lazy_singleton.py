"""
Decorator for creating a lazy-instantiated singleton.

This implements the common singleton design pattern in a way that delays
the creation of a class instance until it is first accessed. After the 
initial instantiation, future references return the same instance.
"""
def lazy_singleton(cls):
    instance = None

    def get_instance_callback(*args, **kwargs):
        nonlocal instance #reference from outer scope

        if instance is None:
            instance = cls(*args, *kwargs)
        return instance
    
    return get_instance_callback