# __init__.py

import importlib.util
import os
import sys

# Save the original sys.path
original_sys_path = sys.path.copy()

# Get the directory of the current file
current_directory = os.path.dirname(__file__)
print(f"Current directory: {current_directory}")

def print_module_location(module_name):
    """Print the location of the specified module, if available."""
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        if spec.origin:
            print(f"Module '{module_name}' found at: {spec.origin}")
        else:
            print(f"Module '{module_name}' found, but its origin could not be determined.")
    else:
        print(f"Module '{module_name}' could not be found.")

try:
    # Ensure the module is in the path
    sys.path.append(current_directory)

    # Import and expose classes and functions from distribution_embedding
    from distribution_embedding import *

    # Optionally, print the location of the module for debugging purposes
    print_module_location('distribution_embedding')

except ImportError:
    print(
        "The 'distribution_embedding' shared object could not be loaded. "
        "Please compile the C++ bindings by running 'poetry run compile_bindings'."
    )

finally:
    # Restore the original sys.path
    sys.path = original_sys_path

# Expose specific classes and functions
__all__ = [
    'DistributionEmbedding',  # Classes and functions you want to expose
    'other_function',
    'SomeClass'
]
