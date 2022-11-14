"""
Define a database for tracking the choices made by a generative function in the trace
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import jax.numpy as jnp

# Convenience types
ChoiceName = str
ChoiceIndex = int


@dataclass
class ChoiceDatabase:
    max_choices: int
    lookup_table: Dict[ChoiceName, ChoiceIndex] = field(default_factory=dict)
    next_index: ChoiceIndex = 0

    def register_scalar_choice(self, name: ChoiceName) -> ChoiceIndex:
        """
        Adds a scalar choice to the database and assigns it a unique index in the trace.

        args:
            name: the name of the choice to register
        returns:
            the index of that choice in the trace
        raises:
            RuntimeError if max_choices exceeded
        """
        # We cannot register more than the pre-specified fixed number of choices
        if self.next_index >= self.max_choices:
            raise RuntimeError(
                (
                    f"Cannnot add additional choice {name}; "
                    f"{self.max_choices} choices have already been registered!"
                )
            )

        # If we do have room left, use the next available index
        self.lookup_table[name] = self.next_index
        self.next_index += 1

        # Return the index we used
        return self.lookup_table[name]

    def register_vector_choice(self, name: ChoiceName, size: int) -> List[ChoiceIndex]:
        """
        Adds a vector choice to the database, giving each element a unique trace index.

        args:
            name: the name of the choice to register
            size: the size of the choice vector
        returns:
            the list of indices of the choice in the trace
        raises:
            RuntimeError if max_choices exceeded
        """
        # We cannot register more than the pre-specified fixed number of choices
        if self.next_index + size > self.max_choices:
            raise RuntimeError(
                (
                    f"Cannnot add additional choice {name} with {size} entries; "
                    f"no more than {self.max_choices} choices may be registered!"
                )
            )

        # Register each entry
        choice_indices = [
            self.register_scalar_choice(f"{name}[{i}]") for i in range(size)
        ]

        return choice_indices

    def add_choice_to_trace(
        self,
        name: ChoiceName,
        value: jnp.ndarray,
        trace: jnp.ndarray,
        selection: jnp.ndarray,
        predetermined_values: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Add the given choice to the trace.

        args:
            name: the name of the choice to save
            value: the value to save
            trace: the current trace
            selections: a vector of the same length as the trace that is 1 where
                a value has been pre-specified for that choice and 0 elsewhere.
            predetermined_values: a vector of the same length as the trace containing
                the values of any pre-specified choices.
        returns:
            - the value assigned (may differ if the selection vector assigns a pre-
                specified value to a choice)
            - the new trace
        raises:
            RuntimeError if the given choice was not previously registered or if it
            has more than 1 axis
        """
        # Determine if the choice is a scalar or vector
        dims = value.ndim
        if dims == 0:
            names = [name]
            value = value.reshape(-1)
        elif dims == 1:
            names = [f"{name}[{i}]" for i in range(value.size)]
        else:
            raise RuntimeError(
                f"Can only add vector or scalar choices to the trace (got {dims} dims)."
            )

        # Try to get the trace index for each value, and raise an error if is not found
        try:
            choice_indices = jnp.array([self.lookup_table[name] for name in names])
        except KeyError:
            raise RuntimeError(f"{name} not found in database; did you register it?")

        # Save either the provided value or the pre-specified value, depending on what
        # was selected
        values_to_save = jnp.where(
            selection[choice_indices], predetermined_values[choice_indices], value
        )
        trace = trace.at[choice_indices].set(values_to_save)

        # Return the values that were saved and the new trace
        return values_to_save, trace
