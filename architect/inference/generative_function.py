"""
Define types for a restricted class of probabilistic programs.

A probabilistic program is one that simulates the output of some stochastic process by
making random choices from some prior distribution. Here, we consider a restricted class
of programs that have bounded support --- i.e. they rely on a known, fixed number of
random choices. This results in a probabilistic program that generates a fixed-length
trace of its execution.

A probabilistic program is represented by a function that, in addition to returning
some output, also returns a trace (a vector of all the random choices made during
execution) and un-normalized log likelihood. The function should also take (in addition
to its regular arguments), two arrays indicating any pre-defined random choices (the
first vector should be 1 for any random variables with pre-defined choices, and the
second vector should include the specified value for that random variable).
"""
from dataclasses import dataclass, field
from typing import Dict, List

import jax.numpy as jnp

######################################################################################
# The function will need to keep track of the choices it makes using a choice database
######################################################################################
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
        self, name: ChoiceName, value: jnp.ndarray, trace: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Add the given choice to the trace.

        args:
            name: the name of the choice to save
            value: the value to save
            trace: the current trace
        returns:
            the new trace
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

        # Try to save each name/value, and raise an error if one is not found
        try:
            for name, value_i in zip(names, value):
                choice_index = self.lookup_table[name]
                trace = trace.at[choice_index].set(value_i)
        except KeyError:
            raise RuntimeError(f"{name} not found in database; did you register it?")

        # Return the new trace
        return trace
