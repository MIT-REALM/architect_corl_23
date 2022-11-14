import jax.numpy as jnp
import pytest

from architect.inference import ChoiceDatabase


def test_choice_database():
    """Test the choice database"""
    # Make a choice database to test
    max_choices = 4
    db = ChoiceDatabase(max_choices)

    # There should be no choices in the database yet
    assert len(db.lookup_table.keys()) == 0

    # If we add a scalar choice, that should work fine
    index = db.register_scalar_choice("first_choice")
    assert index is not None and index < max_choices

    # If we add a scalar choice, that should also work fine
    size = 3
    indices = db.register_vector_choice("second_choice", size)
    assert len(indices) == size

    # The database is now full, so trying to register additional choices should fail
    with pytest.raises(RuntimeError):
        db.register_scalar_choice("bad_choice")

    with pytest.raises(RuntimeError):
        db.register_vector_choice("bad_choice", size)

    # We can now save values for the registered values in a trace, without any
    # predetermined values
    trace = jnp.zeros((max_choices,))
    selection = jnp.zeros((max_choices,))
    predetermined_values = jnp.zeros((max_choices,))
    _, trace = db.add_choice_to_trace(
        "first_choice", jnp.array(1.0), trace, selection, predetermined_values
    )
    _, trace = db.add_choice_to_trace(
        "second_choice",
        jnp.array([2.0, 3.0, 4.0]),
        trace,
        selection,
        predetermined_values,
    )

    # The trace should now contain those values
    assert jnp.allclose(trace, jnp.array([1.0, 2.0, 3.0, 4.0]))

    # Test setting a choice with a predetermined value for the last 2 elements
    selection = jnp.array([0.0, 0.0, 1.0, 1.0])
    _, trace = db.add_choice_to_trace(
        "second_choice",
        jnp.array([2.0, 3.0, 4.0]),
        trace,
        selection,
        jnp.array([10, 20, 30, 40]),
    )
    assert jnp.allclose(trace, jnp.array([1.0, 2.0, 30, 40]))

    # If we try to save a choice that has not been registered, that should give an error
    with pytest.raises(RuntimeError):
        db.add_choice_to_trace(
            "bad_choice", jnp.array(1.0), trace, selection, predetermined_values
        )
