class TestStagedChoose:
    pass


#     def test_static_integer_index(self):
#         result = staged_choose(1, [10, 20, 30])
#         assert result == 20

#     def test_jax_array_index(self):
#         result = staged_choose(jnp.array(2), [10, 20, 30])
#         assert jnp.array_equal(result, jnp.array(30))

#     def test_diff_index(self):

#         diff = Diff(jnp.array(0), jnp.array(0))
#         result = staged_choose(diff, [10, 20, 30])
#         assert jnp.array_equal(result, jnp.array(10))

#     def test_heterogeneous_types(self):
#         result = staged_choose(2, [True, 2, False])
#         assert result == 0

#     def test_wrap_mode(self):
#         result = staged_choose(jnp.array(3), [10, 20, 30])
#         assert jnp.array_equal(result, jnp.array(10))
