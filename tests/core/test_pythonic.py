import genjax
import jax.numpy as jnp
import penzai.pz as pz
from genjax.typing import FloatArray


class TestPythonic:
    def test_pythonic(self):
        @pz.pytree_dataclass
        class Foo(genjax.PythonicPytree):
            x: FloatArray
            y: FloatArray

        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])

        f = Foo(x, y)

        assert f[1] == Foo(x[1], y[1])

        assert jnp.all(f[:2].x == x[:2])
        assert jnp.all(f[:2].y == y[:2])

        assert jnp.all(f[2:].x == x[2:])
        assert jnp.all(f[2:].y == y[2:])

        assert jnp.all(f[1::4].x == x[1::4])
        assert jnp.all(f[1::4].y == y[1::4])

        assert len(f) == x.shape[0]

        fi = iter(f)
        assert next(fi) == Foo(x[0], y[0])
        assert next(fi) == Foo(x[1], y[1])

        ff = f + f

        assert len(ff) == 2 * len(f)
        assert jnp.allclose(ff.x, jnp.concatenate((x, x)))

        p = Foo(jnp.array(-1.0), jnp.array(-10.0))
        fp = f.prepend(p)

        assert len(fp) == 1 + len(f)
        assert jnp.all(fp[0].x == p.x)
        assert jnp.all(fp[0].y == p.y)
        assert fp[0] == p
