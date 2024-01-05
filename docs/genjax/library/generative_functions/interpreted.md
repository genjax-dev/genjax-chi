::: genjax._src.generative_functions.interpreted
    options:
      show_root_heading: true

---

!!! note "Restrictions of the `Interpreted` language"

    The `Interpreted` language simultaneously very expressive (allowing users to utilize native Python control flow, and recursion). This language expressivity comes at a cost: `Interpreted` generative functions **cannot** be used within `Static` generative functions, or the existing set of combinators (which expect `JAXGenerativeFunction`, of which the `Interpreted` type is not).

    In addition, the `Interpreted` language does not support any AD interfaces.

    We therefore recommend this language for sketching ideas, and for pedagogy - but users are encouraged to migrate their code to the `Static` & combinators subset of GenJAX, to make use of advanced AD interfaces, parallel inference, and much better performance.

## Usage
