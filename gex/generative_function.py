class GEXTrace:
    def __init__(self, gen_fn, jitted, args, retval, choices, score):
        self.gen_fn = gen_fn
        self.jitted = jitted
        self.args = args
        self.retval = retval
        self.choices = choices
        self.score = score

    def get_choices(self):
        return self.choices

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score
