class GEXTrace:
    def __init__(self, gen_fn, jitted, args, retval, choices, score):
        self.args = args
        self.score = score
        self.choices = choices
        self.retval = retval
        self.gen_fn = gen_fn
        self.jitted = jitted

    def get_choices(self):
        return self.choices

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score
