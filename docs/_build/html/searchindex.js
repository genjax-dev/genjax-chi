Search.setIndex({"docnames": ["genjax/c_interface", "genjax/combinators", "genjax/gen_fn", "genjax/interface", "genjax/wasm_interface", "index"], "filenames": ["genjax/c_interface.rst", "genjax/combinators.rst", "genjax/gen_fn.rst", "genjax/interface.rst", "genjax/wasm_interface.rst", "index.rst"], "titles": ["Exposing C++ generative functions", "Generative function combinators", "What is a generative function?", "Generative function interface", "From GenJAX to WebAssembly", "Index"], "terms": {"while": 0, "genjax": [0, 1, 3, 5], "i": [0, 1, 3, 4, 5], "fast": 0, "gpu": 0, "courtesi": 0, "jax": [0, 1, 2, 3, 4, 5], "sometim": 0, "we": [0, 2], "want": 0, "lower": [0, 2], "level": 0, "control": [0, 1], "over": [0, 1], "implement": [0, 1, 2, 3], "our": [0, 2], "emit": 0, "code": [0, 2, 4], "which": [0, 1, 2, 3], "optim": 0, "cpu": 0, "devic": 0, "so": 0, "wish": 0, "leverag": 0, "modern": 0, "compil": [0, 5], "toolchain": 0, "like": [0, 5], "llvm": 0, "fortun": 0, "can": [0, 1, 2], "written": 0, "util": [0, 2], "an": [0, 1, 2, 3], "xla": 0, "primit": [0, 2], "call": [0, 1, 2, 3], "custom_cal": 0, "In": [0, 2, 4], "thi": [0, 1, 2, 3, 4], "note": [0, 4], "ll": [0, 4], "walk": [0, 4], "through": [0, 4], "process": 0, "us": [0, 1, 2], "gentl": 0, "base": [0, 1, 5], "excel": 0, "dan": 0, "foreman": 0, "mackei": 0, "To": 0, "need": [0, 1], "setup": 0, "pybind11": 0, "built": [0, 4], "cmake": 0, "ve": [0, 2], "done": 0, "directori": 0, "ani": [0, 3], "support": [0, 2, 3, 4], "infrastructur": 0, "includ": [0, 2], "header": 0, "onli": 0, "librari": 0, "provid": [0, 1, 2], "tool": 0, "defin": [0, 2, 3], "gen_fn": [0, 2, 3], "h": [0, 2], "": [0, 1, 2], "sketch": 0, "lib": 0, "cpp": 0, "templat": 0, "method": [0, 2, 3], "A": [1, 2, 3, 5], "object": [1, 2], "accept": [1, 2, 3], "1": [1, 2], "more": 1, "return": [1, 2, 3], "new": [1, 2, 5], "often": 1, "chang": 1, "structur": [1, 2], "intern": 1, "choic": [1, 3], "express": [1, 2], "pattern": [1, 2], "comput": [1, 2, 3], "modul": [1, 3, 4, 5], "allow": [1, 4], "branch": 1, "flow": 1, "differ": 1, "shape": [1, 2], "It": 1, "encod": [1, 2], "trace": [1, 2, 3], "sum": 1, "type": [1, 3], "bypass": 1, "restrict": 1, "from": [1, 2, 5], "lax": 1, "cond": 1, "ar": [1, 2, 3], "pass": 1, "switchcombin": 1, "must": 1, "same": 1, "argument": [1, 2, 3], "valu": [1, 3], "The": [1, 3], "have": 1, "dtype": [1, 2, 3], "class": 1, "sourc": [1, 2, 3], "broadcast": 1, "vectori": 1, "version": 1, "mapcombin": 1, "kernel": 1, "core": [1, 3], "datatyp": [1, 3], "generativefunct": [1, 3], "partial": 1, "func": 1, "arg": [1, 3], "keyword": [1, 2], "applic": 1, "given": [1, 3], "tupl": [1, 3], "futur": 1, "dictionari": 1, "static": [1, 2], "unrol": 1, "act": 1, "previou": 1, "output": 1, "input": 1, "unfoldcombin": 1, "length": 1, "int": 1, "concis": 2, "set": [1, 2, 3], "interfac": [1, 2, 5], "design": 2, "customiz": 2, "bayesian": 2, "infer": [2, 3], "programm": [2, 3], "formal": 2, "mathemat": 2, "represent": 2, "probabilist": [2, 5], "model": [2, 3], "enough": 2, "permit": 2, "random": [1, 2, 3], "captur": 2, "notion": 2, "variabl": 2, "exist": [1, 2], "uncertainti": [1, 2], "describ": 2, "marco": 2, "cusumano": 2, "towner": 2, "thesi": 2, "One": [2, 4], "refer": 2, "li": 2, "gen": [2, 3, 5], "jl": 2, "julia": 2, "akin": 2, "languag": 2, "reli": 2, "upon": 2, "u": [2, 4], "intermedi": 2, "program": [2, 4, 5], "oper": 2, "transform": [2, 3], "speak": 2, "abov": 2, "you": 2, "d": 2, "jump": 2, "right": 2, "read": 2, "about": 2, "visit": 2, "pure": 2, "python": [2, 3, 4], "roughli": 2, "subset": 2, "import": [2, 3], "def": [2, 3], "kei": [2, 3], "x": [2, 3], "normal": [2, 3], "print": [2, 3], "jaxgenerativefunct": 2, "0x107d052d0": [], "decor": 2, "abil": [2, 4], "let": 2, "studi": 2, "prngkei": [2, 3], "314159": [2, 3], "make_jaxpr": 2, "lambda": 2, "u32": 2, "2": 2, "b": 2, "c": [2, 5], "f32": [2, 3], "addr": 2, "_normal": [2, 3], "form": 2, "see": [1, 2], "ha": 2, "someth": 2, "few": 2, "constant": 2, "doesn": 2, "t": [2, 3], "nativ": [2, 4], "know": 2, "how": 2, "handl": 2, "semant": [2, 3], "simul": [2, 3], "4": 2, "iota": 2, "dimens": 2, "0": [2, 3], "uint32": 2, "slice": 2, "limit_indic": 2, "start_indic": 2, "stride": 2, "e": 2, "squeez": 2, "f": [2, 3], "g": 2, "none": 2, "j": [2, 4], "k": 2, "threefry2x32": 2, "l": 2, "concaten": 2, "m": 2, "reshap": 2, "new_siz": 2, "n": 2, "o": 2, "p": [2, 3], "q": 2, "r": [2, 3], "v": 2, "w": 2, "y": 2, "z": 2, "ba": 2, "bb": 2, "bc": 2, "i32": 2, "broadcast_in_dim": 2, "broadcast_dimens": 2, "bd": 2, "gather": 2, "dimension_numb": 2, "gatherdimensionnumb": 2, "offset_dim": 2, "collapsed_slice_dim": 2, "start_index_map": 2, "fill_valu": 2, "indices_are_sort": 2, "true": 2, "mode": 2, "gatherscattermod": 2, "promise_in_bound": 2, "slice_s": 2, "unique_indic": 2, "bf": 2, "shift_right_log": 2, "9": 2, "bg": 2, "1065353216": 2, "bh": 2, "bitcast_convert_typ": 2, "new_dtyp": 2, "float32": [2, 3], "bi": 2, "sub": 2, "bj": 2, "9999999403953552": 2, "bk": 2, "mul": 2, "bl": 2, "add": 2, "bm": 2, "bn": 2, "max": 2, "bo": 2, "erf_inv": 2, "bp": 2, "4142135381698608": 2, "bq": 2, "integer_pow": 2, "br": 2, "6": 2, "2831854820251465": 2, "log": 2, "bt": 2, "bu": 2, "bv": 2, "div": 2, "bw": 2, "bx": 2, "reduce_sum": 2, "ax": 2, "bz": 2, "That": 2, "quit": 2, "lot": 2, "under": [2, 3], "hood": 2, "expand": 2, "sampl": [2, 3], "updat": 2, "prng": 2, "record": 2, "probabl": 2, "densiti": 2, "result": [2, 3], "piec": 2, "data": 2, "out": 2, "essenti": 2, "repeat": 2, "each": [1, 2], "algorithm": 3, "combin": [3, 5], "map": [3, 5], "conceptu": 3, "expos": [1, 3, 5], "when": 3, "kwarg": 3, "thei": 3, "correspond": 3, "below": [1, 3], "sim": 3, "cdot": 3, "appli": 3, "ret": 3, "paramet": [1, 3], "compat": 3, "first": 3, "element": 3, "evolv": 3, "second": 3, "exampl": [3, 4], "devicearrai": 3, "10823099": 3, "score": 3, "instanc": [1, 3, 4], "along": 3, "execut": 3, "tr": 3, "jaxtrac": 3, "retval": 3, "jaxchoicemap": 3, "tree": 3, "distributiontrac": 3, "framework": [4, 5], "concept": 5, "hardwar": 5, "acceler": 5, "capabl": [4, 5], "what": 5, "gener": [4, 5], "function": [4, 5], "do": 5, "look": 5, "switch": 5, "unfold": 5, "configur": 1, "share": 1, "address": 1, "splat": 1, "sequenc": 1, "singl": 1, "potenti": 1, "usag": [1, 4], "detail": 1, "0x1060d9750": 2, "new_kei": 3, "build": 4, "top": 4, "take": 4, "advantag": 4, "deploy": 4, "proper": 4, "opportun": 4, "convert": 4, "jit": 4, "tf": 4, "readi": 4, "via": 4, "tensorflow": 4, "one": 4, "path": 4, "web": 4, "simpl": 4, "pathwai": 4, "webassembli": 5}, "objects": {"genjax.combinators": [[1, 0, 0, "-", "map"], [1, 0, 0, "-", "switch"], [1, 0, 0, "-", "unfold"]], "genjax.combinators.map": [[1, 1, 1, "", "MapCombinator"]], "genjax.combinators.map.MapCombinator": [[1, 1, 1, "", "partial"]], "genjax.combinators.map.MapCombinator.partial": [[1, 2, 1, "", "args"], [1, 2, 1, "", "func"], [1, 2, 1, "", "keywords"]], "genjax.combinators.switch": [[1, 1, 1, "", "SwitchCombinator"]], "genjax.combinators.unfold": [[1, 1, 1, "", "UnfoldCombinator"]], "genjax": [[3, 0, 0, "-", "interface"]], "genjax.interface": [[3, 3, 1, "", "sample"], [3, 3, 1, "", "simulate"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:attribute", "3": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "attribute", "Python attribute"], "3": ["py", "function", "Python function"]}, "titleterms": {"expos": 0, "c": 0, "gener": [0, 1, 2, 3], "function": [0, 1, 2, 3], "A": 0, "new": 0, "modul": 0, "combin": 1, "switch": 1, "map": 1, "unfold": 1, "what": 2, "i": 2, "do": 2, "look": 2, "like": 2, "genjax": [2, 4], "interfac": 3, "index": 5, "from": 4, "webassembli": 4}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.viewcode": 1, "sphinx": 56}})