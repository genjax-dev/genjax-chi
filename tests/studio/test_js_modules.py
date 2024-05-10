#%%

import gen.studio.plot as plot 
import gen.studio.js_modules as js_modules 
import importlib 
importlib.reload(js_modules)
importlib.reload(plot)
from gen.studio.plot  import JSRef, d3, Math

def test_jswrapper_init():
    wrapper = JSRef("TestModule", "test_method")
    assert wrapper == {
        "pyobsplot-type": "ref",
        "module": "TestModule",
        "name": "test_method",
    }
    assert wrapper.__name__ == "test_method"
    assert wrapper.__doc__ is None

def test_jswrapper_call():
    def test_inner(fn, *args):
        return fn(*args)
    
    wrapper = JSRef("TestModule", "test_method", inner=test_inner)
    result = wrapper(1, 2, 3)
    
    assert result == {
        "pyobsplot-type": "function",
        "module": "TestModule",
        "name": "test_method", 
        "args": (1, 2, 3)
    }
    
def test_jsmodule_getattr():
    result = d3.test_method
    assert isinstance(result, JSRef)
    assert result == {
        "pyobsplot-type": "ref",
        "module": "d3",
        "name": "test_method",
    }

def test_math_getattr():
    result = Math.test_method
    assert isinstance(result, JSRef)
    assert result == {
        "pyobsplot-type": "ref",
        "module": "Math",
        "name": "test_method",
    }
    
def test_d3_method_call():
    result = d3.test_method(1, 2, 3)
    assert result == {
        "pyobsplot-type": "function",
        "module": "d3",
        "name": "test_method",
        "args": (1, 2, 3)
    }

def test_math_method_call():
    result = Math.test_method(4, 5, 6)
    assert result == {
        "pyobsplot-type": "function", 
        "module": "Math",
        "name": "test_method",
        "args": (4, 5, 6)
    }

def run_tests():
    test_jsmodule_getattr()
    test_math_getattr()
    test_jswrapper_call()
    test_jswrapper_init()
    test_d3_method_call()
    test_math_method_call()   
    print("all tests pass")

run_tests()    

#%%