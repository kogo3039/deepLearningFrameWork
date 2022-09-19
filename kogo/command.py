import weakref
import numpy as np
import contextlib
import kogo

# ========
#  Config
# ========

class Config:
    enable_backprop = True
    
@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)


# =====================
#  Varialbe / Function
# =====================

class Variable:
    
    __array_proiority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported.".format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    # 특징 추출 overiding
    @property
    def shape(self):
        return self.data.shape
    @property
    def ndim(self):
        return self.data.ndim
    @property
    def size(self):
        return self.data.size
    @property
    def dtype(self):
        return self.data.dtype
    def __len__(self):
        return len(self.data)
    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        p = str(self.data).replace('\n', '\n'+' '*9)
        return 'Variable(' + p + ')'
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
                
        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs] #weakref(output)
            gxs = f.backward(*gys)
            
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
                
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = x
                else:
                    x.grad = x.grad + x
                if x.creator is not None:
                    add_func(x.creator)
                    
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None #weakref(output)

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    
    def __call__(self, *inputs):
        
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        if Config.enable_backprop:
            
            self.generation = max([x.generation for x in inputs])
            
            for output in outputs:
                output.set_creator(self)
                
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
            
        return outputs if len(outputs)>1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()
    def backward(self, gys):
        raise NotImplementedError()
    
                
# =================
#  함수 상속 연산자 
# =================

def add(x, y):
    y = as_array(y)
    return Add()(x, y)
class Add(Function):
    def forward(self, x, y):
        return x+y
    def backward(self, gy):
        return gy, gy

def mul(x, y):
    y = as_array(y)
    return Mul()(x, y)
class Mul(Function):
    def forward(self, x, y):
        return x * y
    def backward(self, gy):
        x, y = self.inputs[0].data, self.inputs[1].data
        return gy*y, gy*x

def neg(x):
    return Neg()(x)
class Neg(Function):
    def forward(self, x):
        return -x
    def backward(self, gy):
        return -gy

def sub(x, y):
    y = as_array(y)
    return Sub()(x, y)
def rsub(x, y):
    y = as_array(y)
    return Sub()(y, x)
class Sub(Function):
    def forward(self, x, y):
        return x - y
    def backward(self, gy):
        x, y = self.inputs[0].data, self.inputs[1].data
        return gy, -gy

def div(x, y):
    y = as_array(y)
    return Div()(x, y)
def rdiv(x, y):
    y = as_array(y)
    return Div()(y, x)
class Div(Function):
    def forward(self, x, y):
        return x / y
    def backward(self, gy):
        x, y = self.inputs[0].data, self.inputs[1].data
        return gy/y, gy*(-x/y**2)

def pow(x, c):
    return Pow(c)(x)
class Pow(Function):
    def __init__(self, c):
        self.c = c
    def forward(self, x):
        return x**self.c
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        return gy*c*x**(c-1)

# =================
#  연산자 오버로드 
# =================

def setup_variable():
    Variable.__add__      = add
    Variable.__radd__     = add
    Variable.__mul__      = mul
    Variable.__rmul__     = mul
    Variable.__neg__      = neg
    Variable.__sub__      = sub
    Variable.__rsub__     = rsub
    Variable.__truediv__  = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__      = pow





    









