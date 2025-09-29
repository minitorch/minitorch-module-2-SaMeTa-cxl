"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(a: float, b: float) -> float:
    """Multiply two float"""
    return a * b

def id(a: float) -> float: 
    """Return identical float"""
    return a

def add(a: float, b: float) -> float:
    """Add two float"""
    return a + b

def neg(a: float) -> float:
    """Return the negative a"""
    return -a

def lt(a: float, b: float) -> bool:
    """Check if a is less than b"""
    return a < b

def eq(a: float, b: float) -> bool:
    """Check if a is eqaul to b"""
    return a == b

def max(a: float, b: float) -> float:
    """Check the biggest of a and b"""
    return a if a > b else b 

def is_close(a: float, b: float) -> bool:
    """Return if a and b are close, specifically if their distance less than 1e-2"""
    return abs(a - b) < 1e-2

def sigmoid(a: float) -> float:
    r"""Calculate sigmoid(a) as f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}"""
    return 1.0 / (1.0 + math.exp(-a)) if a >= 0.0 else math.exp(a) / (1.0 + math.exp(a))

def relu(a: float) -> float:
    """f(x) = max(0, a)"""
    return max(0.0, a)

def log(a: float) -> float:
    """ln(x)"""
    return math.log(a)

def exp(a: float) -> float:
    """e^x"""
    return math.exp(a)

def log_back(a: float, b: float) -> float:
    """The derivitive of ln(a) * b = b / a"""
    return b / a

def inv(a: float) -> float:
    """1 / x"""
    return 1.0 / a

def inv_back(a: float, b: float) -> float:
    """The derivitive of (1 / a) *b"""
    return b / (-a * a)

def relu_back(a: float, b: float) -> float:
    """The derivitive of relu(a) * b"""
    return b if a >= 0.0 else 0.0



# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list

    """
    def iter_fn(ls: Iterable[float]) -> Iterable[float]:
        result = []
        for ele in ls:
            result.append(fn(ele))
        return result

    return iter_fn
    # raise NotImplementedError("Need to include this file from past assignment.")


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Use `map` and `neg` to negate each element in `ls`"""
    return map(neg)(ls)
    # raise NotImplementedError("Need to include this file from past assignment.")


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """
    def iter_fn(a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
        result = []
        for ele_a, ele_b in zip(a, b):
            result.append(fn(ele_a, ele_b))
        
        return result

    return iter_fn
    # raise NotImplementedError("Need to include this file from past assignment.")


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add the elements of `ls1` and `ls2` using `zipWith` and `add`"""
    return zipWith(add)(ls1, ls2)
    # raise NotImplementedError("Need to include this file from past assignment.")


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`

    """
    def iter_fn(ls: Iterable[float]) -> float:
        result = start
        for ele in ls:
            result = fn(result, ele)
        
        return result
    
    return iter_fn
    # raise NotImplementedError("Need to include this file from past assignment.")


def sum(ls: Iterable[float]) -> float:
    """Sum up a list using `reduce` and `add`."""
    return reduce(add, 0)(ls)
    # raise NotImplementedError("Need to include this file from past assignment.")


def prod(ls: Iterable[float]) -> float:
    """Product of a list using `reduce` and `mul`."""
    return reduce(mul, 1)(ls)
    # raise NotImplementedError("Need to include this file from past assignment.")
