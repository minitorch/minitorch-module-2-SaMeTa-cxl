from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol
# from minitorch.scalar import Scalar

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    # raise NotImplementedError("Need to implement for Task 1.1")
    delta = f(*(vals[:arg]), vals[arg] + epsilon, *(vals[arg + 1:])) - f(*vals)
    slope = delta / epsilon
    return slope

variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")
    stack = [variable]
    # get num of output of every variable
    while len(stack) != 0:
        now = stack.pop(0)
        for parent in now.parents:
            # print(f"child: {now.name}, parent: {parent.name}")
            if not parent.is_constant():
                if parent.num_output == 0:
                    stack.append(parent)
                parent.num_output += 1
    
    right_most_set = [variable]
    result = []
    # calculate topo order
    while len(right_most_set) != 0:
        now = right_most_set.pop()
        result.append(now)
        for parent in now.parents:
            # print("parent name: ", parent.name)
            # print(parent.num_output)
            parent.num_output -= 1
            if parent.num_output == 0:
                right_most_set.append(parent)

    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    # raise NotImplementedError("Need to implement for Task 1.4")
    topo_order = topological_sort(variable)
    variable.accumulate_derivative(deriv)
    for var in topo_order:
        if var.is_leaf():
            continue
        if hasattr(var, "derivative"):
            result = var.chain_rule(var.derivative)
        else:
            result = var.chain_rule(var.grad)

        for var_input, d_input in result:
            var_input.accumulate_derivative(d_input)


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
