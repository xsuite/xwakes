from component import Component
from element import Element
from budget import Budget
from test.test_common import functions
from test.test_element import create_n_elements
from random import choice, seed
import matplotlib.pyplot as plt
import numpy as np


def plot_component(component: Component, plot_impedance: bool = True, plot_wake: bool = True, start: float = 1,
                   stop: float = 10000, points: int = 200, step_size: float = None, plot_real: bool = True,
                   plot_imag: bool = True) -> None:
    """
    Function for plotting real and imaginary parts of impedance and wake functions of a single component
    :param component: The component to be plotted
    :param plot_impedance: A flag indicating if impedance should be plotted
    :param plot_wake: A flag indicating if wake should be plotted
    :param start: The first value on the x-axis of the plot
    :param stop: The last value on the x-axis of the plot
    :param points: The number of points to be evaluated for each line of the plot (alternative to step_size)
    :param step_size: The distance between each point on the x-axis (alternative to points)
    :param plot_real: A flag indicating if the real values should be plotted
    :param plot_imag: A flag indicating if the imaginary values should be plotted
    :return: Nothing
    """
    assert (plot_wake or plot_impedance) and (plot_real or plot_imag), "There is nothing to plot"
    assert stop - start > 0, "stop must be greater than start"
    if step_size:
        assert step_size > 0, "Negative step_size not possible"
        xs = np.arange(start, stop, step_size, dtype=float)
    else:
        xs = np.linspace(start, stop, points)

    legends = []
    if plot_impedance:
        ys = component.impedance(xs)
        if plot_real:
            plt.plot(xs, ys.real)
            legends.append("Re[Z(f)]")
        if plot_imag:
            plt.plot(xs, ys.imag)
            legends.append("Im[Z(f)]")
    if plot_wake:
        ys = component.wake(xs)
        if plot_real:
            plt.plot(xs, ys.real)
            legends.append("Re[W(z)]")
        if plot_imag:
            plt.plot(xs, ys.imag)
            legends.append("Im[W(z)]")

    plt.legend(legends)
    plt.show()


def plot_element_in_plane(element: Element, plane: str, plot_impedance: bool = True, plot_wake: bool = True,
                          start: float = 1, stop: float = 10000, points: int = 200, step_size: float = None,
                          plot_real: bool = True, plot_imag: bool = True):
    """
    Sums all Components of Element in specified plane and plots the resulting Component with the parameters given.
    :param element: The Element with Components to be plotted
    :param plane: The plane in which the Components to be plotted lie
    :param plot_impedance: A flag indicating if impedance should be plotted
    :param plot_wake: A flag indicating if wake should be plotted
    :param start: The first value on the x-axis of the plot
    :param stop: The last value on the x-axis of the plot
    :param points: The number of points to be evaluated for each line of the plot (alternative to step_size)
    :param step_size: The distance between each point on the x-axis (alternative to points)
    :param plot_real: A flag indicating if the real values should be plotted
    :param plot_imag: A flag indicating if the imaginary values should be plotted
    :return: Nothing
    """
    plane = plane.lower()
    assert plane in ['x', 'y', 'z'], f"{plane} is not a valid plane. Must be 'x', 'y' or 'z'"
    component = sum([c for c in element.components if c.plane == plane])

    assert component, f"Element has no components in plane {plane}"
    plot_component(component, plot_impedance=plot_impedance, plot_wake=plot_wake, start=start, stop=stop,
                   points=points, step_size=step_size, plot_real=plot_real, plot_imag=plot_imag)
