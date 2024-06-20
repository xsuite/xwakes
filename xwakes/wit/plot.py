from .component import Component
from .element import Element
from .model import Model
from .parameters import *

from typing import List, Dict, Union, Optional, Set
from collections import defaultdict

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
    import matplotlib.pyplot as plt
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


def plot_component_impedance(component: Component, logscale_x: bool = True, logscale_y: bool = True,
                             points: int = 1000, start=MIN_FREQ, stop=MAX_FREQ, title: Optional[str] = None) -> None:
    import matplotlib.pyplot as plt
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot(111)
    fs = np.geomspace(start, stop, points)
    impedances = component.impedance(fs)
    reals, imags = impedances.real, impedances.imag
    ax.plot(fs, reals, label='real')
    ax.plot(fs, imags, label='imag')
    ax.set_xlim(start, stop)
    if title:
        plt.title(title)

    if logscale_x:
        ax.set_xscale('log')

    if logscale_y:
        ax.set_yscale('log')

    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Z [Ohm / m]')

    plt.legend()
    plt.show()


def plot_component_wake(component: Component, logscale_x: bool = True, logscale_y: bool = True,
                        points: int = 1000, start=MIN_TIME, stop=MAX_TIME, title: Optional[str] = None) -> None:
    import matplotlib.pyplot as plt
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot(111)
    ts = np.geomspace(start, stop, points)
    ax.plot(ts, component.wake(ts))
    ax.set_xlim(start, stop)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Wake [V / C * m]')

    if title:
        plt.title(title)

    if logscale_x:
        ax.set_xscale('log')

    if logscale_y:
        ax.set_yscale('log')

    plt.show()


def generate_contribution_plots(model: Model, start_freq: float = MIN_FREQ, stop_freq: float = MAX_FREQ,
                                start_time: float = MIN_TIME, stop_time: float = MAX_TIME, points: int = 1000,
                                freq_scale: str = 'log', time_scale: str = 'log', absolute: bool = False) -> None:

    import matplotlib.pyplot as plt

    # TODO: use roi's to generate grid
    fs = np.geomspace(start_freq, stop_freq, points)
    ts = np.geomspace(start_time, stop_time, points)

    all_tags = set([e.tag for e in model.elements])
    elements: Dict[str, Union[int, Element]] = defaultdict(int)

    for element in model.elements:
        elements[element.tag] += element

    components_defined_for_tag: Dict[str, Set[str]] = dict()
    all_type_strings = set()
    for tag, element in elements.items():
        components_defined_for_tag[tag] = {c.get_shorthand_type() for c in element.components}
        all_type_strings.update(components_defined_for_tag[tag])

    tags = list(all_tags)
    cumulative_elements: List[Element] = []
    defined_type_strings: List[set] = []
    current_defined = set()
    current_element = 0
    for tag in tags:
        new_element = current_element + elements[tag]
        current_defined = current_defined.union(components_defined_for_tag[tag])
        defined_type_strings.append(current_defined)
        cumulative_elements.append(new_element)
        current_element = new_element

    for type_string in all_type_strings:
        wakes = np.asarray([np.zeros(shape=ts.shape)])
        real_impedances = np.asarray([np.zeros(shape=fs.shape)])
        imag_impedances = np.asarray([np.zeros(shape=fs.shape)])
        for i, element in enumerate(cumulative_elements):
            if type_string not in defined_type_strings[i]:
                wakes = np.vstack((wakes, wakes[-1]))
                real_impedances = np.vstack((real_impedances, real_impedances[-1]))
                imag_impedances = np.vstack((imag_impedances, imag_impedances[-1]))
                continue

            component = [c for c in element.components if c.get_shorthand_type() == type_string][0]
            if component.wake is not None:
                array = component.wake(ts)
                wakes = np.vstack((wakes, array))
            else:
                wakes = np.vstack((wakes, wakes[-1]))

            if component.impedance is not None:
                impedances = component.impedance(fs)
                real_impedances = np.vstack((real_impedances, impedances.real))
                imag_impedances = np.vstack((imag_impedances, impedances.imag))
            else:
                real_impedances = np.vstack((real_impedances, real_impedances[-1]))
                imag_impedances = np.vstack((imag_impedances, imag_impedances[-1]))

        titles = (f'Re[Z] contribution - {type_string}',
                  f'Im[Z] contribution - {type_string}',
                  f'Wake contribution - {type_string}')

        for i, (array, xs, title) in enumerate(zip((real_impedances, imag_impedances, wakes),
                                                   (fs, fs, ts), titles)):
            if sum(array[-1]) == 0:
                continue

            if not absolute:
                array = np.divide(array, array[-1])

            fig: plt.Figure = plt.figure()
            ax: plt.Axes = fig.add_subplot(111)
            for j in range(len(all_tags) - 1, -1, -1):
                ax.fill_between(xs, array[j], array[j + 1], label=tags[j])

            ax.set_xscale(time_scale if i == 2 else freq_scale)
            if not absolute:
                ax.set_ylim(0, 1)
            ax.set_xlim(start_time if i == 2 else start_freq, stop_time if i == 2 else stop_freq)
            plt.title(title)
            plt.legend()
            plt.show()


def plot_total_impedance_and_wake(model: Model, logscale_y=True, start_freq: float = MIN_FREQ,
                                  stop_freq: float = MAX_FREQ, start_time: float = MIN_TIME,
                                  stop_time: float = MAX_TIME, points: int = 1000,
                                  logscale_freq: bool = True, logscale_time: bool = True):
    element = sum(model.elements)
    for component in element.components:
        if component.impedance:
            plot_component_impedance(component, logscale_y=logscale_y,
                                     title=f"Total impedance - {component.get_shorthand_type()}", start=start_freq,
                                     stop=stop_freq, points=points, logscale_x=logscale_freq)
        if component.wake:
            plot_component_wake(component, logscale_y=logscale_y,
                                title=f"Total wake - {component.get_shorthand_type()}", start=start_time,
                                stop=stop_time, points=points, logscale_x=logscale_time)
