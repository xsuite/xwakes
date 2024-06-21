from __future__ import annotations

from .parameters import *
from .utils import unique_sigfigs

from typing import Optional, Callable, Tuple, Union, List

import numpy as np


def mix_fine_and_rough_sampling(start: float, stop: float, rough_points: int,
                                fine_points: int, rois: List[Tuple[float, float]]):
    """
    Mix a fine and rough (geometric) sampling between start and stop,
    refined in the regions of interest rois.
    :param start: The lowest bound of the sampling
    :param stop: The highest bound of the sampling
    :param rough_points: The total number of data points to be
                         generated for the rough grid, between start
                         and stop.
    :param fine_points: The number of points in the fine grid of each roi.
    :param rois: List of unique tuples with the lower and upper bound of
                 of each region of interest.
    :return: An array with the sampling obtained.
    """
    intervals = [np.linspace(max(i, start), min(f, stop), fine_points)
                 for i, f in rois
                 if (start <= i <= stop or start <= f <= stop)]
    rough_sampling = np.geomspace(start, stop, rough_points)
    # the following concatenates adds the rough points to the fine sampling and sorts
    # the result. Then duplicates are removed, where two points are considered
    # equal if they are within 7 significant figures of each other.
    return unique_sigfigs(
        np.sort(np.hstack((*intervals, rough_sampling)),kind='stable'), 7)


class Component:
    """
    A data structure representing the impedance- and wake functions of some Component in a specified plane.
    """

    def __init__(self, impedance: Optional[Callable] = None, wake: Optional[Callable] = None, plane: str = '',
                 source_exponents: Tuple[int, int] = (-1, -1), test_exponents: Tuple[int, int] = (-1, -1),
                 name: str = "Unnamed Component", f_rois: Optional[List[Tuple[float, float]]] = None,
                 t_rois: Optional[List[Tuple[float, float]]] = None):
        """
        The initialization function for the Component class.
        :param impedance: A callable function representing the impedance function of the Component. Can be undefined if
        the wake function is defined.
        :param wake: A callable function representing the wake function of the Component. Can be undefined if
        the impedance function is defined.
        :param plane: The plane of the Component, either 'x', 'y' or 'z'. Must be specified for valid initialization
        :param source_exponents: The exponents in the x and y planes experienced by the source particle. Also
        referred to as 'a' and 'b'. Must be specified for valid initialization
        :param test_exponents: The exponents in the x and y planes experienced by the source particle. Also
        referred to as 'a' and 'b'. Must be specified for valid initialization
        :param name: An optional user-specified name of the component
        """
        # Enforces that either impedance or wake is defined.
        assert impedance or wake, "The impedance- and wake functions cannot both be undefined."
        # The impedance- and wake functions as callable objects, e.g lambda functions
        self.impedance = impedance
        self.wake = wake
        self.name = name

        # The plane of the Component, either 'x', 'y' or 'z'
        assert plane.lower() in ['x', 'y', 'z'], \
            "Cannot initialize Component object without specified plane"
        self.plane = plane

        assert source_exponents != (-1, -1) and len(source_exponents) == 2, \
            "Cannot initialize Component object without specified source exponents (a, b)"
        self.source_exponents = source_exponents
        assert test_exponents != (-1, -1) and len(test_exponents) == 2, \
            "Cannot initialize Component object without specified test exponents (c, d)"
        self.test_exponents = test_exponents
        self.power_x = (source_exponents[0] + test_exponents[0] + (plane == 'x')) / 2
        self.power_y = (source_exponents[1] + test_exponents[1] + (plane == 'y')) / 2
        self.f_rois = f_rois if f_rois else []
        self.t_rois = t_rois if t_rois else []

    def generate_wake_from_impedance(self) -> None:
        """
        Uses the impedance function of the Component object to generate its wake function, using
        a Fourier transform.
        :return: Nothing
        """
        # # If the object already has a wake function, there is no need to generate it.
        # if self.wake:
        #     pass
        # # In order to generate a wake function, we need to make sure the impedance function is defined.
        # assert self.impedance, "Tried to generate wake from impedance, but impedance is not defined."
        #
        # raise NotImplementedError

        # Temporary solution to avoid crashes
        self.wake = lambda x: 0

    def generate_impedance_from_wake(self) -> None:
        """
        Uses the wake function of the Component object to generate its impedance function, using
        a Fourier transform.
        :return: Nothing
        """
        # # If the object already has an impedance function, there is no need to generate it.
        # if self.impedance:
        #     pass
        # # In order to generate an impedance function, we need to make sure the wake function is defined.
        # assert self.wake, "Tried to generate impedance from wake, but wake is not defined."
        #
        # raise NotImplementedError

        # Temporary solution to avoid crashes
        self.impedance = lambda x: 0

    def is_compatible(self, other: Component) -> bool:
        """
        Compares all parameters of the self-object with the argument of the function and returns True if all of their
        attributes, apart from the impedance and wake functions, are identical.
        i.e. Returns True if self and other are compatible for Component addition, False otherwise
        :param other: Another Component
        :return: True if self and other can be added together, False otherwise
        """
        return all([self.source_exponents == other.source_exponents,
                    self.test_exponents == other.test_exponents,
                    self.plane == other.plane,
                    self.power_x == other.power_x,
                    self.power_y == other.power_y])

    def __add__(self, other: Component) -> Component:
        """
        Defines the addition operator for two Components
        :param self: The left addend
        :param other: The right addend
        :return: A new Component whose impedance and wake functions are the sums
        of the respective functions of the two addends.
        """
        # Enforce that the two addends are in the same plane
        assert self.plane == other.plane, "The two addends correspond to different planes and cannot be added.\n" \
                                          f"{self.plane} != {other.plane}"

        # Enforce that the two addends have the same exponent parameters
        assert self.source_exponents == other.source_exponents and self.test_exponents == other.test_exponents, \
            "The two addends have different exponent parameters and cannot be added.\n" \
            f"Source: {self.source_exponents} != {other.source_exponents}\n" \
            f"Test: {self.test_exponents} != {other.test_exponents}"

        # Defines an empty array to hold the two summed functions
        sums = []

        # Iterates through the two pairs of functions: impedances, then wakes
        for left, right in zip((self.impedance, self.wake), (other.impedance, other.wake)):
            # If neither addend has a defined function, we will not bother to calculate that of the sum
            if (not left) and (not right):
                sums.append(None)
            else:
                # # Generates the missing function for the addend which is missing it
                # if not left:
                #     [self.generate_impedance_from_wake, self.generate_wake_from_impedance][len(sums)]()
                # elif not right:
                #     [other.generate_impedance_from_wake, other.generate_wake_from_impedance][len(sums)]()
                #

                # TODO: Temporary fix until Fourier transform implemented
                if not left:
                    sums.append(right)
                elif not right:
                    sums.append(left)
                else:
                    # Appends the sum of the functions of the two addends to the list "sums"
                    sums.append(lambda x, l=left, r=right: l(x) + r(x))

        # Initializes and returns a new Component
        return Component(sums[0], sums[1], self.plane, self.source_exponents, self.test_exponents,
                         f_rois=self.f_rois + other.f_rois, t_rois=self.t_rois + other.t_rois)

    def __radd__(self, other: Union[int, Component]) -> Component:
        """
        Implements the __radd__ method for the Component class. This is only done to facilitate the syntactically
        practical use of the sum() method for Components. sum(iterable) works by adding all of the elements of the
        iterable to 0 sequentially. Thus, the behavior of the initial 0 + iterable[0] needs to be defined. In the case
        that the left addend of any addition involving a Component is not itself a Component, the resulting sum
        is simply defined to be the right addend.
        :param other: The left addend of an addition
        :return: The sum of self and other if other is a Component, otherwise just self.
        """
        # Checks if the left addend, other, is not a Component
        if not isinstance(other, Component):
            # In which case, the right addend is simply returned
            return self

        # Otherwise, their sum is returned (by invocation of Component.__add__(self, other))
        return self + other

    def __mul__(self, scalar: complex) -> Component:
        """
        Defines the behavior of multiplication of a Component by some, potentially complex, scalar
        :param scalar: A scalar value to be multiplied with some Component
        :return: A newly initialized Component identical to self in every way apart from the impedance-
        and wake functions, which have been multiplied by the scalar.
        """
        # Throws an AssertionError if scalar is not of the type complex, float or int
        assert isinstance(scalar, complex) or isinstance(scalar, float) or isinstance(scalar, int)
        # Initializes and returns a new Component with attributes like self, apart from the scaled functions
        return Component((lambda x: scalar * self.impedance(x)) if self.impedance else None,
                         (lambda x: scalar * self.wake(x)) if self.wake else None, self.plane,
                         self.source_exponents, self.test_exponents, self.name, self.f_rois, self.t_rois)

    def __rmul__(self, scalar: complex) -> Component:
        """
        Generalizes scalar multiplication of Component to be possibly from left and right. Both of these operations
        are identical.
        :param scalar: A scalar value to be multiplied with some Component
        :return: The result of calling Component.__mul__(self, scalar): A newly initialized Component identical to self
        in every way apart from the impedance- and wake functions, which have been multiplied by the scalar.
        """
        # Simply swaps the places of scalar and self in order to invoke the previously defined __mul__ function
        return self * scalar

    def __truediv__(self, scalar: complex) -> Component:
        """
        Implements the __truediv__ method for the Component class in order to produce the expected behavior when
        dividing a Component by some scalar. That is, the scalar multiplication of the multiplicative inverse of
        the scalar.
        :param scalar: A scalar value for the Component to be divided by
        :return: The result of calling Component.__mul__(self, 1 / scalar): A newly initialized Component identical to
        self in every way apart from the impedance- and wake functions, which have been multiplied by (1 / scalar).
        """
        # Defines the operation c / z to be equivalent to c * (1 / z) for some Component c and scalar z.
        return self * (1 / scalar)

    def __str__(self) -> str:
        """
        Implements the __str__ method for the Component class, providing an informative printout of the attributes
        of the Component.
        :return: A multi-line string containing information about the attributes of self, including which of the
        impedance- and wake functions are defined.
        """
        return f"{self.name} with parameters:\nPlane:\t\t\t{self.plane}\n" \
               f"Source exponents:\t{', '.join(str(i) for i in self.source_exponents)}\n" \
               f"Test exponents:\t\t{', '.join(str(i) for i in self.test_exponents)}\n" \
               f"Impedance function:\t{'DEFINED' if self.impedance else 'UNDEFINED'}\n" \
               f"Wake function:\t\t{'DEFINED' if self.wake else 'UNDEFINED'}\n" \
               f"Impedance-regions of interest: {', '.join(str(x) for x in self.f_rois)}\n" \
               f"Wake-regions of interest: {', '.join(str(x) for x in self.t_rois)}"

    def __lt__(self, other: Component) -> bool:
        """
        Implements the __lt__ (less than) method for the Component class. This has no real physical interpretation,
        it is just syntactically practical when iterating through the Components when adding two Elements.
        (see Element.__add__)
        :param other: The right hand side of a "<"-inequality
        :return: True if the self-Component is "less than" the other-Component by some arbitrary, but consistent,
        comparison.
        """
        # The two Components are compared by their attributes
        return [self.plane, self.source_exponents, self.test_exponents] < \
               [other.plane, other.source_exponents, other.test_exponents]

    def __eq__(self, other: Component) -> bool:
        """
        Implements the __eq__ method for the Component class. Two Components are designated as "equal" if the following
        two conditions hold:
        1. They are compatible for addition. That is, all of their attributes, apart from impedance- and wake functions,
        are exactly equal.
        2. The resulting values from evaluating the impedance functions of the two components respectively for 50
        points between 1 and 10000 all need to be close within some given tolerance. The same has to hold for the wake
        function.
        This somewhat approximated numerical approach to the equality comparator aims to compensate for small
        numerical/precision errors accumulated for two Components which have taken different "paths" to what should
        analytically be identical Components.
        :param other: The right hand side of the equality comparator
        :return: True if the two Components have identical attributes and sufficiently close functions, False otherwise
        """
        # First checks if the two Components are compatible for addition, i.e. if they have the same non-function
        # attributes
        if not self.is_compatible(other):
            # If they do not, they are not equal
            return False

        # Creates a numpy array of 50 points and verifies that the evaluations of the functions of the two components
        # for all of these components are sufficiently close. If they are, True is returned, otherwise, False is
        # returned.
        xs = np.linspace(1, 10000, 50)
        return (np.allclose(self.impedance(xs), other.impedance(xs), rtol=REL_TOL, atol=ABS_TOL) and
                np.allclose(self.wake(xs), other.wake(xs), rtol=REL_TOL, atol=ABS_TOL))

    def impedance_to_array(self, rough_points: int, start: float = MIN_FREQ,
                           stop: float = MAX_FREQ,
                           precision_factor: float = FREQ_P_FACTOR) -> Tuple[np.ndarray, np.ndarray]:
        """
        Produces a frequency grid based on the f_rois attribute of the component and evaluates the component's
        impedance function at these frequencies.
        :param rough_points: The total number of data points to be
                             generated for the rough grid, between start
                             and stop.
        :param start: The lowest frequency in the desired frequency grid
        :param stop: The highest frequency in the desired frequency grid
        :param precision_factor: A number indicating the ratio of points
               which should be placed within the regions of interest.
               If =0, the frequency grid will ignore the intervals in f_rois.
               If =1, the points will be distributed 1:1 between the
               rough grid and the fine grid on each roi. In general, =n
               means that there will be n times more points in the fine
               grid of each roi, than in the rough grid.
        :return: A tuple of two numpy arrays with same shape, giving
                 the frequency grid and impedances respectively
        """
        if len(self.f_rois) == 0:
            xs = np.geomspace(start, stop, rough_points)
            return xs, self.impedance(xs)

        # eliminate duplicates
        f_rois_no_dup = set(self.f_rois)

        fine_points_per_roi = int(round(rough_points * precision_factor))

        xs = mix_fine_and_rough_sampling(start, stop, rough_points,
                                         fine_points_per_roi,
                                         list(f_rois_no_dup))

        return xs, self.impedance(xs)

    def wake_to_array(self, rough_points: int, start: float = MIN_TIME,
                      stop: float = MAX_TIME,
                      precision_factor: float = TIME_P_FACTOR) -> Tuple[np.ndarray, np.ndarray]:
        """
        Produces a time grid based on the t_rois attribute of the component and evaluates the component's
        wake function at these time points.
        :param rough_points: The total number of data points to be
                             generated for the rough grid, between start
                             and stop.
        :param start: The lowest time in the desired time grid
        :param stop: The highest time in the desired time grid
        :param precision_factor: A number indicating the ratio of points
               which should be placed within the regions of interest.
               If =0, the frequency grid will ignore the intervals in t_rois.
               If =1, the points will be distributed 1:1 between the
               rough grid and the fine grid on each roi. In general, =n
               means that there will be n times more points in the fine
               grid of each roi, than in the rough grid.
        :return: A tuple of two numpy arrays with same shape, giving the
                 time grid and wakes respectively
        """
        if len(self.t_rois) == 0:
            xs = np.geomspace(start, stop, rough_points)
            return xs, self.wake(xs)

        # eliminate duplicates
        t_rois_no_dup = set(self.t_rois)

        fine_points_per_roi = int(round(rough_points * precision_factor))

        xs = mix_fine_and_rough_sampling(start, stop, rough_points,
                                         fine_points_per_roi,
                                         list(t_rois_no_dup))

        return xs, self.wake(xs)

    def discretize(self, freq_points: int, time_points: int, freq_start: float = MIN_FREQ, freq_stop: float = MAX_FREQ,
                   time_start: float = MIN_TIME, time_stop: float = MAX_TIME,
                   freq_precision_factor: float = FREQ_P_FACTOR,
                   time_precision_factor: float = TIME_P_FACTOR) -> Tuple[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Combines the two discretization-functions in order to fully discretize the wake and impedance of the object
        as specified by a number of parameters.
        :param freq_points: The total number of frequency/impedance points
        :param time_points: The total number of time/wake points
        :param freq_start: The lowest frequency in the frequency grid
        :param freq_stop: The highest frequency in the frequency grid
        :param time_start: The lowest time in the time grid
        :param time_stop: The highest time in the time grid
        :param freq_precision_factor: The ratio of points in the fine frequency grid
        (#points in frequency ROIs / rough frequency points)
        :param time_precision_factor: The ratio of points in the fine time grid
        (#points in time ROIs / rough time points)
        :return: A tuple of two tuples, containing the results of impedance_to_array and wake_to_array respectively
        """
        return (self.impedance_to_array(freq_points, freq_start, freq_stop, freq_precision_factor),
                self.wake_to_array(time_points, time_start, time_stop, time_precision_factor))

    def get_shorthand_type(self) -> str:
        """
        :return: A 5-character string indicating the plane as well as source- and test exponents of the component
        """
        return self.plane + "".join(str(x) for x in (self.source_exponents + self.test_exponents))
