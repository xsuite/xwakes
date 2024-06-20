from __future__ import annotations

from .component import Component, Union

from typing import List
from collections import defaultdict

from scipy.special import comb
import numpy as np
import copy

class Element:
    def __init__(self, length: float = 0, beta_x: float = 0, beta_y: float = 0,
                 components: List[Component] = None, name: str = "Unnamed Element",
                 tag: str = "", description: str = ""):
        """
        The initialization function of the Element class.
        :param length: The length of the element, must be specified for valid initialization
        :param beta_x: The size of the beta function in the x-plane at the position of the Element, must be specified
        for valid initialization
        :param beta_y: The size of the beta function in the y-plane at the position of the Element, must be specified
        for valid initialization
        :param components: A list of the Components corresponding to the Element being initialized. If the list contains
        multiple Components with the same values for (plane, source_exponents, test_exponents), and error is thrown. If
        the list is not specified, the Element is initialized with an empty components-list.
        :param name: A user-specified name of the Element
        :param tag: A string corresponding to a specific component
        """
        assert length > 0, "The element must have a specified valid length"
        assert beta_x > 0 and beta_y > 0, "The element must have valid specified beta_x and beta_y values"
        self.length = length
        self.beta_x = beta_x
        self.beta_y = beta_y
        self.name = name
        self.tag = tag
        self.description = description
        if components:
            comp_dict = defaultdict(int)
            for c in components:
                comp_dict[(c.plane, c.source_exponents, c.test_exponents)] += c
            components = comp_dict.values()
            self.components = sorted(components, key=lambda x: (x.plane, x.source_exponents, x.test_exponents))
        else:
            self.components: List[Component] = []

    def rotated(self, theta: float, rotate_beta: bool = False) -> Element:
        """
        Returns a copy if the self-Element which has been rotated counterclockwise in the transverse plane by an
        angle of theta radians. If the rotate_beta flag is enable, the beta_x and beta_y values of the new Element
        are rotated correspondingly.
        :param theta: The angle, in radians, the Element is to be rotated by
        :param rotate_beta: A flag indicating whether or not the beta_x and beta_y values of the new Element should also
        be rotated by theta. Enabled by default.
        :return: A newly initialized copy of the self-element which has been rotated counterclockwise in the
        transverse plane by an angle theta.
        """
        # Confines theta to the interval [0, 2 * pi)
        theta %= 2 * np.pi
        # Precalculates the sine and cosine of theta to save time
        costheta, sintheta = np.cos(theta), np.sin(theta)
        # Creates a dictionary where the keys are Component attributes and the values are Components with those
        # corresponding attributes. That way, Components created by the rotation can always be added to other compatible
        # Components.
        rotated_components = defaultdict(int)

        # Iterates through all the components of the self-Element
        for cmp in self.components:
            # Defines the x- and y-coefficients depending on what plane the Component is in
            if cmp.plane == 'x':
                coefx, coefy = costheta, sintheta
            else:
                coefx, coefy = -sintheta, costheta
            # For compactness, a, b, c and d are used to refer to the source_exponents and the test_exponents
            # of the component
            a, b, c, d = cmp.source_exponents + cmp.test_exponents
            for i in range(a + 1):
                for j in range(b + 1):
                    for k in range(c + 1):
                        for l in range(d + 1):
                            # Calculates new a, b, c and d values for the new Component
                            new_a, new_b, new_c, new_d = i + j, a - i + b - j, k + l, c - k + d - l
                            # Product of binomial coefficients
                            binprod = int(comb(a, i, exact=True) * comb(b, j, exact=True) *
                                          comb(c, k, exact=True) * comb(d, l, exact=True))
                            # Multiply by power of cos and sin
                            coef = ((-1) ** (j + l)) * binprod * (costheta ** (i + b - j + k + d - l)) * \
                                   (sintheta ** (a - i + j + c - k + l))
                            # Depending on if the component is in the longitudinal or transverse plane, one or two
                            # components are created and added to the element at the correct key in the
                            # rotated_components dictionary
                            if cmp.plane == 'z':
                                # If the component is scaled by less than 10 ** -6 we assume that it is zero
                                if abs(coef) > 1e-6:
                                    rotated_components['z', new_a, new_b, new_c, new_d] += \
                                        coef * Component(impedance=cmp.impedance, wake=cmp.wake, plane='z',
                                                         source_exponents=(new_a, new_b),
                                                         test_exponents=(new_c, new_d))
                            else:
                                if abs(coefx * coef) > 1e-6:
                                    rotated_components['x', new_a, new_b, new_c, new_d] += \
                                        (coefx * coef) * Component(impedance=cmp.impedance, wake=cmp.wake, plane='x',
                                                                   source_exponents=(new_a, new_b),
                                                                   test_exponents=(new_c, new_d))
                                if abs(coefy * coef) > 1e-6:
                                    rotated_components['y', new_a, new_b, new_c, new_d] += \
                                        (coefy * coef) * Component(impedance=cmp.impedance, wake=cmp.wake, plane='y',
                                                                   source_exponents=(new_a, new_b),
                                                                   test_exponents=(new_c, new_d))

        # New beta_x and beta_y values are defined if the rotate_beta flag is active
        new_beta_x = ((costheta * np.sqrt(self.beta_x) -
                       sintheta * np.sqrt(self.beta_y)) ** 2) if rotate_beta else self.beta_x
        new_beta_y = ((sintheta * np.sqrt(self.beta_x) +
                       costheta * np.sqrt(self.beta_y)) ** 2) if rotate_beta else self.beta_y

        # Initializes and returns a new element with parameters calculated above. Its Component list is a list of
        # the values in the rotated_components dictionary, sorted by the key which is used consistently for
        # Component comparisons
        return Element(self.length, new_beta_x, new_beta_y,
                       sorted(list(rotated_components.values()),
                              key=lambda x: (x.plane, x.source_exponents, x.test_exponents)),
                       self.name, self.tag, self.description)

    def is_compatible(self, other: Element, verbose: bool = False) -> bool:
        """
        Compares all non-components parameters of the two Elements and returns False if they are not all equal within
        some tolerance. The Component lists of the two Elements also need to be of equal length in order for the
        function to return True.
        :param other: An element for the self-Element to be compared against
        :param verbose: A flag which can be activated to give feedback on the values of the parameters which caused
        the function to return False
        :return: True if all non-components parameters are equal within some tolerance and the Components lists of the
        two Elements are of the same length. False otherwise.
        """
        if verbose:
            if abs(self.length - other.length) > 1e-8:
                print(f"Different lengths: {self.length} != {other.length}")
                return False
            if abs(self.beta_x - other.beta_x) > 1e-8:
                print(f"Different beta_x: {self.beta_x} != {other.beta_x}")
                return False
            if abs(self.beta_y - other.beta_y) > 1e-8:
                print(f"Different beta_y: {self.beta_y} != {other.beta_y}")
                return False
            if len(self.components) != len(other.components):
                print(f"Different number of components: {len(self.components)} != {len(other.components)}")
                return False
            return True
        else:
            return all([abs(self.length - other.length) < 1e-8,
                        abs(self.beta_x - other.beta_x) < 1e-8,
                        abs(self.beta_y - other.beta_y) < 1e-8,
                        len(self.components) == len(other.components)])

    def changed_betas(self, new_beta_x: float, new_beta_y: float) -> Element:
        element_copy = copy.deepcopy(self)
        x_ratio = self.beta_x / new_beta_x
        y_ratio = self.beta_y / new_beta_y
        element_copy.components = [((x_ratio ** c.power_x) * (y_ratio ** c.power_y)) * c for c in self.components]
        element_copy.beta_x = new_beta_x
        element_copy.beta_y = new_beta_y
        return element_copy

    def __add__(self, other: Element) -> Element:
        """
        Defines the addition operator for two objects of the class Element
        :param self: The left addend
        :param other: The right addend
        :return: A new object of the class Element which represents the sum of the two addends. Its length is simply
        the sum of the length of the addends, its beta functions are a weighted, by length, sum of the beta functions
        of the addends, and its list of components include all of the components of the addends, added together where
        possible.
        """
        # Defines new attributes based on attributes of the addends
        new_length = self.length + other.length
        new_beta_x = (self.length * self.beta_x + other.length * other.beta_x) / new_length
        new_beta_y = (self.length * self.beta_y + other.length * other.beta_y) / new_length
        new_components = []

        # Pre-calculates some ratios which will be used in multiple calculations
        ratios = (self.beta_x / new_beta_x, self.beta_y / new_beta_y,
                  other.beta_x / new_beta_x, other.beta_y / new_beta_y)

        # i and j represent indices in the component lists of the left and right element respectively
        i, j = 0, 0
        # This while loop iterates through pairs of components in the component-lists of the elements until
        # one of the lists is exhausted.
        while i < len(self.components) and j < len(other.components):
            # If the two components are compatible for addition, they are weighted according to the beta functions
            if self.components[i].is_compatible(other.components[j]):
                comp1 = self.components[i]
                power_x = comp1.power_x
                power_y = comp1.power_y
                left_coefficient = (ratios[0] ** power_x) * (ratios[1] ** power_y)
                right_coefficient = (ratios[2] ** power_x) * (ratios[3] ** power_y)
                new_components.append(left_coefficient * comp1 + right_coefficient * other.components[j])
                i += 1
                j += 1
            # If the left component is "less than" the right component, by some arbitrary pre-defined measure, we know
            # that there cannot be a component in the right element compatible with our left component. Therefore, we
            # simply add the left component to the list of new components
            elif self.components[i] < other.components[j]:
                left_coefficient = (ratios[0] ** self.components[i].power_x) * (ratios[1] ** self.components[i].power_y)
                new_components.append(left_coefficient * self.components[i])
                i += 1
            else:
                right_coefficient = ((ratios[2] ** other.components[j].power_x) *
                                     (ratios[3] ** other.components[j].power_y))
                new_components.append(right_coefficient * other.components[j])
                j += 1

        # When the while-loop above exits, there could still be unprocessed components remaining in either the
        # component list of either the left- or the right element, but not both. We simply append any remaining
        # components to our new_components
        if i != len(self.components):
            for c in self.components[i:]:
                left_coefficient = (ratios[0] ** c.power_x) * (ratios[1] ** c.power_y)
                new_components.append(left_coefficient * c)
        elif j != len(other.components):
            for c in other.components[j:]:
                right_coefficient = (ratios[2] ** c.power_x) * (ratios[3] ** c.power_y)
                new_components.append(right_coefficient * c)

        # Creates and returns a new element which represents the sum of the two added elements.
        return Element(new_length, new_beta_x, new_beta_y, new_components)

    def __radd__(self, other: Union[int, Element]) -> Element:
        """
        Implements the __rad__ method for the Element class. This is only done to facilitate the syntactically
        practical use of the sum() method for Elements. sum(iterable) works by adding all of the elements of the
        iterable to 0 sequentially. Thus, the behavior of the initial 0 + iterable[0] needs to be defined. In the case
        that the left addend of any addition involving an Element is not itself an Element, the resulting sum
        is simply defined to be the right addend.
        :param other: The left addend of an addition
        :return: The sum of self and other if other is an Element, otherwise just self.
        """
        # Checks if the left addend, other, is not an Element
        if not isinstance(other, Element):
            # In which case, the right addend is simply returned
            return self

        # Otherwise, their sum is returned (by invocation of Component.__add__(self, other))
        return self + other

    def __mul__(self, scalar: float) -> Element:
        """
        Implements the __mul__ method for the Element class. Defines the behavior of multiplication of an Element by
        some scalar.
        :param scalar: A scalar value to be multiplied with some Element (cannot be complex)
        :return: A newly initialized Element which has the same beta_x and beta_y values as the self-Element, but has
        its length and all of the Components in its Component list multiplied by the scalar
        """
        # Multiplies the length of the self-element by the scalar
        new_length = self.length * scalar
        # Creates a new list of Components which is a copy of the Component list of the self-element except every
        # Component is multiplied by the scalar
        new_components = [c * scalar for c in self.components]
        # Initializes and returns a new Element with the arguments defined above
        return Element(new_length, self.beta_x, self.beta_y, new_components, self.name, self.tag, self.description)

    def __rmul__(self, scalar: float) -> Element:
        """
        Generalizes scalar multiplication of Element to be possibly from left and right. Both of these operations
        are identical.
        :param scalar: A scalar value to be multiplied with some Element
        :return: The result of calling Element.__mul__(self, scalar): A newly initialized Element which has the same
        beta_x and beta_y values as the self-Element, but has its length and all of the Components in its Component list
        multiplied by the scalar.
        """
        # Simply swaps the places of scalar and self in order to invoke the previously defined __mul__ function
        return self * scalar

    def __eq__(self, other: Element) -> bool:
        """
        Implements the __eq__ method for the Element class. Two Elements are designated as "equal" if the following
        two conditions hold:
        1. They have equal attributes within some tolerance and the length of the components-lists need to be identical.
        2. For every pair of components in the components-lists of the two elements, the two Components need to
        evaluate as equal.
        This somewhat approximated empirical approach to the equality comparator aims to compensate for small
        numerical/precision errors accumulated for two Elements which have taken different "paths" to what should
        analytically be identical Elements.
        Note that this comparator requires the evaluation of every pair of wake- and impedance functions in some
        number of points for every component in each of the elements. As such, it can be, though not extremely,
        somewhat computationally intensive, and should not be used excessively.
        :param other: The right hand side of the equality comparator
        :return: True if the two Elements have sufficiently close attributes and every pair of components evaluate as
        equal by the __eq__ method for the Component class.
        """
        # Verifies that the two elements have sufficiently close attributes and components-lists of the same length
        if not self.is_compatible(other):
            return False

        # Returns true if every pair of components in the two lists evaluate as true by the __eq__ method defined in
        # the Component class
        return all(c1 == c2 for c1, c2 in zip(self.components, other.components))

    def __str__(self):
        return f"{self.name} with parameters:\n" \
               f"Length:\t\t{self.length}\n" \
               f"Beta_x:\t\t{self.beta_x}\n" \
               f"Beta_y:\t\t{self.beta_y}\n" \
               f"#components:\t{len(self.components)}"

    def get_component(self, type_string: str):
        for comp in self.components:
            if comp.get_shorthand_type() == type_string:
                return comp

        raise KeyError(f"'{self.name}' has no component of the type '{type_string}'.")
