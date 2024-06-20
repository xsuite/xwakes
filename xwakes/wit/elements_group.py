from __future__ import annotations

import numpy as np

from .element import Element
from .component import Component, Union

from typing import List


class ElementsGroup(Element):
    """
    A class used to store many elements of the same kind (i.e. Collimators, Roman Pots, Broadband resonators...).
    Each of these groups require different handling of the input files and impedance computations (for some we use IW2D,
    while for others we simply read the wake from a file), therefore this should be used as a base class from which
    specific classes are derived.
    """
    def __init__(self, elements_list: List[Element], name: str = "Unnamed Element", tag: str = "",
                 description: str = ""):
        """
        The initialization function of the ElementsGroup class.
        :param elements_list: The list of elements in this group
        :param name: A user-specified name of the group
        :param tag: An optional keyword that can be used to help grouping different ElementGroup's
        :param description: A user-specified description of the group
        """
        if len(elements_list) == 0:
            raise ValueError('Elements_list cannot be empty')

        self.elements_list = elements_list

        sum_element = sum(elements_list)

        # Initialize the components by summing all components with the same values of (plane, source_exponents,
        # test_exponents)
        super().__init__(components=sum_element.components, name=name, tag=tag, description=description,
                         length=sum_element.length, beta_x=sum_element.beta_x, beta_y=sum_element.beta_y)

    def __add__(self, other: Union[Element, ElementsGroup]) -> ElementsGroup:
        """
        Defines the addition operator for two objects of the class ElementsGroup or for an ElementsGroup and an Element.
        :param self: The left addend
        :param other: The right addend
        :return: A new object of the class Element which represents the sum of the two addends. For two ElementsGroups
        it is an Element given by the sum of all the elements in the two groups, while for an ElementsGroup and an
        Element it is the Element given by the sum of the Element and all the elements in the ElementsGroup.
        """
        if isinstance(other, Element):
            elements_list = self.elements_list + [other]
        elif isinstance(other, ElementsGroup):
            elements_list = self.elements_list + other.elements_list
        else:
            raise TypeError("An ElementsGroup can only be summed with an ElementGroup or an Element")

        return ElementsGroup(elements_list, name=self.name, tag=self.tag, description=self.description)

    def __radd__(self, other: Union[int, Element, ElementsGroup]) -> ElementsGroup:
        """
        Implements the __rad__ method for the ElementsGroup class. This is only done to facilitate the syntactically
        practical use of the sum() method for ElementsGroup. sum(iterable) works by adding all of the elements of the
        iterable to 0 sequentially. Thus, the behavior of the initial 0 + iterable[0] needs to be defined. In the case
        that the left addend of any addition involving an Element is not itself an Element, the resulting sum
        is simply defined to be the right addend.
        :param other: The left addend of an addition
        :return: The sum of self and other if other is an ElementGroup or an Element, otherwise just self.
        """
        if type(other) == int and other != 0:
            raise ValueError("ElementsGroup right addition can only be performed with a zero addend and it is only "
                             "implemented to enable the sum on a list of ElementsGroup's")
        # Checks if the left addend, other, is not an Element
        if not (isinstance(other, ElementsGroup) or isinstance(other, Element)):
            # In which case, the right addend is simply returned
            return self

        # Otherwise, their sum is returned (by invocation of ElementsGroup.__add__(self, other))
        return self + other

    def __mul__(self, scalar: float) -> ElementsGroup:
        """
        Implements the __mul__ method for the ElementsGroup class. Defines the behavior of multiplication of an Element
        by some scalar.
        :param scalar: A scalar value to be multiplied with some Element (cannot be complex)
        :return: A newly initialized GroupElement in which every element is multiplied by the scalar.
        """
        mult_elements = []
        for element in self.elements_list:
            mult_elements.append(element * scalar)

        return ElementsGroup(mult_elements, name=self.name, tag=self.tag, description=self.description)

    def __rmul__(self, scalar: float) -> ElementsGroup:
        """
        Generalizes scalar multiplication of ElementsGroup to be possibly from left and right. Both of these operations
        are identical.
        :param scalar: A scalar value to be multiplied with some ElementsGroup
        :return: The result of calling ElementsGroup.__mul__(self, scalar): A newly initialized GroupElement in which
        every element is multiplied by the scalar.
        """
        # Simply swaps the places of scalar and self in order to invoke the previously defined __mul__ function
        return self * scalar

    def __eq__(self, other: ElementsGroup) -> bool:
        """
        Implements the __eq__ method for the Element class. Two Elements are designated as "equal" if corresponding
        elements in the elements lists are equal and if all the other non-components parameters are equal up to a small
        tolerance
        """
        if len(self.elements_list) != len(other.elements_list):
            return False

        # Verifies that the two elements have sufficiently close attributes and components-lists of the same length
        if not self.is_compatible(other):
            return False

        return all(e1 == e2 for e1, e2 in zip(self.elements_list, other.elements_list))

    def __str__(self):
        string = f"{self.name} ElementsGroup composed of the following elements\n"
        for element in self.elements_list:
            string += str(element)
            string += "============\n"
        return string

    def rotated_element(self, name: str, theta: float, rotate_beta: bool = False) -> ElementsGroup:
        """
        Returns a copy if the self-ElementsGroup in which a user-specified element has been rotated counterclockwise
        in the transverse plane by an angle of theta radians. If the rotate_beta flag is enable, the beta_x and beta_y
        values of the element in the new ElementsGroup are rotated correspondingly.
        :param name: the name of the Element to be rotated
        :param theta: The angle, in radians, the ElementsGroup is to be rotated by
        :param rotate_beta: A flag indicating whether or not the beta_x and beta_y values of the new ElementsGroup
        should also be rotated by theta. Enabled by default.
        :return: A newly initialized copy of the self-ElementsGroup which has been rotated counterclockwise in the
        transverse plane by an angle theta.
        """
        rotated_elements = self.elements_list.copy()
        found = False
        for i, element in enumerate(rotated_elements):
            if element.name == name:
                rotated_elements[i] = element.rotated(theta, rotate_beta)
                found = True

        assert found, "Element to rotate was not found in the group"

        return ElementsGroup(rotated_elements, name=self.name, tag=self.tag, description=self.description)

    def rotated(self, theta: float, rotate_beta: bool = False) -> ElementsGroup:
        """
        Returns a copy if the self-ElementsGroup which has been rotated counterclockwise in the transverse plane by an
        angle of theta radians. If the rotate_beta flag is enable, the beta_x and beta_y values of the new ElementsGroup
        are rotated correspondingly.
        :param theta: The angle, in radians, the ElementsGroup is to be rotated by
        :param rotate_beta: A flag indicating whether or not the beta_x and beta_y values of the new ElementsGroup
        should also be rotated by theta. Enabled by default.
        :return: A newly initialized copy of the self-ElementsGroup which has been rotated counterclockwise in the
        transverse plane by an angle theta.
        """
        rotated_elements = []

        for element in rotated_elements:
            rotated_elements.append(element.rotated(theta, rotate_beta))

        return ElementsGroup(rotated_elements, name=self.name, tag=self.tag, description=self.description)

    def changed_betas(self, new_beta_x: float, new_beta_y: float) -> ElementsGroup:
        elements_list_new = []
        for i_elem, elem in enumerate(self.elements_list):
            elements_list_new.append(elem.changed_betas(new_beta_x, new_beta_y))

        return ElementsGroup(elements_list_new, name=self.name, tag=self.tag, description=self.description)

    def get_element(self, name_string: str):
        for element in self.elements_list:
            if element.name == name_string:
                return element

        raise KeyError(f"'{self.name}' has no element named '{name_string}'.")

