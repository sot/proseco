# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

from ..core import ACABox


def test_box_init():
    """ACABox initialization functionality"""
    box = ACABox()
    assert box.y == 20
    assert box.z == 20

    box = ACABox(None)
    assert box.y == 20
    assert box.z == 20

    box = ACABox((15, 10))
    assert box.y == 15
    assert box.z == 10

    box = ACABox([15, 10])
    assert box.y == 15
    assert box.z == 10

    box2 = ACABox(box)
    assert box2.y == 15
    assert box2.z == 10

    with pytest.raises(ValueError) as err:
        ACABox([1, 2, 3])
        assert 'size arg must be' in str(err)


def test_box_row_col_max():
    """ACABox row, col properties and max() method"""
    box = ACABox((15, 10))
    assert box.row == 3
    assert box.col == 2
    assert box.max() == 15


def test_box_eq():
    """ACABox equality and inequality"""
    dither1 = (15, 10)
    dither3 = (16, 10)
    box1 = ACABox(dither1)
    box2 = ACABox(dither1)
    box3 = ACABox(dither3)

    # ACABox to ACABox test
    assert box1 == box2
    assert box1 != box3

    # Test against left and right sides that need conversion to ACABox first
    assert dither1 == box1
    assert dither1 != box3
    assert box1 == dither1
    assert box3 != dither1


def test_box_add():
    """ACABox left and right addition"""
    box1 = ACABox((10, 15))

    box2 = box1 + 5
    assert box2 == (15, 20)

    box2 = 5 + box1
    assert box2 == (15, 20)

    box2 = (5, 10) + box1
    assert box2 == (15, 25)

    box2 = box1 + (5, 10)
    assert box2 == (15, 25)


def test_box_greater():
    """ACABox 'greater than' comparison"""
    box1 = ACABox((10, 15))

    # True if either component of box1 is bigger than the test value
    assert box1 > (9.9, 14.9)
    assert box1 > (9.9, 15)
    assert box1 > (10, 14.9)

    assert not box1 > (10, 15)
    assert not box1 > box1
    assert not box1 > (11, 16)
