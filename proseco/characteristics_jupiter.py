from astropy.table import Table


class JupiterPositionTable(Table):
    """
    A subclass of astropy Table for Jupiter positions.

    This is mainly to provide a specific docstring and type alias, but it does
    implicitly document that the table should have 'time', 'row', and 'col' columns.
    """

    @classmethod
    def empty(cls):
        return cls({"time": [], "row": [], "col": []})


# Date range when jupiter vmag >= -2.0 (dim) and doesn't need to be checked.
# This cycle this is 2026-05-04 to 2026-11-02.
exclude_dates = [
    {
        "start": "2026:124",
        "stop": "2026:304",
    }
]
