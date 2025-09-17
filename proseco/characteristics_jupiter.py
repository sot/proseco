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
exclude_dates = [
    {"start": "2024:128:03:25:16.500", "stop": "2024:162:18:00:48.500"},
    {"start": "2025:115:15:28:12.500", "stop": "2025:246:23:43:03.500"},
    {"start": "2026:123:09:50:55.500", "stop": "2026:304:08:56:43.500"},
    {"start": "2027:139:00:21:41.500", "stop": "2027:348:07:56:35.500"},
    {"start": "2028:162:01:00:04.500", "stop": "2029:016:00:45:56.500"},
    {"start": "2029:192:13:47:11.500", "stop": "2030:041:03:26:27.500"},
    {"start": "2030:234:04:44:40.500", "stop": "2031:001:01:00:03.000"},
]
