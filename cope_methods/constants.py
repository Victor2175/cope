"""
Constant mappings and defaults for ForceSMIP methods.
"""

VARIABLE_MAP = {
    "tasmax": ("monmaxtasmax", "tasmax"),
    "tasmin": ("monmintasmin", "tasmin"),
    "prmax": ("monmaxpr", "pr"),
    "zmta": ("zmta", "ta"),
    "tas": ("tas", "tas"),
    "tos": ("tos", "tos"),
    "pr": ("pr", "pr"),
    "psl": ("psl", "psl"),
}

DEFAULT_TEST_MODELS = ["1B", "1D", "1E", "1G", "1J"]