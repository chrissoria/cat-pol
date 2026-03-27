# SPDX-FileCopyrightText: 2025-present Christopher Soria <chrissoria@berkeley.edu>
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .san_diego import fetch_san_diego_ordinances
from .san_francisco import fetch_sf_ordinances
from .salinas import fetch_salinas_ordinances
from .legistar import fetch_oakland_ordinances, fetch_long_beach_ordinances, fetch_fresno_ordinances
from .berkeley import fetch_berkeley_ordinances
from .sd_county import fetch_sd_county_ordinances
from .federal import fetch_federal_laws
from .los_angeles import fetch_la_ordinances
from .codepublishing import fetch_clovis_code, fetch_newport_beach_code
from .truthsocial import fetch_trump_truths

__all__ = [
    "fetch_san_diego_ordinances",
    "fetch_sf_ordinances",
    "fetch_salinas_ordinances",
    "fetch_oakland_ordinances",
    "fetch_long_beach_ordinances",
    "fetch_fresno_ordinances",
    "fetch_berkeley_ordinances",
    "fetch_la_ordinances",
    "fetch_sd_county_ordinances",
    "fetch_clovis_code",
    "fetch_newport_beach_code",
    "fetch_federal_laws",
    "fetch_trump_truths",
]
