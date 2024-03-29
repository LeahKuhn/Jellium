#
# @BEGIN LICENSE
#
# jellium_scf by Psi4 Developer, a plugin to:
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2017 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

import psi4
import psi4.driver.p4util as p4util
from psi4.driver.procrouting import proc_util
from psi4.driver.procrouting import proc

def run_jellium_scf(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    jellium_scf can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('jellium_scf')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    # build empty reference wavefunction to pass into plugin

    ref_molecule = kwargs.get('molecule', psi4.core.get_active_molecule())
    base_wfn = psi4.core.Wavefunction.build(ref_molecule, psi4.core.get_global_option('BASIS'))
    ref_wfn = proc.scf_wavefunction_factory('HF', base_wfn, psi4.core.get_global_option('REFERENCE'))

    jellium_scf_wfn = psi4.core.plugin('jellium_scf.so', ref_wfn)

    return jellium_scf_wfn


# Integration with driver routines
psi4.driver.procedures['energy']['jellium_scf'] = run_jellium_scf


def exampleFN():
    # Your Python code goes here
    pass
